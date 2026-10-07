"""Hybrid lexical + semantic retrieval for NEXUS temporal memory.

The lexical scorer is assimilated from the pinned Apache-2.0 rank_bm25
snapshot under ``core/native/rank_bm25``.  JARVIS owns the retrieval contract,
filtering, fusion and temporal-ledger semantics around that implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from core.native.rank_bm25 import BM25Plus
from core.temporal_memory import TemporalClaim, TemporalMemoryStore


_TOKEN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


@dataclass(frozen=True)
class LexicalClaimMatch:
    claim: TemporalClaim
    score: float


@dataclass(frozen=True)
class HybridClaimMatch:
    claim: TemporalClaim
    fused_score: float
    lexical_score: float | None
    semantic_distance: float | None


def _tokens(text: str) -> list[str]:
    return [item.casefold() for item in _TOKEN.findall(text)]


class TemporalHybridRetriever:
    """Bounded in-process retrieval over the authoritative temporal ledger.

    BM25 supplies exact/token relevance while the existing VectorIndex supplies
    semantic similarity. Reciprocal-rank fusion deliberately combines *ranks*
    rather than incomparable raw BM25 and vector-distance scales.
    """

    def __init__(
        self,
        store: TemporalMemoryStore,
        *,
        lexical_candidate_limit: int = 1000,
        rrf_k: int = 60,
    ) -> None:
        if not isinstance(store, TemporalMemoryStore):
            raise TypeError("store must be a TemporalMemoryStore")
        if isinstance(lexical_candidate_limit, bool) or not 1 <= lexical_candidate_limit <= 1000:
            raise ValueError("lexical_candidate_limit must be between 1 and 1000")
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or not 1 <= rrf_k <= 1000:
            raise ValueError("rrf_k must be between 1 and 1000")
        self.store = store
        self.lexical_candidate_limit = lexical_candidate_limit
        self.rrf_k = rrf_k

    @staticmethod
    def _claim_text(claim: TemporalClaim) -> str:
        return f"{claim.subject} {claim.predicate} {claim.value}"

    def lexical_search(
        self,
        query: str,
        *,
        verified_only: bool = False,
        limit: int = 20,
    ) -> tuple[LexicalClaimMatch, ...]:
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("query must be non-empty text up to 500 characters")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        query_tokens = _tokens(query)
        if not query_tokens:
            return ()

        claims = self.store.active_claims(
            verified_only=verified_only,
            limit=self.lexical_candidate_limit,
        )
        if not claims:
            return ()
        corpus = [_tokens(self._claim_text(claim)) for claim in claims]
        # BM25Plus keeps IDF positive and behaves predictably for tiny personal
        # corpora where a useful token may appear in more than half the claims.
        ranker = BM25Plus(corpus)
        scores = ranker.get_scores(query_tokens)
        ordered = sorted(
            range(len(claims)),
            key=lambda idx: (float(scores[idx]), claims[idx].confidence, claims[idx].observed_at),
            reverse=True,
        )
        matches: list[LexicalClaimMatch] = []
        for idx in ordered:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            matches.append(LexicalClaimMatch(claim=claims[idx], score=score))
            if len(matches) >= limit:
                break
        return tuple(matches)

    def hybrid_search(
        self,
        query: str,
        embedding: Sequence[float],
        *,
        verified_only: bool = False,
        limit: int = 20,
        lexical_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ) -> tuple[HybridClaimMatch, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        for name, value in (("lexical_weight", lexical_weight), ("semantic_weight", semantic_weight)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0.0:
                raise ValueError(f"{name} must be a non-negative number")
        total_weight = float(lexical_weight) + float(semantic_weight)
        if total_weight <= 0.0:
            raise ValueError("at least one retrieval weight must be positive")
        lexical_weight = float(lexical_weight) / total_weight
        semantic_weight = float(semantic_weight) / total_weight

        candidate_limit = min(100, max(limit * 4, limit))
        lexical = self.lexical_search(query, verified_only=verified_only, limit=candidate_limit)
        semantic = self.store.semantic_search(
            embedding,
            active_only=True,
            verified_only=verified_only,
            limit=candidate_limit,
        )

        by_id: dict[str, dict[str, object]] = {}
        for rank, match in enumerate(lexical, start=1):
            entry = by_id.setdefault(match.claim.claim_id, {"claim": match.claim, "score": 0.0})
            entry["score"] = float(entry["score"]) + lexical_weight / (self.rrf_k + rank)
            entry["lexical_score"] = match.score
        for rank, match in enumerate(semantic, start=1):
            entry = by_id.setdefault(match.claim.claim_id, {"claim": match.claim, "score": 0.0})
            entry["score"] = float(entry["score"]) + semantic_weight / (self.rrf_k + rank)
            entry["semantic_distance"] = match.distance

        ordered = sorted(
            by_id.values(),
            key=lambda entry: (
                float(entry["score"]),
                bool(entry["claim"].verified),
                float(entry["claim"].confidence),
                entry["claim"].observed_at,
            ),
            reverse=True,
        )
        return tuple(
            HybridClaimMatch(
                claim=entry["claim"],
                fused_score=float(entry["score"]),
                lexical_score=(
                    float(entry["lexical_score"]) if "lexical_score" in entry else None
                ),
                semantic_distance=(
                    float(entry["semantic_distance"]) if "semantic_distance" in entry else None
                ),
            )
            for entry in ordered[:limit]
        )
