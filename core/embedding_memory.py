"""Host-owned bridge between local embeddings and temporal memory.

This layer gives NEXUS text-to-vector memory without teaching the temporal
ledger about any specific ML runtime. Embeddings remain evidence/retrieval
material only and never grant action authority.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from core.hybrid_retrieval import HybridClaimMatch, TemporalHybridRetriever
from core.local_embedding import EmbeddingInfo, EmbeddingProvider
from core.temporal_memory import SemanticClaimMatch, TemporalClaim, TemporalMemoryStore


class TemporalEmbeddingMemory:
    """Generate and index temporal-memory embeddings entirely in-process."""

    def __init__(
        self,
        store: TemporalMemoryStore,
        provider: EmbeddingProvider,
        *,
        prefer_native: bool = True,
        extension_path: str | Path | None = None,
    ) -> None:
        if not isinstance(store, TemporalMemoryStore):
            raise TypeError("store must be a TemporalMemoryStore")
        info = provider.info
        if not isinstance(info, EmbeddingInfo):
            raise TypeError("provider.info must be EmbeddingInfo")
        if not 1 <= info.dimension <= 8192:
            raise ValueError("embedding provider dimension is unsupported")
        self.store = store
        self.provider = provider
        self._vector_status = store.enable_vector_index(
            info.dimension,
            prefer_native=prefer_native,
            extension_path=extension_path,
        )
        self._hybrid = TemporalHybridRetriever(store)

    @staticmethod
    def _claim_text(claim: TemporalClaim) -> str:
        return f"{claim.subject} {claim.predicate} {claim.value}"

    def status(self) -> dict[str, object]:
        info = self.provider.info
        return {
            "configured": True,
            "engine": info.engine,
            "model_id": info.model_id,
            "dimension": info.dimension,
            "normalized": info.normalized,
            "network_required": False,
            "vector_index": dict(self._vector_status),
        }

    def index_claim(self, claim_id: str) -> None:
        claim = self.store.get(claim_id)
        vector = self.provider.embed(self._claim_text(claim))
        if len(vector) != self.provider.info.dimension:
            raise ValueError("embedding provider returned an unexpected dimension")
        self.store.index_claim_embedding(claim.claim_id, vector)

    def index_claims(self, claim_ids: Sequence[str], *, batch_size: int = 32) -> int:
        if isinstance(claim_ids, (str, bytes)) or not isinstance(claim_ids, Sequence):
            raise ValueError("claim_ids must be a sequence")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be between 1 and 64")
        if len(claim_ids) > 1000:
            raise ValueError("at most 1000 claims may be indexed at once")

        indexed = 0
        for offset in range(0, len(claim_ids), batch_size):
            batch_ids = claim_ids[offset : offset + batch_size]
            claims = [self.store.get(claim_id) for claim_id in batch_ids]
            vectors = self.provider.embed_texts([self._claim_text(claim) for claim in claims])
            if len(vectors) != len(claims):
                raise ValueError("embedding provider returned an unexpected batch size")
            for claim, vector in zip(claims, vectors):
                if len(vector) != self.provider.info.dimension:
                    raise ValueError("embedding provider returned an unexpected dimension")
                self.store.index_claim_embedding(claim.claim_id, vector)
                indexed += 1
        return indexed

    def index_active_claims(
        self,
        *,
        verified_only: bool = False,
        limit: int = 1000,
        batch_size: int = 32,
    ) -> int:
        claims = self.store.active_claims(verified_only=verified_only, limit=limit)
        return self.index_claims([claim.claim_id for claim in claims], batch_size=batch_size)

    def semantic_search_text(
        self,
        query: str,
        *,
        active_only: bool = True,
        verified_only: bool = False,
        limit: int = 20,
    ) -> tuple[SemanticClaimMatch, ...]:
        vector = self.provider.embed(query)
        return self.store.semantic_search(
            vector,
            active_only=active_only,
            verified_only=verified_only,
            limit=limit,
        )

    def hybrid_search_text(
        self,
        query: str,
        *,
        verified_only: bool = False,
        limit: int = 20,
        lexical_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ) -> tuple[HybridClaimMatch, ...]:
        vector = self.provider.embed(query)
        return self._hybrid.hybrid_search(
            query,
            vector,
            verified_only=verified_only,
            limit=limit,
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
        )
