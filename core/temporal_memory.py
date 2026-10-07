"""Provenance-first temporal memory for NEXUS.

The design borrows the useful temporal-knowledge idea found in projects such as
Graphiti while preserving JARVIS' local-first, owner-controlled architecture.
Facts are claims with provenance and validity intervals; model output never
becomes a verified fact merely because a model asserted it.

SQLite is used deliberately: no extra daemon, no vector service, and no hidden
network dependency. The schema is append-oriented so corrections preserve
history instead of silently rewriting the past.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Iterator, Mapping, Sequence
import uuid

from core.vector_index import VectorIndex


_CLAIM_ID = re.compile(r"[a-f0-9]{32}")
_SOURCE_ID = re.compile(r"[A-Za-z0-9_.:/@-]{1,240}")


class TemporalMemoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class SemanticClaimMatch:
    claim: "TemporalClaim"
    distance: float


@dataclass(frozen=True)
class TemporalClaim:
    claim_id: str
    subject: str
    predicate: str
    value: str
    source: str
    observed_at: str
    valid_from: str
    valid_to: str | None
    confidence: float
    verified: bool
    superseded_by: str | None
    metadata: Mapping[str, Any]

    @property
    def active(self) -> bool:
        return self.valid_to is None and self.superseded_by is None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _iso(value: str | datetime | None, *, default_now: bool = False) -> str | None:
    if value is None:
        return _utc_now() if default_now else None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("Timestamp must be ISO-8601.") from None
    else:
        raise ValueError("Timestamp must be ISO-8601 text or datetime.")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _text(value: str, label: str, *, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text.")
    clean = value.strip().replace("\x00", "")
    if not clean or len(clean) > limit:
        raise ValueError(f"Invalid {label}.")
    return clean


def _confidence(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence must be numeric.")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    return number


def _metadata(value: Mapping[str, Any] | None) -> str:
    payload = dict(value or {})
    try:
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be JSON-serializable.") from exc
    if len(encoded.encode("utf-8")) > 32768:
        raise ValueError("metadata is too large.")
    return encoded


class TemporalMemoryStore:
    """Durable temporal claim store with explicit provenance.

    The store never auto-verifies claims. Callers must set ``verified=True``
    only after an external/owner verification step. A correction creates a new
    claim and closes the previous claim's validity interval atomically.
    """

    def __init__(self, path: str | Path) -> None:
        db_path = Path(path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = db_path
        self._vector_index: VectorIndex | None = None
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        try:
            yield conn
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS temporal_claims (
                    claim_id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                    verified INTEGER NOT NULL CHECK(verified IN (0,1)),
                    superseded_by TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(superseded_by) REFERENCES temporal_claims(claim_id)
                );
                CREATE INDEX IF NOT EXISTS idx_temporal_active
                    ON temporal_claims(subject, predicate, valid_to, superseded_by);
                CREATE INDEX IF NOT EXISTS idx_temporal_source
                    ON temporal_claims(source, observed_at);
                CREATE INDEX IF NOT EXISTS idx_temporal_validity
                    ON temporal_claims(valid_from, valid_to);
                """
            )
            conn.commit()

    @staticmethod
    def _row(row: sqlite3.Row) -> TemporalClaim:
        try:
            metadata = json.loads(row["metadata_json"])
        except Exception:
            metadata = {}
        return TemporalClaim(
            claim_id=row["claim_id"],
            subject=row["subject"],
            predicate=row["predicate"],
            value=row["value"],
            source=row["source"],
            observed_at=row["observed_at"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            confidence=float(row["confidence"]),
            verified=bool(row["verified"]),
            superseded_by=row["superseded_by"],
            metadata=metadata if isinstance(metadata, dict) else {},
        )

    def add_claim(
        self,
        *,
        subject: str,
        predicate: str,
        value: str,
        source: str,
        observed_at: str | datetime | None = None,
        valid_from: str | datetime | None = None,
        confidence: float = 0.5,
        verified: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> TemporalClaim:
        subject = _text(subject, "subject", limit=1000)
        predicate = _text(predicate, "predicate", limit=300)
        value = _text(value, "value", limit=20000)
        source = _text(source, "source", limit=240)
        if not _SOURCE_ID.fullmatch(source):
            raise ValueError("source contains unsupported characters.")
        observed = _iso(observed_at, default_now=True)
        starts = _iso(valid_from, default_now=False) or observed
        score = _confidence(confidence)
        if not isinstance(verified, bool):
            raise ValueError("verified must be boolean.")
        metadata_json = _metadata(metadata)
        claim_id = uuid.uuid4().hex
        created = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO temporal_claims(
                    claim_id, subject, predicate, value, source, observed_at,
                    valid_from, valid_to, confidence, verified, superseded_by,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL, ?, ?)
                """,
                (
                    claim_id, subject, predicate, value, source, observed,
                    starts, score, 1 if verified else 0, metadata_json, created,
                ),
            )
            conn.commit()
        return self.get(claim_id)

    def get(self, claim_id: str) -> TemporalClaim:
        if not isinstance(claim_id, str) or not _CLAIM_ID.fullmatch(claim_id):
            raise ValueError("Invalid claim id.")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM temporal_claims WHERE claim_id = ?",
                (claim_id,),
            ).fetchone()
        if row is None:
            raise TemporalMemoryError("Temporal claim does not exist.")
        return self._row(row)

    def supersede(
        self,
        claim_id: str,
        *,
        value: str,
        source: str,
        observed_at: str | datetime | None = None,
        confidence: float = 0.5,
        verified: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> TemporalClaim:
        old = self.get(claim_id)
        if not old.active:
            raise TemporalMemoryError("Only an active claim can be superseded.")
        value = _text(value, "value", limit=20000)
        source = _text(source, "source", limit=240)
        if not _SOURCE_ID.fullmatch(source):
            raise ValueError("source contains unsupported characters.")
        observed = _iso(observed_at, default_now=True)
        score = _confidence(confidence)
        if not isinstance(verified, bool):
            raise ValueError("verified must be boolean.")
        metadata_json = _metadata(metadata)
        replacement_id = uuid.uuid4().hex
        created = _utc_now()

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT valid_to, superseded_by FROM temporal_claims WHERE claim_id = ?",
                (claim_id,),
            ).fetchone()
            if current is None or current["valid_to"] is not None or current["superseded_by"] is not None:
                conn.rollback()
                raise TemporalMemoryError("Claim changed while being superseded.")
            conn.execute(
                """
                INSERT INTO temporal_claims(
                    claim_id, subject, predicate, value, source, observed_at,
                    valid_from, valid_to, confidence, verified, superseded_by,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL, ?, ?)
                """,
                (
                    replacement_id, old.subject, old.predicate, value, source,
                    observed, observed, score, 1 if verified else 0,
                    metadata_json, created,
                ),
            )
            conn.execute(
                """
                UPDATE temporal_claims
                SET valid_to = ?, superseded_by = ?
                WHERE claim_id = ?
                """,
                (observed, replacement_id, claim_id),
            )
            conn.commit()
        return self.get(replacement_id)

    def active_claims(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        verified_only: bool = False,
        limit: int = 100,
    ) -> tuple[TemporalClaim, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000.")
        clauses = ["valid_to IS NULL", "superseded_by IS NULL"]
        args: list[Any] = []
        if subject is not None:
            clauses.append("subject = ?")
            args.append(_text(subject, "subject", limit=1000))
        if predicate is not None:
            clauses.append("predicate = ?")
            args.append(_text(predicate, "predicate", limit=300))
        if verified_only:
            clauses.append("verified = 1")
        args.append(limit)
        sql = (
            "SELECT * FROM temporal_claims WHERE " + " AND ".join(clauses)
            + " ORDER BY observed_at DESC, claim_id DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return tuple(self._row(row) for row in rows)

    def claims_at(
        self,
        moment: str | datetime,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        limit: int = 100,
    ) -> tuple[TemporalClaim, ...]:
        instant = _iso(moment)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000.")
        clauses = ["valid_from <= ?", "(valid_to IS NULL OR valid_to > ?)"]
        args: list[Any] = [instant, instant]
        if subject is not None:
            clauses.append("subject = ?")
            args.append(_text(subject, "subject", limit=1000))
        if predicate is not None:
            clauses.append("predicate = ?")
            args.append(_text(predicate, "predicate", limit=300))
        args.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM temporal_claims WHERE " + " AND ".join(clauses)
                + " ORDER BY valid_from DESC LIMIT ?",
                args,
            ).fetchall()
        return tuple(self._row(row) for row in rows)

    def history(
        self,
        *,
        subject: str,
        predicate: str | None = None,
        limit: int = 200,
    ) -> tuple[TemporalClaim, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000.")
        subject = _text(subject, "subject", limit=1000)
        args: list[Any] = [subject]
        sql = "SELECT * FROM temporal_claims WHERE subject = ?"
        if predicate is not None:
            sql += " AND predicate = ?"
            args.append(_text(predicate, "predicate", limit=300))
        sql += " ORDER BY valid_from DESC, observed_at DESC LIMIT ?"
        args.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return tuple(self._row(row) for row in rows)

    def enable_vector_index(
        self,
        dimension: int,
        *,
        prefer_native: bool = True,
        extension_path: str | Path | None = None,
    ) -> dict[str, object]:
        """Enable semantic retrieval inside this process.

        The database is deliberately separate from the temporal ledger so a
        vector index can be rebuilt without changing claim history.
        """
        vector_path = self.path.with_name(self.path.name + ".vectors")
        self._vector_index = VectorIndex(
            vector_path,
            dimension=dimension,
            prefer_native=prefer_native,
            extension_path=extension_path,
        )
        return self._vector_index.status()

    def index_claim_embedding(
        self,
        claim_id: str,
        embedding: Sequence[float],
    ) -> None:
        if self._vector_index is None:
            raise TemporalMemoryError("Vector index is not enabled.")
        claim = self.get(claim_id)
        self._vector_index.upsert(
            claim.claim_id,
            embedding,
            payload={
                "subject": claim.subject,
                "predicate": claim.predicate,
                "source": claim.source,
                "verified": claim.verified,
            },
        )

    def semantic_search(
        self,
        embedding: Sequence[float],
        *,
        active_only: bool = True,
        verified_only: bool = False,
        limit: int = 20,
    ) -> tuple[SemanticClaimMatch, ...]:
        if self._vector_index is None:
            raise TemporalMemoryError("Vector index is not enabled.")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")

        # Ask for extra candidates because stale/superseded entries are filtered
        # against the authoritative temporal ledger below.
        matches = self._vector_index.search(embedding, limit=min(100, limit * 4))
        results: list[SemanticClaimMatch] = []
        for match in matches:
            try:
                claim = self.get(match.entity_id)
            except TemporalMemoryError:
                continue
            if active_only and not claim.active:
                continue
            if verified_only and not claim.verified:
                continue
            results.append(SemanticClaimMatch(claim=claim, distance=match.distance))
            if len(results) >= limit:
                break
        return tuple(results)

    def search(
        self,
        query: str,
        *,
        active_only: bool = True,
        verified_only: bool = False,
        limit: int = 50,
    ) -> tuple[TemporalClaim, ...]:
        query = _text(query, "query", limit=500)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200.")
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        clauses = [
            "(subject LIKE ? ESCAPE '\\' OR predicate LIKE ? ESCAPE '\\' OR value LIKE ? ESCAPE '\\')"
        ]
        args: list[Any] = [pattern, pattern, pattern]
        if active_only:
            clauses.extend(["valid_to IS NULL", "superseded_by IS NULL"])
        if verified_only:
            clauses.append("verified = 1")
        args.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM temporal_claims WHERE " + " AND ".join(clauses)
                + " ORDER BY verified DESC, confidence DESC, observed_at DESC LIMIT ?",
                args,
            ).fetchall()
        return tuple(self._row(row) for row in rows)
