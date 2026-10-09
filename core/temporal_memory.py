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
import os
from pathlib import Path
import re
import sqlite3
import stat
from typing import Any, Iterable, Iterator, Mapping, Sequence
import uuid

from core.vector_index import VectorIndex


_CLAIM_ID = re.compile(r"[a-f0-9]{32}")
_SOURCE_ID = re.compile(r"[A-Za-z0-9_.:/@-]{1,240}")
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_STATE_SIDECARS = ("", "-wal", "-shm", "-journal")

# Temporal memory is intentionally a small, owner-local evidence ledger.  These
# limits keep malformed/imported content from turning it into an unbounded disk
# sink.  The SQLite page limit is also applied to every connection below.
DEFAULT_MAX_CLAIMS = 100_000
DEFAULT_MAX_STORAGE_BYTES = 512 * 1024 * 1024
DEFAULT_WRITE_RESERVE_BYTES = 128 * 1024


class TemporalMemoryError(RuntimeError):
    pass


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _looks_like_unc(value: str) -> bool:
    normalized = value.replace("/", "\\")
    return normalized.startswith("\\\\")


def _reject_indirect_components(path: Path) -> None:
    """Reject symlinks and Windows junctions in every existing component."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink() or _is_reparse_point(current):
            raise TemporalMemoryError(
                "Temporal memory state must not traverse symlinks or reparse points."
            )


def _reject_non_local_path(path: Path) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        drive_type = ctypes.windll.kernel32.GetDriveTypeW(str(Path(path.anchor)))
    except (AttributeError, OSError, ValueError) as exc:
        raise TemporalMemoryError(
            "Could not verify that temporal memory uses a local fixed drive."
        ) from exc
    # DRIVE_FIXED is the only location certified for durable owner state.
    if drive_type != 3:
        raise TemporalMemoryError("Temporal memory must use a local fixed drive.")


def _validated_state_path(value: str | Path) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise TemporalMemoryError("Temporal memory path must be filesystem text.") from exc
    if not isinstance(raw, str) or not raw or "\x00" in raw or _looks_like_unc(raw):
        raise TemporalMemoryError("Temporal memory requires a local absolute path.")
    try:
        candidate = Path(raw).expanduser()
    except (OSError, RuntimeError, ValueError) as exc:
        raise TemporalMemoryError("Temporal memory path is invalid.") from exc
    if not candidate.is_absolute():
        raise TemporalMemoryError("Temporal memory requires an absolute path.")
    absolute = candidate.absolute()
    _reject_indirect_components(absolute)
    try:
        resolved = absolute.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise TemporalMemoryError("Temporal memory path cannot be resolved safely.") from exc
    if resolved != absolute:
        raise TemporalMemoryError(
            "Temporal memory state must not be redirected through filesystem links."
        )
    if resolved == _SOURCE_ROOT or resolved.is_relative_to(_SOURCE_ROOT):
        raise TemporalMemoryError("Temporal memory state must be outside application source.")
    _reject_non_local_path(resolved)
    return resolved


def _harden_windows_path(path: Path, *, directory: bool) -> None:
    """Allow only the current Windows user and LocalSystem to access state."""
    try:
        import ntsecuritycon
        import win32api
        import win32con
        import win32security

        token = win32security.OpenProcessToken(
            win32api.GetCurrentProcess(), win32con.TOKEN_QUERY
        )
        owner = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        system = win32security.CreateWellKnownSid(
            win32security.WinLocalSystemSid, None
        )
        inheritance = (
            win32security.CONTAINER_INHERIT_ACE
            | win32security.OBJECT_INHERIT_ACE
            if directory
            else 0
        )
        acl = win32security.ACL()
        for sid in (owner, system):
            acl.AddAccessAllowedAceEx(
                win32security.ACL_REVISION_DS,
                inheritance,
                ntsecuritycon.FILE_ALL_ACCESS,
                sid,
            )
        win32security.SetNamedSecurityInfo(
            str(path),
            win32security.SE_FILE_OBJECT,
            win32security.DACL_SECURITY_INFORMATION
            | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
            None,
            None,
            acl,
            None,
        )
    except (ImportError, OSError, AttributeError) as exc:
        raise TemporalMemoryError(
            "Could not establish private Windows temporal-memory state."
        ) from exc


def _prepare_private_directory(path: Path) -> None:
    _reject_indirect_components(path)
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise TemporalMemoryError("Could not create temporal-memory state directory.") from exc
    _reject_indirect_components(path)
    try:
        info = path.lstat()
    except OSError as exc:
        raise TemporalMemoryError("Temporal-memory state directory is inaccessible.") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise TemporalMemoryError("Temporal-memory state parent must be a directory.")
    if os.name == "posix":
        if info.st_uid != os.geteuid():
            raise TemporalMemoryError("Temporal-memory state directory must be owner-controlled.")
        try:
            path.chmod(0o700)
        except OSError as exc:
            raise TemporalMemoryError("Could not make temporal-memory state private.") from exc
        if stat.S_IMODE(path.stat().st_mode) != 0o700:
            raise TemporalMemoryError("Temporal-memory directory must use mode 0700.")
    elif os.name == "nt":
        _harden_windows_path(path, directory=True)
    else:
        raise TemporalMemoryError(
            "Temporal-memory persistence is not certified on this platform."
        )


def _protect_private_file(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise TemporalMemoryError("Temporal-memory state file disappeared.") from None
    except OSError as exc:
        raise TemporalMemoryError("Temporal-memory state file is inaccessible.") from exc
    if (
        path.is_symlink()
        or _is_reparse_point(path)
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
    ):
        raise TemporalMemoryError(
            "Temporal-memory state must be a single-link regular local file."
        )
    if os.name == "posix":
        try:
            path.chmod(0o600)
        except OSError as exc:
            raise TemporalMemoryError("Could not make temporal-memory file private.") from exc
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise TemporalMemoryError("Temporal-memory files must use mode 0600.")
    elif os.name == "nt":
        _harden_windows_path(path, directory=False)
    else:
        raise TemporalMemoryError(
            "Temporal-memory persistence is not certified on this platform."
        )


def _prepare_private_file(path: Path) -> None:
    if path.exists() or path.is_symlink() or _is_reparse_point(path):
        _protect_private_file(path)
        return
    flags = os.O_CREAT | os.O_RDWR
    if os.name == "posix":
        flags |= os.O_NOFOLLOW
    elif os.name == "nt":
        flags |= os.O_BINARY | getattr(os, "O_NOINHERIT", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise TemporalMemoryError("Could not create temporal-memory state file.") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise TemporalMemoryError(
                "Temporal-memory state must be a single-link regular local file."
            )
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)
    _protect_private_file(path)


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

    MAX_CLAIMS = DEFAULT_MAX_CLAIMS
    MAX_STORAGE_BYTES = DEFAULT_MAX_STORAGE_BYTES
    WRITE_RESERVE_BYTES = DEFAULT_WRITE_RESERVE_BYTES

    @staticmethod
    def validate_path(path: str | Path) -> Path:
        """Return a canonical owner-local state path without creating it."""
        return _validated_state_path(path)

    def __init__(self, path: str | Path) -> None:
        self.path = self.validate_path(path)
        _prepare_private_directory(self.path.parent)
        _prepare_private_file(self.path)
        self._vector_index: VectorIndex | None = None
        self._assert_storage_budget(self.WRITE_RESERVE_BYTES)
        self._initialize()

    @staticmethod
    def _file_family(path: Path) -> tuple[Path, ...]:
        return tuple(Path(str(path) + suffix) for suffix in _STATE_SIDECARS)

    @classmethod
    def _protect_file_family(cls, path: Path) -> None:
        for candidate in cls._file_family(path):
            if candidate.exists() or candidate.is_symlink() or _is_reparse_point(candidate):
                _protect_private_file(candidate)

    def _state_storage_bytes(self) -> int:
        total = 0
        # The shared-memory file is fixed-size coordination state, while the DB
        # and active WAL are the durable/growing portions of this budget.
        for candidate in (self.path, Path(str(self.path) + "-wal")):
            if candidate.exists() or candidate.is_symlink() or _is_reparse_point(candidate):
                _protect_private_file(candidate)
                try:
                    total += candidate.stat().st_size
                except OSError as exc:
                    raise TemporalMemoryError(
                        "Could not measure temporal-memory storage usage."
                    ) from exc
        return total

    def _assert_storage_budget(self, reserve: int = 0) -> None:
        maximum = self.MAX_STORAGE_BYTES
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or not 1024 * 1024 <= maximum <= DEFAULT_MAX_STORAGE_BYTES
        ):
            raise TemporalMemoryError("Temporal-memory storage budget is invalid.")
        if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve < 0:
            raise TemporalMemoryError("Temporal-memory write reserve is invalid.")
        if self._state_storage_bytes() + reserve > maximum:
            raise TemporalMemoryError("Temporal-memory storage budget is exhausted.")

    def _assert_write_capacity(self, conn: sqlite3.Connection) -> None:
        maximum = self.MAX_CLAIMS
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or not 1 <= maximum <= DEFAULT_MAX_CLAIMS
        ):
            raise TemporalMemoryError("Temporal-memory claim quota is invalid.")
        self._assert_storage_budget(self.WRITE_RESERVE_BYTES)
        count = int(conn.execute("SELECT COUNT(*) FROM temporal_claims").fetchone()[0])
        if count >= maximum:
            raise TemporalMemoryError("Temporal-memory claim quota is exhausted.")

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute("PRAGMA synchronous = FULL")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 0")
        conn.execute("PRAGMA wal_autocheckpoint = 256")
        journal_limit = min(16 * 1024 * 1024, max(1024 * 1024, self.MAX_STORAGE_BYTES // 16))
        conn.execute(f"PRAGMA journal_size_limit = {journal_limit}")
        page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
        maximum_pages = max(1, self.MAX_STORAGE_BYTES // page_size)
        conn.execute(f"PRAGMA max_page_count = {maximum_pages}")
        if hasattr(conn, "setlimit"):
            conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1024 * 1024)
            conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 128 * 1024)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._protect_file_family(self.path)
        try:
            conn = sqlite3.connect(str(self.path), timeout=10.0)
        except sqlite3.Error as exc:
            raise TemporalMemoryError("Could not open temporal-memory state.") from exc
        try:
            self._configure_connection(conn)
            yield conn
        finally:
            conn.close()
            self._protect_file_family(self.path)

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
        self._assert_storage_budget()

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
            conn.execute("BEGIN IMMEDIATE")
            self._assert_write_capacity(conn)
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
        if observed < old.valid_from:
            raise TemporalMemoryError(
                "A replacement claim cannot predate the claim it supersedes."
            )
        score = _confidence(confidence)
        if not isinstance(verified, bool):
            raise ValueError("verified must be boolean.")
        metadata_json = _metadata(metadata)
        replacement_id = uuid.uuid4().hex
        created = _utc_now()

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._assert_write_capacity(conn)
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
        vector_path = self.validate_path(self.path.with_name(self.path.name + ".vectors"))
        _prepare_private_file(vector_path)
        try:
            self._vector_index = VectorIndex(
                vector_path,
                dimension=dimension,
                prefer_native=prefer_native,
                extension_path=extension_path,
            )
        finally:
            self._protect_file_family(vector_path)
        return self._vector_index.status()

    def index_claim_embedding(
        self,
        claim_id: str,
        embedding: Sequence[float],
    ) -> None:
        if self._vector_index is None:
            raise TemporalMemoryError("Vector index is not enabled.")
        claim = self.get(claim_id)
        self._protect_file_family(self._vector_index.path)
        try:
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
        finally:
            self._protect_file_family(self._vector_index.path)

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
        self._protect_file_family(self._vector_index.path)
        try:
            matches = self._vector_index.search(embedding, limit=min(100, limit * 4))
        finally:
            self._protect_file_family(self._vector_index.path)
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
