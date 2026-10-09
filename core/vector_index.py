"""In-process vector index owned by JARVIS core.

The native backend is the pinned sqlite-vec source assimilated under
core/native/sqlite_vec. No daemon or sidecar is required. When a compiled
loadable extension is not present, JARVIS falls back to an exact Python scan so
memory remains functional and deterministic.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import heapq
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import stat
import struct
from typing import Sequence


_MAX_ENTRIES = 100_000
_MAX_PAYLOAD_BYTES = 65_536
_MAX_STORAGE_BYTES = 512 * 1024 * 1024


class VectorIndexError(RuntimeError):
    pass


@dataclass(frozen=True)
class VectorMatch:
    entity_id: str
    distance: float
    payload: dict


def _extension_suffix() -> str:
    if os.name == "nt":
        return ".dll"
    if platform.system() == "Darwin":
        return ".dylib"
    return ".so"


def _pack(vector: Sequence[float], dimension: int) -> bytes:
    if isinstance(vector, (str, bytes, bytearray)) or len(vector) != dimension:
        raise ValueError(f"vector must contain exactly {dimension} values")
    values: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("vector values must be finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("vector values must be finite numbers")
        values.append(number)
    return struct.pack("<" + "f" * dimension, *values)


def _unpack(blob: bytes, dimension: int) -> tuple[float, ...]:
    expected = 4 * dimension
    if len(blob) != expected:
        raise VectorIndexError("stored vector has an invalid byte length")
    return struct.unpack("<" + "f" * dimension, blob)


def _l2(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


class VectorIndex:
    """Small vector index with sqlite-vec acceleration and an exact fallback."""

    def __init__(
        self,
        path: str | Path,
        *,
        dimension: int,
        prefer_native: bool = True,
        extension_path: str | Path | None = None,
    ) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or not 1 <= dimension <= 8192:
            raise ValueError("dimension must be between 1 and 8192")
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        if extension_path is not None:
            configured_extension = Path(extension_path).expanduser()
            if not configured_extension.is_absolute():
                raise ValueError("extension_path must be absolute")
            self.extension_path = configured_extension
        else:
            self.extension_path = None
        self.backend = "python-exact"
        self._initialize(prefer_native=prefer_native)

    def _candidate_extension(self) -> Path | None:
        if self.extension_path is not None:
            return self.extension_path
        native_root = Path(__file__).resolve().parent / "native" / "sqlite_vec"
        machine = platform.machine().lower() or "unknown"
        candidate = native_root / "bin" / f"{platform.system().lower()}-{machine}" / ("vec0" + _extension_suffix())
        return candidate if candidate.is_file() else None

    @staticmethod
    def _load_extension(conn: sqlite3.Connection, path: Path) -> None:
        if path.suffix.casefold() != _extension_suffix():
            raise VectorIndexError("sqlite-vec extension has an unexpected file type")
        resolved = path.resolve(strict=True)
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise VectorIndexError("sqlite-vec extension is unavailable") from exc
        reparse = bool(
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
        if path.is_symlink() or reparse or not stat.S_ISREG(info.st_mode):
            raise VectorIndexError("sqlite-vec extension must be a regular local file")
        conn.enable_load_extension(True)
        try:
            conn.load_extension(str(resolved))
        finally:
            conn.enable_load_extension(False)

    def _connect(self) -> sqlite3.Connection:
        if self._storage_bytes() > _MAX_STORAGE_BYTES:
            raise VectorIndexError("vector index exceeds its storage budget")
        conn = sqlite3.connect(str(self.path), timeout=10.0)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 10000")
            conn.execute("PRAGMA journal_size_limit = 16777216")
            conn.execute("PRAGMA wal_autocheckpoint = 256")
            page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
            maximum_pages = max(1, _MAX_STORAGE_BYTES // page_size)
            actual_limit = int(conn.execute(f"PRAGMA max_page_count = {maximum_pages}").fetchone()[0])
            if actual_limit > maximum_pages:
                raise VectorIndexError("vector index exceeds its storage budget")
            if self.backend == "sqlite-vec":
                extension = self._candidate_extension()
                if extension is None:
                    raise VectorIndexError("sqlite-vec backend lost its packaged extension")
                self._load_extension(conn, extension)
            return conn
        except BaseException:
            conn.close()
            raise

    def _initialize(self, *, prefer_native: bool) -> None:
        extension = self._candidate_extension() if prefer_native else None
        if extension is not None:
            try:
                with closing(sqlite3.connect(str(self.path), timeout=10.0)) as conn:
                    self._load_extension(conn, extension)
                    conn.execute(
                        f"CREATE VIRTUAL TABLE IF NOT EXISTS jarvis_vectors USING vec0("
                        f"entity_id TEXT PRIMARY KEY, embedding float[{self.dimension}], +payload TEXT)"
                    )
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS jarvis_vector_meta("
                        "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                    )
                    self._check_dimension(conn)
                    conn.commit()
                self.backend = "sqlite-vec"
                return
            except (OSError, sqlite3.Error):
                self.backend = "python-exact"

        with closing(sqlite3.connect(str(self.path), timeout=10.0)) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jarvis_vector_fallback (
                    entity_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jarvis_vector_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            self._check_dimension(conn)
            conn.commit()

    def _check_dimension(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT value FROM jarvis_vector_meta WHERE key='dimension'"
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO jarvis_vector_meta(key, value) VALUES('dimension', ?)",
                (str(self.dimension),),
            )
        else:
            try:
                stored_dimension = int(row[0])
            except (TypeError, ValueError, OverflowError):
                raise VectorIndexError("vector index metadata is invalid") from None
            if stored_dimension != self.dimension:
                raise VectorIndexError("vector index dimension does not match existing data")

    @staticmethod
    def _entity_id(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("entity_id must be text")
        clean = value.strip()
        if not clean or len(clean) > 512 or "\x00" in clean:
            raise ValueError("invalid entity_id")
        return clean

    @staticmethod
    def _payload(value: dict | None) -> str:
        if value is not None and not isinstance(value, dict):
            raise ValueError("payload must be a JSON object")
        try:
            encoded = json.dumps(value or {}, sort_keys=True, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("payload must be JSON-serializable") from exc
        if len(encoded.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
            raise ValueError("payload exceeds the 64 KiB limit")
        return encoded

    @staticmethod
    def _decoded_payload(value: object) -> dict:
        if value in (None, ""):
            return {}
        try:
            encoded_size = len(value.encode("utf-8")) if isinstance(value, str) else 0
        except UnicodeEncodeError:
            raise VectorIndexError("stored vector payload is invalid") from None
        if not isinstance(value, str) or encoded_size > _MAX_PAYLOAD_BYTES:
            raise VectorIndexError("stored vector payload is invalid")
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            raise VectorIndexError("stored vector payload is invalid") from None
        if not isinstance(decoded, dict):
            raise VectorIndexError("stored vector payload is not an object")
        return decoded

    def _storage_bytes(self) -> int:
        total = 0
        for candidate in (
            self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm"),
            Path(str(self.path) + "-journal"),
        ):
            try:
                total += candidate.stat().st_size
            except FileNotFoundError:
                continue
        return total

    def _ensure_insert_budget(
        self,
        conn: sqlite3.Connection,
        *,
        table: str,
        entity_id: str,
        candidate_bytes: int,
    ) -> None:
        exists = conn.execute(
            f"SELECT 1 FROM {table} WHERE entity_id = ? LIMIT 1", (entity_id,)
        ).fetchone()
        if exists is None:
            count = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            if count >= _MAX_ENTRIES:
                raise VectorIndexError("vector index reached its entry limit")
        if self._storage_bytes() + candidate_bytes > _MAX_STORAGE_BYTES:
            raise VectorIndexError("vector index write exceeds its storage budget")

    def upsert(self, entity_id: str, vector: Sequence[float], *, payload: dict | None = None) -> None:
        entity = self._entity_id(entity_id)
        packed = _pack(vector, self.dimension)
        payload_json = self._payload(payload)
        with closing(self._connect()) as conn:
            # Reserve the writer before reading quotas, including across processes.
            conn.execute("BEGIN IMMEDIATE")
            table = "jarvis_vectors" if self.backend == "sqlite-vec" else "jarvis_vector_fallback"
            self._ensure_insert_budget(
                conn,
                table=table,
                entity_id=entity,
                candidate_bytes=len(packed) + len(payload_json.encode("utf-8")) + 4096,
            )
            try:
                if self.backend == "sqlite-vec":
                    conn.execute("DELETE FROM jarvis_vectors WHERE entity_id = ?", (entity,))
                    conn.execute(
                        "INSERT INTO jarvis_vectors(entity_id, embedding, payload) VALUES (?, ?, ?)",
                        (entity, packed, payload_json),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO jarvis_vector_fallback(entity_id, embedding, payload_json)
                        VALUES (?, ?, ?)
                        ON CONFLICT(entity_id) DO UPDATE SET
                            embedding=excluded.embedding,
                            payload_json=excluded.payload_json
                        """,
                        (entity, packed, payload_json),
                    )
                # Account for pages still held in SQLite's cache before committing.
                page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
                page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
                pending_growth = max(0, page_count * page_size - self.path.stat().st_size)
                if self._storage_bytes() + pending_growth > _MAX_STORAGE_BYTES:
                    raise VectorIndexError("vector index write exceeds its storage budget")
                conn.commit()
            except sqlite3.Error as exc:
                if getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_FULL:
                    raise VectorIndexError("vector index write exceeds its storage budget") from None
                raise

    def delete(self, entity_id: str) -> bool:
        entity = self._entity_id(entity_id)
        table = "jarvis_vectors" if self.backend == "sqlite-vec" else "jarvis_vector_fallback"
        with closing(self._connect()) as conn:
            cursor = conn.execute(f"DELETE FROM {table} WHERE entity_id = ?", (entity,))
            conn.commit()
            return cursor.rowcount > 0

    def search(self, vector: Sequence[float], *, limit: int = 10) -> tuple[VectorMatch, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        packed = _pack(vector, self.dimension)
        if self.backend == "sqlite-vec":
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    """
                    SELECT entity_id, distance, payload
                    FROM jarvis_vectors
                    WHERE embedding MATCH ?
                    ORDER BY distance
                    LIMIT ?
                    """,
                    (packed, limit),
                ).fetchall()
            return tuple(
                VectorMatch(
                    entity_id=str(row["entity_id"]),
                    distance=float(row["distance"]),
                    payload=self._decoded_payload(row["payload"]),
                )
                for row in rows
            )

        query = _unpack(packed, self.dimension)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT entity_id, embedding, payload_json FROM jarvis_vector_fallback"
            )

            def matches():
                for row in rows:
                    stored = _unpack(row["embedding"], self.dimension)
                    yield VectorMatch(
                        entity_id=str(row["entity_id"]),
                        distance=_l2(query, stored),
                        payload=self._decoded_payload(row["payload_json"]),
                    )

            return tuple(
                heapq.nsmallest(
                    limit,
                    matches(),
                    key=lambda item: (item.distance, item.entity_id),
                )
            )

    def count(self) -> int:
        table = "jarvis_vectors" if self.backend == "sqlite-vec" else "jarvis_vector_fallback"
        with closing(self._connect()) as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def status(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "dimension": self.dimension,
            "count": self.count(),
            "in_process": True,
        }
