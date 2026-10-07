"""In-process vector index owned by JARVIS core.

The native backend is the pinned sqlite-vec source assimilated under
core/native/sqlite_vec. No daemon or sidecar is required. When a compiled
loadable extension is not present, JARVIS falls back to an exact Python scan so
memory remains functional and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import struct
from typing import Sequence


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
        self.extension_path = Path(extension_path).expanduser() if extension_path else None
        self.backend = "python-exact"
        self._initialize(prefer_native=prefer_native)

    def _candidate_extension(self) -> Path | None:
        if self.extension_path is not None:
            return self.extension_path
        configured = os.environ.get("JARVIS_SQLITE_VEC_PATH", "").strip()
        if configured:
            return Path(configured).expanduser()
        native_root = Path(__file__).resolve().parent / "native" / "sqlite_vec"
        machine = platform.machine().lower() or "unknown"
        candidate = native_root / "bin" / f"{platform.system().lower()}-{machine}" / ("vec0" + _extension_suffix())
        return candidate if candidate.is_file() else None

    @staticmethod
    def _load_extension(conn: sqlite3.Connection, path: Path) -> None:
        resolved = path.resolve(strict=True)
        conn.enable_load_extension(True)
        try:
            conn.load_extension(str(resolved))
        finally:
            conn.enable_load_extension(False)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        if self.backend == "sqlite-vec":
            extension = self._candidate_extension()
            if extension is None:
                conn.close()
                raise VectorIndexError("sqlite-vec backend lost its packaged extension")
            self._load_extension(conn, extension)
        return conn

    def _initialize(self, *, prefer_native: bool) -> None:
        extension = self._candidate_extension() if prefer_native else None
        if extension is not None:
            try:
                with sqlite3.connect(str(self.path), timeout=10.0) as conn:
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
                self.backend = "sqlite-vec"
                return
            except (OSError, sqlite3.Error):
                self.backend = "python-exact"

        with sqlite3.connect(str(self.path), timeout=10.0) as conn:
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

    def _check_dimension(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT value FROM jarvis_vector_meta WHERE key='dimension'"
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO jarvis_vector_meta(key, value) VALUES('dimension', ?)",
                (str(self.dimension),),
            )
        elif int(row[0]) != self.dimension:
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
        try:
            return json.dumps(dict(value or {}), sort_keys=True, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("payload must be JSON-serializable") from exc

    def upsert(self, entity_id: str, vector: Sequence[float], *, payload: dict | None = None) -> None:
        entity = self._entity_id(entity_id)
        packed = _pack(vector, self.dimension)
        payload_json = self._payload(payload)
        with self._connect() as conn:
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
            conn.commit()

    def delete(self, entity_id: str) -> bool:
        entity = self._entity_id(entity_id)
        table = "jarvis_vectors" if self.backend == "sqlite-vec" else "jarvis_vector_fallback"
        with self._connect() as conn:
            cursor = conn.execute(f"DELETE FROM {table} WHERE entity_id = ?", (entity,))
            conn.commit()
            return cursor.rowcount > 0

    def search(self, vector: Sequence[float], *, limit: int = 10) -> tuple[VectorMatch, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        packed = _pack(vector, self.dimension)
        if self.backend == "sqlite-vec":
            with self._connect() as conn:
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
                    payload=json.loads(row["payload"]) if row["payload"] else {},
                )
                for row in rows
            )

        query = _unpack(packed, self.dimension)
        scored: list[VectorMatch] = []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT entity_id, embedding, payload_json FROM jarvis_vector_fallback"
            ).fetchall()
        for row in rows:
            stored = _unpack(row["embedding"], self.dimension)
            scored.append(
                VectorMatch(
                    entity_id=str(row["entity_id"]),
                    distance=_l2(query, stored),
                    payload=json.loads(row["payload_json"]) if row["payload_json"] else {},
                )
            )
        scored.sort(key=lambda item: (item.distance, item.entity_id))
        return tuple(scored[:limit])

    def count(self) -> int:
        table = "jarvis_vectors" if self.backend == "sqlite-vec" else "jarvis_vector_fallback"
        with self._connect() as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def status(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "dimension": self.dimension,
            "count": self.count(),
            "in_process": True,
        }
