"""Single-host SQLite journal; never a source of executable authority."""
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import sqlite3
import threading
from uuid import uuid4

from core.authority_contracts import ActionRequest


class MissionState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting_confirmation"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class MissionConflict(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


class MissionStore:
    MAX_ACTIVE = 32
    MAX_TOTAL = 1000
    MAX_ATTEMPTS = 16

    def __init__(self, directory):
        # Windows file-lock/reparse certification is a separate gate.
        if os.name != "posix":
            raise RuntimeError("Durable mission host is not certified on this platform.")
        import fcntl
        self.directory = Path(directory).absolute()
        if self.directory.resolve() != self.directory:
            raise ValueError("Mission state must not traverse symbolic links.")
        if self.directory.is_relative_to(Path(__file__).resolve().parents[1]):
            raise ValueError("Mission state must be outside application source.")
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        if self.directory.stat().st_mode & 0o077:
            raise ValueError("Mission directory must be private (mode 0700).")
        self._guard = threading.RLock()
        self._closed = False
        self._lease = os.open(self.directory / "host.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.directory / "missions.sqlite"
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            os.close(fd)
            self._db = sqlite3.connect(path, timeout=2, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise RuntimeError("Unsupported mission schema version.")
            with self._db:
                self._db.execute("""CREATE TABLE IF NOT EXISTS missions (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, goal TEXT NOT NULL,
                    tool TEXT NOT NULL, capability TEXT NOT NULL, arguments TEXT NOT NULL,
                    digest TEXT NOT NULL, request_id TEXT NOT NULL UNIQUE, route TEXT NOT NULL,
                    state TEXT NOT NULL, version INTEGER NOT NULL, attempts INTEGER NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, receipt TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0)""")
                self._db.execute("CREATE INDEX IF NOT EXISTS mission_owner ON missions(owner, created_at)")
                self._db.execute("""CREATE TABLE IF NOT EXISTS events (
                    mission_id TEXT NOT NULL, sequence INTEGER NOT NULL, state TEXT NOT NULL,
                    at TEXT NOT NULL, PRIMARY KEY(mission_id, sequence))""")
                self._db.execute("PRAGMA user_version=1")
                # Exclusive host lease established first; no other host is live.
                for row in self._db.execute("SELECT id, version FROM missions WHERE state='running'").fetchall():
                    stamp = now()
                    self._db.execute("UPDATE missions SET state='unknown', version=version+1, updated_at=? WHERE id=?", (stamp, row["id"]))
                    self._event(row["id"], row["version"] + 1, "unknown", stamp)
        except Exception:
            if hasattr(self, "_db"):
                self._db.close()
            os.close(self._lease)
            raise

    def _event(self, mid, sequence, state, stamp):
        self._db.execute("INSERT INTO events VALUES (?,?,?,?)", (mid, sequence, state, stamp))

    def close(self):
        with self._guard:
            if not self._closed:
                self._db.close()
                os.close(self._lease)
                self._closed = True

    def _row(self, owner, mid):
        row = self._db.execute("SELECT * FROM missions WHERE id=? AND owner=?", (mid, owner)).fetchone()
        if row is None:
            raise KeyError("Mission not found")
        return dict(row)

    def get(self, owner, mid):
        with self._guard:
            return self._row(owner, mid)

    def list(self, owner):
        with self._guard:
            return [dict(r) for r in self._db.execute(
                "SELECT id, goal, state, version, created_at, updated_at FROM missions WHERE owner=? ORDER BY created_at DESC LIMIT 50", (owner,))]

    def events(self, owner, mid):
        with self._guard:
            self._row(owner, mid)
            return [dict(r) for r in self._db.execute("SELECT sequence,state,at FROM events WHERE mission_id=? ORDER BY sequence", (mid,))]

    def create(self, owner, goal, tool, request: ActionRequest, route):
        if not isinstance(goal, str) or not 0 < len(goal) <= 4000:
            raise ValueError("Mission goal must contain 1–4000 characters.")
        if not isinstance(owner, str) or not 0 < len(owner) <= 128 or route not in {"model", "reflex", "owner"}:
            raise ValueError("Invalid host identity or route.")
        with self._guard, self._db:
            count = self._db.execute("SELECT COUNT(*) FROM missions WHERE owner=?", (owner,)).fetchone()[0]
            active = self._db.execute("SELECT COUNT(*) FROM missions WHERE owner=? AND state IN ('queued','running','waiting_confirmation')", (owner,)).fetchone()[0]
            total = self._db.execute("SELECT COUNT(*) FROM missions").fetchone()[0]
            if count >= self.MAX_TOTAL or active >= self.MAX_ACTIVE or total >= 10000:
                raise MissionConflict("Mission storage quota reached; owner maintenance required.")
            mid, stamp = str(uuid4()), now()
            self._db.execute("""INSERT INTO missions
                (id,owner,goal,tool,capability,arguments,digest,request_id,route,state,version,attempts,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,'queued',0,0,?,?)""",
                (mid, owner, goal, tool, request.capability_id, request.arguments_json,
                 request.arguments_digest, request.request_id, route, stamp, stamp))
            self._event(mid, 0, "queued", stamp)
        return mid

    def claim(self, owner, mid, version):
        with self._guard, self._db:
            row = self._row(owner, mid)
            if row["version"] != version or row["state"] not in {"queued", "waiting_confirmation"}:
                raise MissionConflict("Mission state changed or is not executable.")
            if row["attempts"] >= self.MAX_ATTEMPTS:
                raise MissionConflict("Mission attempt budget exhausted.")
            stamp = now()
            self._db.execute("UPDATE missions SET state='running', version=version+1, attempts=attempts+1, updated_at=? WHERE id=?", (stamp, mid))
            self._event(mid, version + 1, "running", stamp)
            return self._row(owner, mid)

    def complete(self, owner, mid, receipt):
        payload = json.dumps(receipt.to_dict(), ensure_ascii=False, allow_nan=False)
        if len(payload.encode()) > 262144:
            raise ValueError("Receipt exceeds journal budget.")
        state = {"require_confirmation": "waiting_confirmation", "unverified": "unknown"}.get(receipt.result.status.value, receipt.result.status.value)
        with self._guard, self._db:
            row = self._row(owner, mid)
            if (row["state"] != "running" or receipt.task_id != mid
                    or receipt.request_id != row["request_id"]
                    or receipt.normalized_argument_digest != row["digest"]):
                raise MissionConflict("Receipt does not match claimed action.")
            stamp = now()
            self._db.execute("UPDATE missions SET state=?, receipt=?, version=version+1, updated_at=? WHERE id=?", (state, payload, stamp, mid))
            self._event(mid, row["version"] + 1, state, stamp)

    def cancel(self, owner, mid, version):
        with self._guard, self._db:
            row = self._row(owner, mid)
            if row["version"] != version or row["state"] not in {"queued", "waiting_confirmation", "running"}:
                raise MissionConflict("Mission state changed or is already terminal.")
            if row["cancel_requested"]:
                return row
            state = row["state"] if row["state"] == "running" else "cancelled"
            stamp = now()
            self._db.execute("UPDATE missions SET state=?, cancel_requested=1, version=version+1, updated_at=? WHERE id=?", (state, stamp, mid))
            self._event(mid, row["version"] + 1, state, stamp)
            return self._row(owner, mid)
