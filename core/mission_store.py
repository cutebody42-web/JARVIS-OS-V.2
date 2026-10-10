"""Single-host SQLite journal; never a source of executable authority."""
from datetime import datetime, timezone
from enum import Enum
import errno
import json
import os
from pathlib import Path
import sqlite3
import stat
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


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _reject_windows_reparse_path(path: Path) -> None:
    """Reject junctions/symlinks at every existing component of a state path."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if _is_reparse_point(current):
            raise ValueError("Mission state must not traverse Windows reparse points.")


def _harden_windows_directory(path: Path) -> None:
    """Protect mission state so only this Windows user and SYSTEM inherit access."""
    try:
        import ntsecuritycon
        import win32api
        import win32con
        import win32security

        token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
        owner = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        system = win32security.CreateWellKnownSid(win32security.WinLocalSystemSid, None)
        acl = win32security.ACL()
        inheritance = win32security.CONTAINER_INHERIT_ACE | win32security.OBJECT_INHERIT_ACE
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
    except (ImportError, OSError) as exc:
        raise RuntimeError("Could not establish a private Windows mission directory.") from exc


def _open_host_lease(path: Path) -> int:
    if os.name == "posix":
        import fcntl

        lease = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lease
        except Exception:
            os.close(lease)
            raise

    if os.name == "nt":
        import msvcrt

        if _is_reparse_point(path):
            raise ValueError("Mission host lease must not be a Windows reparse point.")
        lease = os.open(path, os.O_CREAT | os.O_RDWR | os.O_BINARY)
        try:
            if not stat.S_ISREG(os.fstat(lease).st_mode) or _is_reparse_point(path):
                raise ValueError("Mission host lease must be a regular local file.")
            if os.fstat(lease).st_size == 0:
                os.write(lease, b"\0")
                os.fsync(lease)
            os.lseek(lease, 0, os.SEEK_SET)
            try:
                msvcrt.locking(lease, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EDEADLK, errno.EAGAIN}:
                    raise BlockingIOError(errno.EWOULDBLOCK, "Mission host lease is already held.") from exc
                raise
            return lease
        except Exception:
            os.close(lease)
            raise

    raise RuntimeError("Durable mission host is not certified on this platform.")


def _close_host_lease(lease: int) -> None:
    if os.name == "nt":
        import msvcrt

        try:
            os.lseek(lease, 0, os.SEEK_SET)
            msvcrt.locking(lease, msvcrt.LK_UNLCK, 1)
        except OSError:
            # Closing the descriptor still releases a Windows byte-range lock.
            pass
    os.close(lease)


def _create_database_file(path: Path) -> None:
    if os.name == "posix":
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    elif os.name == "nt":
        if _is_reparse_point(path):
            raise ValueError("Mission database must not be a Windows reparse point.")
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_BINARY)
    else:
        raise RuntimeError("Durable mission host is not certified on this platform.")
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Mission database must be a regular local file.")
        if os.name == "nt" and _is_reparse_point(path):
            raise ValueError("Mission database must not be a Windows reparse point.")
    finally:
        os.close(fd)


class MissionStore:
    MAX_ACTIVE = 32
    MAX_TOTAL = 1000
    MAX_ATTEMPTS = 16

    def __init__(self, directory):
        self.directory = Path(directory).absolute()
        if self.directory.resolve() != self.directory:
            raise ValueError("Mission state must not traverse symbolic links.")
        if self.directory.is_relative_to(Path(__file__).resolve().parents[1]):
            raise ValueError("Mission state must be outside application source.")
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        if os.name == "posix":
            if self.directory.stat().st_mode & 0o077:
                raise ValueError("Mission directory must be private (mode 0700).")
        elif os.name == "nt":
            _reject_windows_reparse_path(self.directory)
            _harden_windows_directory(self.directory)
        else:
            raise RuntimeError("Durable mission host is not certified on this platform.")
        self._guard = threading.RLock()
        self._closed = False
        self._lease = _open_host_lease(self.directory / "host.lock")
        try:
            path = self.directory / "missions.sqlite"
            _create_database_file(path)
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
            _close_host_lease(self._lease)
            raise

    def _event(self, mid, sequence, state, stamp):
        self._db.execute("INSERT INTO events VALUES (?,?,?,?)", (mid, sequence, state, stamp))

    def close(self):
        with self._guard:
            if not self._closed:
                self._db.close()
                _close_host_lease(self._lease)
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
