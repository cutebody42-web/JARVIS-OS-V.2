#!/usr/bin/env python3
"""Surgically wire the wake-word product loop into the local Brain server.

The server is intentionally large and security-sensitive. This patcher changes
four exact anchors and refuses to continue if the expected source layout moves.
It is idempotent and can also be used in --check mode in CI.
"""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "install_wake_word_routes(app, host.wake_word, require_ui)"


class WakeWordServerPatchError(RuntimeError):
    pass


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise WakeWordServerPatchError(f"Expected exactly one {label} anchor, found {count}.")
    return text.replace(old, new, 1)


def patch_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise WakeWordServerPatchError("Local Brain server source is empty.")
    if MARKER in text:
        required = (
            "from api.wake_word_routes import install_wake_word_routes",
            "from core.wake_word_session import WakeWordSession",
            'self.wake_word = WakeWordSession(self.state_dir / "wake-word.json")',
            '"wake_word": bool(self.wake_word.status()["enabled"])',
        )
        if not all(item in text for item in required):
            raise WakeWordServerPatchError("Existing wake-word integration is incomplete.")
        return text

    text = _replace_once(
        text,
        "from fastapi.middleware.cors import CORSMiddleware\nfrom pydantic import BaseModel, Field\n",
        "from fastapi.middleware.cors import CORSMiddleware\nfrom pydantic import BaseModel, Field\n\nfrom api.wake_word_routes import install_wake_word_routes\n",
        "FastAPI import",
    )
    text = _replace_once(
        text,
        "from core.voice_runtime import VoiceRuntimeError, WindowsVoiceRuntime\n",
        "from core.voice_runtime import VoiceRuntimeError, WindowsVoiceRuntime\nfrom core.wake_word_session import WakeWordSession\n",
        "voice runtime import",
    )
    text = _replace_once(
        text,
        "        self.owner_face = OwnerFaceRecognizer()\n        self.voice = WindowsVoiceRuntime()\n",
        "        self.owner_face = OwnerFaceRecognizer()\n        self.voice = WindowsVoiceRuntime()\n        self.wake_word = WakeWordSession(self.state_dir / \"wake-word.json\")\n",
        "host voice initialization",
    )
    text = _replace_once(
        text,
        '            "voice": self.voice.status(),\n',
        '            "voice": {**self.voice.status(), "wake_word": bool(self.wake_word.status()["enabled"])},\n',
        "status voice field",
    )
    text = _replace_once(
        text,
        '    @app.get("/v1/voice/status", dependencies=[Depends(require_ui)])\n',
        '    install_wake_word_routes(app, host.wake_word, require_ui)\n\n    @app.get("/v1/voice/status", dependencies=[Depends(require_ui)])\n',
        "voice route",
    )
    return text


def patch_file(path: str | Path, *, check: bool = False) -> bool:
    source = Path(path)
    if not source.is_file():
        raise WakeWordServerPatchError("Local Brain server file was not found.")
    original = source.read_text("utf-8")
    patched = patch_text(original)
    changed = patched != original
    if check:
        if changed:
            raise WakeWordServerPatchError("Wake-word integration patch has not been applied.")
        return False
    if changed:
        source.write_text(patched, "utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server_file")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        patch_file(args.server_file, check=args.check)
    except WakeWordServerPatchError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
