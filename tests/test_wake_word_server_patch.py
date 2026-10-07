import unittest

from scripts.patch_wakeword_product_loop import (
    MARKER,
    WakeWordServerPatchError,
    patch_text,
)


_FIXTURE = '''from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from core.voice_runtime import VoiceRuntimeError, WindowsVoiceRuntime

class LocalBrainHost:
    def __init__(self):
        self.state_dir = state_dir
        self.owner_face = OwnerFaceRecognizer()
        self.voice = WindowsVoiceRuntime()

    def status(self):
        return {
            "voice": self.voice.status(),
        }

def build_app(host):
    app = FastAPI()
    def require_ui():
        return None

    @app.get("/v1/voice/status", dependencies=[Depends(require_ui)])
    def voice_status():
        return host.voice.status()

    return app
'''


class WakeWordServerPatchTests(unittest.TestCase):
    def test_patch_is_scoped_and_idempotent(self):
        patched = patch_text(_FIXTURE)
        self.assertIn("from api.wake_word_routes import install_wake_word_routes", patched)
        self.assertIn("from core.wake_word_session import WakeWordSession", patched)
        self.assertIn('self.wake_word = WakeWordSession(self.state_dir / "wake-word.json")', patched)
        self.assertIn(MARKER, patched)
        self.assertIn('"wake_word": bool(self.wake_word.status()["enabled"])', patched)
        self.assertEqual(patch_text(patched), patched)

    def test_unexpected_server_layout_fails_closed(self):
        with self.assertRaises(WakeWordServerPatchError):
            patch_text("from fastapi import FastAPI\n")

    def test_partial_existing_patch_fails_closed(self):
        with self.assertRaises(WakeWordServerPatchError):
            patch_text(_FIXTURE + "\n" + MARKER)


if __name__ == "__main__":
    unittest.main()
