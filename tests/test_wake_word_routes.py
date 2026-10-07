import base64
import struct
import unittest

from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from api.wake_word_routes import install_wake_word_routes


class _Session:
    def __init__(self):
        self.enabled = False
        self.model_dir = None

    def status(self):
        return {
            "configured": self.model_dir is not None,
            "enabled": self.enabled,
            "model_dir": self.model_dir,
            "threshold": 0.5,
            "sample_rate": 16000,
            "chunk_samples": 1280,
            "last_score": None,
            "last_activation_at": None,
            "last_error": None,
            "local_only": True,
            "microphone_owner": False,
            "authority": "activation_only",
        }

    def configure(self, model_dir, *, threshold=0.5):
        self.model_dir = model_dir
        value = self.status()
        value["threshold"] = threshold
        return value

    def set_enabled(self, enabled):
        self.enabled = enabled
        return self.status()

    def process_base64_pcm(self, payload):
        base64.b64decode(payload, validate=True)
        return {
            "detected": True,
            "activated": True,
            "score": 0.9,
            "threshold": 0.5,
            "model_name": "wakeword",
            "authority": "activation_only",
        }


class WakeWordRoutesTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.session = _Session()

        def require_ui(authorization: str | None = Header(default=None)):
            if authorization != "Bearer unit-test-token":
                raise HTTPException(status_code=401, detail="Unauthorized")

        install_wake_word_routes(self.app, self.session, require_ui)
        self.client = TestClient(self.app)
        self.headers = {"Authorization": "Bearer unit-test-token"}

    def test_all_routes_require_existing_ui_authentication(self):
        for method, path, payload in (
            ("get", "/v1/wake-word/status", None),
            ("post", "/v1/wake-word/configure", {"model_dir": "C:/models", "threshold": 0.5}),
            ("post", "/v1/wake-word/enabled", {"enabled": True}),
            ("post", "/v1/wake-word/frame", {"pcm16_base64": "AAAA"}),
        ):
            response = getattr(self.client, method)(path, json=payload) if payload is not None else getattr(self.client, method)(path)
            self.assertEqual(response.status_code, 401)

    def test_configure_enable_and_frame_are_activation_only(self):
        configured = self.client.post(
            "/v1/wake-word/configure",
            headers=self.headers,
            json={"model_dir": "C:/owner/wake", "threshold": 0.65},
        )
        self.assertEqual(configured.status_code, 200)
        self.assertTrue(configured.json()["configured"])

        enabled = self.client.post(
            "/v1/wake-word/enabled",
            headers=self.headers,
            json={"enabled": True},
        )
        self.assertEqual(enabled.status_code, 200)
        self.assertTrue(enabled.json()["enabled"])

        raw = struct.pack("<" + "h" * 160, *([0] * 160))
        frame = self.client.post(
            "/v1/wake-word/frame",
            headers=self.headers,
            json={"pcm16_base64": base64.b64encode(raw).decode("ascii")},
        )
        self.assertEqual(frame.status_code, 200)
        self.assertTrue(frame.json()["activated"])
        self.assertEqual(frame.json()["authority"], "activation_only")

    def test_request_models_bound_frame_and_threshold(self):
        response = self.client.post(
            "/v1/wake-word/configure",
            headers=self.headers,
            json={"model_dir": "C:/owner/wake", "threshold": 2.0},
        )
        self.assertEqual(response.status_code, 422)
        response = self.client.post(
            "/v1/wake-word/frame",
            headers=self.headers,
            json={"pcm16_base64": "A" * 240001},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
