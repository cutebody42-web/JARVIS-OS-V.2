"""Owner face identity contracts without requiring a physical camera."""

import json
import unittest

from core.owner_face import OwnerFaceRecognizer
from core.secret_store import NoopSecretStore


def vec(index: int, tweak: float = 0.0):
    values = [0.0] * 64
    values[index] = 1.0
    values[(index + 1) % 64] = tweak
    return values


class StubRecognizer(OwnerFaceRecognizer):
    def __init__(self, frames, **kwargs):
        super().__init__(**kwargs)
        self.frames = frames

    def _camera_embeddings(self, **kwargs):
        return list(self.frames)


class OwnerFaceTests(unittest.TestCase):
    def test_enrollment_stores_templates_not_images_and_matches_owner(self):
        store = NoopSecretStore()
        recognizer = OwnerFaceRecognizer(secret_store=store)
        samples = [vec(0, tweak) for tweak in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)]
        self.assertEqual(recognizer.enroll_embeddings(samples), 6)
        self.assertTrue(recognizer.enrolled)

        matched, score = recognizer.verify_embedding(vec(0, 0.025))
        self.assertTrue(matched)
        self.assertGreater(score, 0.9)

        stranger, stranger_score = recognizer.verify_embedding(vec(10, 0.01))
        self.assertFalse(stranger)
        self.assertLess(stranger_score, 0.2)

        raw = store.get("identity.owner.face.v1")
        self.assertIsNotNone(raw)
        self.assertNotIn("image", raw.lower())
        self.assertNotIn("jpg", raw.lower())

    def test_camera_verification_requires_majority_of_frames(self):
        store = NoopSecretStore()
        enrollment = [vec(0, tweak) for tweak in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)]
        recognizer = StubRecognizer(
            [vec(0, 0.02), vec(0, 0.03), vec(0, 0.04), vec(10, 0.01), vec(11, 0.01)],
            secret_store=store,
        )
        recognizer.enroll_embeddings(enrollment)
        result = recognizer.verify_from_camera(samples=5)
        self.assertTrue(result.recognized)
        self.assertEqual(result.matched_frames, 3)
        self.assertTrue(recognizer.recognized)

    def test_old_prototype_templates_require_fresh_sface_enrollment(self):
        store = NoopSecretStore()
        store.set(
            "identity.owner.face.v1",
            json.dumps({
                "version": 1,
                "threshold": 0.82,
                "templates": [vec(0, 0.01) for _ in range(6)],
            }),
        )
        recognizer = OwnerFaceRecognizer(secret_store=store)
        self.assertFalse(recognizer.enrolled)
        self.assertEqual(recognizer.engine, "opencv_sface_2021dec")

    def test_forget_removes_identity(self):
        store = NoopSecretStore()
        recognizer = OwnerFaceRecognizer(secret_store=store)
        recognizer.enroll_embeddings([vec(0, x) for x in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)])
        recognizer.forget()
        self.assertFalse(recognizer.enrolled)


if __name__ == "__main__":
    unittest.main()
