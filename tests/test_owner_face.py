"""Owner face identity contracts without requiring a physical camera."""

import json
import math
import unittest
from unittest.mock import MagicMock, patch

from core.owner_face import FaceIdentityError, OwnerFaceRecognizer
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

    def test_non_finite_samples_cannot_be_enrolled(self):
        recognizer = OwnerFaceRecognizer(secret_store=NoopSecretStore())
        for invalid in (math.nan, math.inf, -math.inf):
            sample = vec(0)
            sample[1] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                recognizer.enroll_embeddings([sample] * 6)
        self.assertFalse(recognizer.enrolled)

    def test_accidental_stranger_sample_cannot_become_an_owner_template(self):
        recognizer = OwnerFaceRecognizer(secret_store=NoopSecretStore())
        with self.assertRaisesRegex(ValueError, "same owner"):
            recognizer.enroll_embeddings([vec(0)] * 5 + [vec(10)])
        self.assertFalse(recognizer.enrolled)

    def test_stored_malformed_template_fails_closed(self):
        store = NoopSecretStore()
        recognizer = OwnerFaceRecognizer(secret_store=store)
        recognizer.enroll_embeddings([vec(0)] * 6)
        payload = json.loads(store.get("identity.owner.face.v1"))
        payload["templates"][0][0] = math.nan
        store.set("identity.owner.face.v1", json.dumps(payload))
        self.assertFalse(recognizer.enrolled)
        with self.assertRaises(FaceIdentityError):
            recognizer.verify_embedding(vec(0))

    def test_no_face_recheck_revokes_previous_presence(self):
        recognizer = StubRecognizer([vec(0)] * 5, secret_store=NoopSecretStore())
        recognizer.enroll_embeddings([vec(0)] * 6)
        self.assertTrue(recognizer.verify_from_camera().recognized)
        recognizer.frames = []
        self.assertFalse(recognizer.verify_from_camera().recognized)
        self.assertFalse(recognizer.recognized)
        self.assertEqual(recognizer.last_score, 0.0)

    def test_camera_failure_recheck_revokes_previous_presence(self):
        recognizer = StubRecognizer([vec(0)] * 5, secret_store=NoopSecretStore())
        recognizer.enroll_embeddings([vec(0)] * 6)
        recognizer.verify_from_camera()
        with patch.object(recognizer, "_camera_embeddings", side_effect=FaceIdentityError("unavailable")):
            with self.assertRaises(FaceIdentityError):
                recognizer.verify_from_camera()
        self.assertFalse(recognizer.recognized)

    def test_reenrollment_revokes_previous_presence(self):
        recognizer = StubRecognizer([vec(0)] * 5, secret_store=NoopSecretStore())
        recognizer.enroll_embeddings([vec(0)] * 6)
        recognizer.verify_from_camera()
        recognizer.enroll_embeddings([vec(10)] * 6)
        self.assertFalse(recognizer.recognized)

    def test_forget_during_capture_cannot_restore_presence(self):
        recognizer = StubRecognizer([vec(0)] * 5, secret_store=NoopSecretStore())
        recognizer.enroll_embeddings([vec(0)] * 6)

        def capture(**kwargs):
            recognizer.forget()
            return [vec(0)] * 5

        with patch.object(recognizer, "_camera_embeddings", side_effect=capture):
            result = recognizer.verify_from_camera()
        self.assertFalse(result.enrolled)
        self.assertFalse(result.recognized)
        self.assertFalse(recognizer.recognized)

    def test_older_camera_check_cannot_overwrite_a_newer_no_face_check(self):
        recognizer = StubRecognizer([], secret_store=NoopSecretStore())
        recognizer.enroll_embeddings([vec(0)] * 6)

        def earlier_capture(**kwargs):
            # Simulate a second thread completing a newer camera check before
            # the first thread finishes extracting its successful frames.
            with patch.object(recognizer, "_camera_embeddings", return_value=[]):
                self.assertFalse(recognizer.verify_from_camera().recognized)
            return [vec(0)] * 5

        with patch.object(recognizer, "_camera_embeddings", side_effect=earlier_capture):
            result = recognizer.verify_from_camera()
        self.assertFalse(result.recognized)
        self.assertFalse(recognizer.recognized)

    def test_linux_and_macos_use_the_default_camera_backend(self):
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform):
                self._exercise_camera(platform, fallback=False)

    def test_windows_falls_back_if_directshow_cannot_open_camera(self):
        self._exercise_camera("win32", fallback=True)

    def _exercise_camera(self, platform, *, fallback):
        cv2 = MagicMock()
        cv2.CAP_DSHOW = 700
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, object())
        failed = MagicMock()
        failed.isOpened.return_value = False
        cv2.VideoCapture.side_effect = [failed, capture] if fallback else [capture]
        recognizer = OwnerFaceRecognizer(secret_store=NoopSecretStore())
        with (
            patch.dict("sys.modules", {"cv2": cv2}),
            patch("core.owner_face.sys.platform", platform),
            patch("core.owner_face.time.sleep"),
            patch.object(recognizer, "_vision_models"),
            patch.object(recognizer, "_extract_embedding", return_value=vec(0)),
        ):
            self.assertEqual(len(recognizer._camera_embeddings(
                camera_index=0, target_samples=1, timeout_seconds=2,
            )), 1)
        if fallback:
            self.assertEqual(cv2.VideoCapture.call_args_list[0].args, (0, 700))
            failed.release.assert_called_once()
        self.assertEqual(cv2.VideoCapture.call_args_list[-1].args, (0,))
        capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
