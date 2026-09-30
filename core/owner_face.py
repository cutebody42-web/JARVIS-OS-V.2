"""Local owner face identity for JARVIS.

Enrollment stores only normalized numeric face templates in the OS secret store;
camera frames are processed in memory and discarded. Face identity is an
owner-presence signal, not a replacement for companion biometric approval.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import threading
import time
from typing import Iterable

from core.secret_store import SecretStore, get_secret_store


_FACE_KEY = "identity.owner.face.v1"
_TEMPLATE_VERSION = 1


class FaceIdentityError(RuntimeError):
    pass


@dataclass(frozen=True)
class FaceVerification:
    enrolled: bool
    recognized: bool
    score: float
    matched_frames: int
    total_frames: int


def _normalize(vector) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-9:
        raise ValueError("face template vector is empty")
    return [value / norm for value in values]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    return sum(a * b for a, b in zip(left, right))


class OwnerFaceRecognizer:
    def __init__(
        self,
        *,
        secret_store: SecretStore | None = None,
        recognition_ttl_seconds: float = 300.0,
        clock=time.monotonic,
    ):
        if recognition_ttl_seconds <= 0:
            raise ValueError("recognition_ttl_seconds must be positive")
        self.store = secret_store or get_secret_store()
        self._ttl = float(recognition_ttl_seconds)
        self._clock = clock
        self._guard = threading.RLock()
        self._recognized_until = 0.0
        self._last_score = 0.0

    def _payload(self) -> dict | None:
        raw = self.store.get(_FACE_KEY)
        if not raw:
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FaceIdentityError("Stored owner face identity is invalid.") from exc
        if (
            not isinstance(value, dict)
            or value.get("version") != _TEMPLATE_VERSION
            or not isinstance(value.get("templates"), list)
            or not isinstance(value.get("threshold"), (int, float))
        ):
            raise FaceIdentityError("Stored owner face identity is invalid.")
        return value

    @property
    def enrolled(self) -> bool:
        try:
            payload = self._payload()
        except FaceIdentityError:
            return False
        return bool(payload and payload.get("templates"))

    @property
    def recognized(self) -> bool:
        with self._guard:
            return self._clock() < self._recognized_until

    @property
    def last_score(self) -> float:
        with self._guard:
            return self._last_score

    def forget(self) -> None:
        self.store.delete(_FACE_KEY)
        with self._guard:
            self._recognized_until = 0.0
            self._last_score = 0.0

    def enroll_embeddings(self, embeddings: Iterable[Iterable[float]]) -> int:
        templates = [_normalize(vector) for vector in embeddings]
        if len(templates) < 6:
            raise ValueError("At least six owner face samples are required.")
        width = len(templates[0])
        if width < 32 or any(len(item) != width for item in templates):
            raise ValueError("Face templates have inconsistent dimensions.")

        similarities: list[float] = []
        for index, left in enumerate(templates):
            for right in templates[index + 1:]:
                similarities.append(_cosine(left, right))
        mean_intra = sum(similarities) / max(1, len(similarities))
        # Conservative adaptive threshold; never below 0.82.
        threshold = max(0.82, min(0.95, mean_intra - 0.08))

        payload = {
            "version": _TEMPLATE_VERSION,
            "threshold": round(threshold, 6),
            "templates": [[round(value, 7) for value in item] for item in templates],
        }
        self.store.set(_FACE_KEY, json.dumps(payload, separators=(",", ":")))
        return len(templates)

    def verify_embedding(self, embedding: Iterable[float]) -> tuple[bool, float]:
        payload = self._payload()
        if payload is None:
            return False, 0.0
        candidate = _normalize(embedding)
        templates = [list(map(float, item)) for item in payload["templates"]]
        score = max((_cosine(candidate, template) for template in templates), default=-1.0)
        threshold = float(payload["threshold"])
        return score >= threshold, score

    @staticmethod
    def _extract_embedding(frame) -> list[float] | None:
        try:
            import cv2
            import numpy as np
        except Exception as exc:
            raise FaceIdentityError("OpenCV face identity runtime is unavailable.") from exc

        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            raise FaceIdentityError("OpenCV face detector could not be loaded.")
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )
        if len(faces) != 1:
            return None
        x, y, w, h = [int(value) for value in faces[0]]
        pad_x = max(4, int(w * 0.08))
        pad_y = max(4, int(h * 0.10))
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(gray.shape[1], x + w + pad_x), min(gray.shape[0], y + h + pad_y)
        face = gray[y0:y1, x0:x1]
        if face.size == 0:
            return None

        face = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)
        face = cv2.equalizeHist(face)
        matrix = np.asarray(face, dtype=np.float32) / 255.0
        dct = cv2.dct(matrix)
        # Low/mid frequency facial structure; discard DC brightness term.
        vector = dct[:24, :24].reshape(-1)[1:]
        return _normalize(vector.tolist())

    def _camera_embeddings(
        self,
        *,
        camera_index: int,
        target_samples: int,
        timeout_seconds: float,
    ) -> list[list[float]]:
        try:
            import cv2
        except Exception as exc:
            raise FaceIdentityError("OpenCV camera runtime is unavailable.") from exc

        if isinstance(camera_index, bool) or not isinstance(camera_index, int) or camera_index < 0:
            raise ValueError("camera_index must be a non-negative integer")
        if not 1 <= target_samples <= 30:
            raise ValueError("target_samples must be between 1 and 30")
        if not 2 <= timeout_seconds <= 30:
            raise ValueError("timeout_seconds must be between 2 and 30")

        backend = getattr(cv2, "CAP_DSHOW", 0)
        capture = cv2.VideoCapture(camera_index, backend) if backend else cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            capture.release()
            raise FaceIdentityError("JARVIS could not open the selected camera.")

        samples: list[list[float]] = []
        deadline = time.monotonic() + timeout_seconds
        try:
            while time.monotonic() < deadline and len(samples) < target_samples:
                ok, frame = capture.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                embedding = self._extract_embedding(frame)
                if embedding is not None:
                    samples.append(embedding)
                time.sleep(0.08)
        finally:
            capture.release()
        return samples

    def enroll_from_camera(
        self,
        *,
        camera_index: int = 0,
        samples: int = 10,
        timeout_seconds: float = 18.0,
    ) -> int:
        embeddings = self._camera_embeddings(
            camera_index=camera_index,
            target_samples=samples,
            timeout_seconds=timeout_seconds,
        )
        if len(embeddings) < max(6, samples // 2):
            raise FaceIdentityError(
                "JARVIS could not collect enough single-face samples for reliable enrollment."
            )
        count = self.enroll_embeddings(embeddings)
        # Enrollment itself does not silently count as later authentication.
        return count

    def verify_from_camera(
        self,
        *,
        camera_index: int = 0,
        samples: int = 5,
        timeout_seconds: float = 10.0,
    ) -> FaceVerification:
        if not self.enrolled:
            return FaceVerification(False, False, 0.0, 0, 0)
        embeddings = self._camera_embeddings(
            camera_index=camera_index,
            target_samples=samples,
            timeout_seconds=timeout_seconds,
        )
        if not embeddings:
            return FaceVerification(True, False, 0.0, 0, 0)

        decisions = [self.verify_embedding(item) for item in embeddings]
        matched = sum(1 for accepted, _ in decisions if accepted)
        score = sum(value for _, value in decisions) / len(decisions)
        required = max(2, math.ceil(len(decisions) * 0.6))
        recognized = matched >= required
        with self._guard:
            self._last_score = score
            self._recognized_until = self._clock() + self._ttl if recognized else 0.0
        return FaceVerification(True, recognized, score, matched, len(decisions))
