"""Local owner face identity for JARVIS.

Enrollment stores only normalized numeric face templates in the OS secret store.
Camera frames are processed in memory and discarded. Face identity is a
convenience/presence signal only; sensitive operations still require the paired
phone biometric approval path.

The production camera path uses OpenCV Zoo YuNet + SFace. Model files are
downloaded lazily from the official OpenCV Hugging Face mirror, verified by
pinned SHA-256, and cached under JARVIS user data.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Iterable

from core.app_paths import user_data_dir
from core.secret_store import SecretStore, get_secret_store


_FACE_KEY = "identity.owner.face.v1"
_TEMPLATE_VERSION = 2
_ENGINE = "opencv_sface_2021dec"

_YUNET_URL = (
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

_SFACE_URL = (
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
_SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"

# OpenCV's public SFace example uses 0.363 cosine similarity as the generic
# same-identity boundary. Owner enrollment can safely be stricter because we
# have several templates from the same person.
_SFACE_BASELINE_COSINE = 0.363


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class OwnerFaceRecognizer:
    def __init__(
        self,
        *,
        secret_store: SecretStore | None = None,
        recognition_ttl_seconds: float = 300.0,
        clock=time.monotonic,
        model_dir: str | Path | None = None,
        http_get=None,
    ):
        if recognition_ttl_seconds <= 0:
            raise ValueError("recognition_ttl_seconds must be positive")
        self.store = secret_store or get_secret_store()
        self._ttl = float(recognition_ttl_seconds)
        self._clock = clock
        self._guard = threading.RLock()
        self._vision_guard = threading.RLock()
        self._recognized_until = 0.0
        self._last_score = 0.0
        self._model_dir = (
            Path(model_dir).expanduser()
            if model_dir is not None
            else user_data_dir() / "vision"
        )
        self._http_get = http_get
        self._detector = None
        self._recognizer = None

    def _payload(self) -> dict | None:
        raw = self.store.get(_FACE_KEY)
        if not raw:
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FaceIdentityError("Stored owner face identity is invalid.") from exc
        if not isinstance(value, dict):
            raise FaceIdentityError("Stored owner face identity is invalid.")
        # Previous prototype embeddings are deliberately not mixed with SFace
        # embeddings. Treat them as unenrolled so the UI asks for a fresh,
        # stronger enrollment instead of silently comparing incompatible data.
        if value.get("version") != _TEMPLATE_VERSION:
            return None
        if (
            value.get("engine") != _ENGINE
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

    @property
    def engine(self) -> str:
        return _ENGINE

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

        # Keep a margin below the owner's own enrollment variation while staying
        # stricter than OpenCV's generic SFace same-identity threshold.
        threshold = max(
            0.45,
            _SFACE_BASELINE_COSINE,
            min(0.72, mean_intra - 0.10),
        )

        payload = {
            "version": _TEMPLATE_VERSION,
            "engine": _ENGINE,
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

    def _download_verified(self, url: str, target: Path, expected_sha256: str) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and _sha256_file(target) == expected_sha256:
            return target

        getter = self._http_get
        if getter is None:
            import requests
            getter = requests.get

        temp = target.with_suffix(target.suffix + ".partial")
        temp.unlink(missing_ok=True)
        digest = hashlib.sha256()
        try:
            response = getter(
                url,
                stream=True,
                timeout=(8, 180),
                allow_redirects=True,
                headers={"User-Agent": "JARVIS-owner-face/2"},
            )
            response.raise_for_status()
            with temp.open("wb") as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if not block:
                        continue
                    digest.update(block)
                    handle.write(block)
        except Exception as exc:
            temp.unlink(missing_ok=True)
            raise FaceIdentityError(
                f"JARVIS could not prepare the face recognition model ({type(exc).__name__})."
            ) from None

        if digest.hexdigest() != expected_sha256:
            temp.unlink(missing_ok=True)
            raise FaceIdentityError("Downloaded face recognition model failed integrity verification.")
        temp.replace(target)
        return target

    def _vision_models(self):
        try:
            import cv2
        except Exception as exc:
            raise FaceIdentityError("OpenCV face identity runtime is unavailable.") from exc

        with self._vision_guard:
            if self._detector is not None and self._recognizer is not None:
                return self._detector, self._recognizer

            detector_path = self._download_verified(
                _YUNET_URL,
                self._model_dir / "face_detection_yunet_2023mar.onnx",
                _YUNET_SHA256,
            )
            recognizer_path = self._download_verified(
                _SFACE_URL,
                self._model_dir / "face_recognition_sface_2021dec.onnx",
                _SFACE_SHA256,
            )
            try:
                if hasattr(cv2, "FaceDetectorYN_create"):
                    detector = cv2.FaceDetectorYN_create(
                        str(detector_path),
                        "",
                        (320, 320),
                        0.85,
                        0.3,
                        5000,
                    )
                else:
                    detector = cv2.FaceDetectorYN.create(
                        str(detector_path),
                        "",
                        (320, 320),
                        0.85,
                        0.3,
                        5000,
                    )
                if hasattr(cv2, "FaceRecognizerSF_create"):
                    recognizer = cv2.FaceRecognizerSF_create(
                        str(recognizer_path),
                        "",
                    )
                else:
                    recognizer = cv2.FaceRecognizerSF.create(
                        str(recognizer_path),
                        "",
                    )
            except Exception as exc:
                raise FaceIdentityError(
                    f"OpenCV SFace initialization failed ({type(exc).__name__})."
                ) from None

            self._detector = detector
            self._recognizer = recognizer
            return detector, recognizer

    def _extract_embedding(self, frame) -> list[float] | None:
        try:
            import numpy as np
        except Exception as exc:
            raise FaceIdentityError("NumPy face identity runtime is unavailable.") from exc

        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        detector, recognizer = self._vision_models()
        height, width = frame.shape[:2]
        if width < 80 or height < 80:
            return None

        with self._vision_guard:
            try:
                detector.setInputSize((int(width), int(height)))
                _, faces = detector.detect(frame)
            except Exception as exc:
                raise FaceIdentityError(
                    f"Owner face detection failed ({type(exc).__name__})."
                ) from None

            if faces is None or len(faces) != 1:
                return None
            face = np.asarray(faces[0], dtype=np.float32)
            try:
                aligned = recognizer.alignCrop(frame, face)
                feature = recognizer.feature(aligned)
            except Exception as exc:
                raise FaceIdentityError(
                    f"Owner face embedding failed ({type(exc).__name__})."
                ) from None

        values = np.asarray(feature, dtype=np.float32).reshape(-1)
        if values.size < 32:
            return None
        return _normalize(values.tolist())

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

        # Prepare verified models before opening the camera so a one-time network
        # download cannot hold the webcam open.
        self._vision_models()

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
                time.sleep(0.10)
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
        return self.enroll_embeddings(embeddings)

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
