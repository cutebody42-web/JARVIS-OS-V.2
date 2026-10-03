"""Fetch pinned OpenCV Zoo face models for the packaged JARVIS Brain."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import sys
import urllib.request


MODELS = (
    (
        "face_detection_yunet_2023mar.onnx",
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    (
        "face_recognition_sface_2021dec.onnx",
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename, url, expected in MODELS:
        target = target_dir / filename
        if target.is_file() and sha256(target) == expected:
            print(f"{filename}: verified existing")
            continue
        partial = target.with_suffix(target.suffix + ".partial")
        partial.unlink(missing_ok=True)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "JARVIS-build/2"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        actual = sha256(partial)
        if actual != expected:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"{filename}: SHA-256 mismatch: expected {expected}, got {actual}"
            )
        partial.replace(target)
        print(f"{filename}: verified {actual}")


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models") / "vision"
    fetch(destination)
