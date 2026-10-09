"""Owner-local text embeddings for NEXUS temporal memory.

The runtime is deliberately offline by construction: the owner supplies an
existing model directory containing an ONNX model, tokenizer and JARVIS
manifest. No model identifier, URL, download helper or network client is
accepted here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
import threading
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np


_MANIFEST_NAME = "jarvis-embedding.json"
_ALLOWED_INPUTS = {"input_ids", "attention_mask", "token_type_ids"}
_MAX_BATCH = 32
_MAX_HIDDEN_ELEMENTS = 16_777_216
_MAX_MANIFEST_BYTES = 65_536
_MAX_TOKENIZER_BYTES = 256 * 1024 * 1024
_MAX_MODEL_BYTES = 8 * 1024 * 1024 * 1024


class LocalEmbeddingError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbeddingInfo:
    model_id: str
    dimension: int
    engine: str = "onnx-local"
    normalized: bool = True
    local_only: bool = True


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def info(self) -> EmbeddingInfo: ...

    def embed_texts(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...

    def embed(self, text: str) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class _Manifest:
    model_id: str
    dimension: int
    model_file: Path
    tokenizer_file: Path
    pooling: str
    normalize: bool
    max_length: int
    pad_token_id: int
    output_name: str | None


def _bounded_int(value: object, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise LocalEmbeddingError(f"Invalid {name} in embedding manifest.")
    return value


def _is_redirect(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _windows_drive_type(path: Path) -> int | None:
    if os.name != "nt":
        return None
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetDriveTypeW(str(Path(path.anchor))))
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise LocalEmbeddingError("Could not verify a local fixed model drive.") from exc


def _reject_windows_redirects(path: Path) -> None:
    absolute = path.absolute()
    drive_type = _windows_drive_type(absolute)
    if drive_type is not None and drive_type != 3:
        raise LocalEmbeddingError("Local embedding models must use a local fixed drive.")
    if os.name != "nt":
        return
    if str(absolute).startswith("\\\\"):
        raise LocalEmbeddingError("Local embedding models cannot use a network share.")
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if _is_redirect(current):
            raise LocalEmbeddingError("Local embedding model paths cannot traverse reparse points.")


def _local_file(root: Path, value: object, label: str, *, max_bytes: int) -> Path:
    if not isinstance(value, str) or not value.strip() or len(value) > 240:
        raise LocalEmbeddingError(f"Invalid {label} in embedding manifest.")
    unresolved = root / value
    _reject_windows_redirects(unresolved)
    if _is_redirect(unresolved):
        raise LocalEmbeddingError(f"Local {label} cannot be a symbolic link or reparse point.")
    candidate = unresolved.resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise LocalEmbeddingError(f"Local {label} does not exist inside the model directory.")
    try:
        size = candidate.stat().st_size
    except OSError:
        raise LocalEmbeddingError(f"Local {label} is unreadable.") from None
    if size <= 0 or size > max_bytes:
        raise LocalEmbeddingError(f"Local {label} exceeds its file-size budget.")
    return candidate


def _load_manifest(root: Path) -> _Manifest:
    manifest_path = root / _MANIFEST_NAME
    if _is_redirect(manifest_path) or not manifest_path.is_file():
        raise LocalEmbeddingError(f"Local embedding model is missing {_MANIFEST_NAME}.")
    try:
        with manifest_path.open("rb") as handle:
            content = handle.read(_MAX_MANIFEST_BYTES + 1)
        if len(content) > _MAX_MANIFEST_BYTES:
            raise LocalEmbeddingError("Local embedding manifest exceeds 64 KiB.")
        raw = json.loads(content.decode("utf-8"))
    except LocalEmbeddingError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise LocalEmbeddingError("Local embedding manifest is unreadable.") from None
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise LocalEmbeddingError("Unsupported local embedding manifest schema.")

    model_id = raw.get("model_id")
    if (
        not isinstance(model_id, str)
        or not 1 <= len(model_id.strip()) <= 160
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in model_id)
    ):
        raise LocalEmbeddingError("Invalid model_id in embedding manifest.")
    dimension = _bounded_int(raw.get("dimension"), "dimension", 1, 8192)
    pooling = raw.get("pooling", "mean")
    if pooling not in {"mean", "cls"}:
        raise LocalEmbeddingError("Embedding pooling must be 'mean' or 'cls'.")
    normalize = raw.get("normalize", True)
    if not isinstance(normalize, bool):
        raise LocalEmbeddingError("Embedding normalize must be boolean.")
    max_length = _bounded_int(raw.get("max_length", 512), "max_length", 8, 8192)
    pad_token_id = _bounded_int(raw.get("pad_token_id", 0), "pad_token_id", 0, 2**31 - 1)
    output_name = raw.get("output_name")
    if output_name is not None and (
        not isinstance(output_name, str)
        or not 1 <= len(output_name.strip()) <= 120
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in output_name)
    ):
        raise LocalEmbeddingError("Invalid output_name in embedding manifest.")

    return _Manifest(
        model_id=model_id.strip(),
        dimension=dimension,
        model_file=_local_file(
            root,
            raw.get("model_file", "model.onnx"),
            "ONNX model file",
            max_bytes=_MAX_MODEL_BYTES,
        ),
        tokenizer_file=_local_file(
            root,
            raw.get("tokenizer_file", "tokenizer.json"),
            "tokenizer file",
            max_bytes=_MAX_TOKENIZER_BYTES,
        ),
        pooling=pooling,
        normalize=normalize,
        max_length=max_length,
        pad_token_id=pad_token_id,
        output_name=output_name.strip() if isinstance(output_name, str) else None,
    )


class LocalOnnxEmbeddingProvider:
    """Lazy, local-files-only ONNX embedding provider.

    Expected model directory::

        model/
          model.onnx
          tokenizer.json
          jarvis-embedding.json

    ``jarvis-embedding.json`` must declare ``schema_version: 1``, ``model_id``
    and ``dimension``. Optional controls are ``pooling`` (mean/cls),
    ``normalize``, ``max_length``, ``pad_token_id``, ``model_file``,
    ``tokenizer_file`` and ``output_name``.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        session_factory: Callable[[Path], Any] | None = None,
        tokenizer_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        root = Path(model_path).expanduser()
        if not root.is_absolute():
            raise LocalEmbeddingError("Local embedding model directory must be absolute.")
        _reject_windows_redirects(root)
        if _is_redirect(root):
            raise LocalEmbeddingError(
                "Local embedding model directory cannot be a symbolic link or reparse point."
            )
        if not root.exists() or not root.is_dir():
            raise LocalEmbeddingError("Local embedding model directory does not exist.")
        self.model_path = root.resolve()
        self._manifest = _load_manifest(self.model_path)
        self._model_identity = self._file_identity(self._manifest.model_file)
        self._tokenizer_identity = self._file_identity(self._manifest.tokenizer_file)
        self._session_factory = session_factory
        self._tokenizer_factory = tokenizer_factory
        self._session: Any | None = None
        self._tokenizer: Any | None = None
        self._output_name: str | None = None
        self._load_lock = threading.RLock()

    @staticmethod
    def _file_identity(path: Path) -> tuple[int, int, int, int]:
        try:
            info = path.stat(follow_symlinks=False)
        except OSError:
            raise LocalEmbeddingError("Local embedding model asset is unreadable.") from None
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)

    def _revalidate_assets(self) -> None:
        if _is_redirect(self._manifest.model_file) or _is_redirect(self._manifest.tokenizer_file):
            raise LocalEmbeddingError("Local embedding model assets changed before loading.")
        if (
            self._file_identity(self._manifest.model_file) != self._model_identity
            or self._file_identity(self._manifest.tokenizer_file) != self._tokenizer_identity
        ):
            raise LocalEmbeddingError("Local embedding model assets changed before loading.")

    @property
    def info(self) -> EmbeddingInfo:
        return EmbeddingInfo(
            model_id=self._manifest.model_id,
            dimension=self._manifest.dimension,
            normalized=self._manifest.normalize,
            local_only=True,
        )

    @property
    def loaded(self) -> bool:
        return self._session is not None and self._tokenizer is not None

    def _load(self) -> tuple[Any, Any]:
        with self._load_lock:
            if self.loaded:
                return self._session, self._tokenizer
            self._revalidate_assets()

            tokenizer_factory = self._tokenizer_factory
            if tokenizer_factory is None:
                try:
                    from tokenizers import Tokenizer
                except ImportError as exc:
                    raise LocalEmbeddingError(
                        "tokenizers is not installed; install the local embeddings extra first."
                    ) from exc
                tokenizer_factory = lambda path: Tokenizer.from_file(str(path))

            session_factory = self._session_factory
            if session_factory is None:
                try:
                    import onnxruntime as ort
                except ImportError as exc:
                    raise LocalEmbeddingError(
                        "onnxruntime is not installed; install the local embeddings extra first."
                    ) from exc
                session_factory = lambda path: ort.InferenceSession(
                    str(path), providers=["CPUExecutionProvider"]
                )

            try:
                tokenizer = tokenizer_factory(self._manifest.tokenizer_file)
                session = session_factory(self._manifest.model_file)
                input_names = {getattr(item, "name", "") for item in session.get_inputs()}
                output_names = [getattr(item, "name", "") for item in session.get_outputs()]
            except LocalEmbeddingError:
                raise
            except Exception as exc:
                raise LocalEmbeddingError(
                    f"Local embedding runtime failed to load ({type(exc).__name__})."
                ) from None

            unknown = input_names - _ALLOWED_INPUTS
            if "input_ids" not in input_names or unknown:
                raise LocalEmbeddingError("Unsupported ONNX embedding input contract.")
            if not output_names or any(not name for name in output_names):
                raise LocalEmbeddingError("ONNX embedding model exposes no usable output.")
            if self._manifest.output_name and self._manifest.output_name not in output_names:
                raise LocalEmbeddingError("Configured embedding output_name is not present in the model.")

            self._output_name = self._manifest.output_name or output_names[0]
            self._tokenizer = tokenizer
            self._session = session
            return session, tokenizer

    @staticmethod
    def _validate_text(text: object) -> str:
        if not isinstance(text, str):
            raise ValueError("Embedding input must be text.")
        clean = text.strip().replace("\x00", "")
        if not clean or len(clean) > 20000:
            raise ValueError("Embedding input must be non-empty text up to 20000 characters.")
        return clean

    def _inputs(
        self,
        tokenizer: Any,
        texts: Sequence[str],
        input_names: set[str],
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        try:
            encoded = [tokenizer.encode(text) for text in texts]
        except Exception as exc:
            raise LocalEmbeddingError(
                f"Local tokenizer failed ({type(exc).__name__})."
            ) from None

        ids: list[list[int]] = []
        type_ids: list[list[int]] = []
        try:
            for item in encoded:
                token_ids = list(getattr(item, "ids", ()))[: self._manifest.max_length]
                if not token_ids:
                    raise LocalEmbeddingError("Tokenizer produced an empty embedding input.")
                ids.append([int(value) for value in token_ids])
                raw_types = list(getattr(item, "type_ids", ()))[: len(token_ids)]
                type_ids.append(
                    [int(value) for value in raw_types]
                    if len(raw_types) == len(token_ids)
                    else [0] * len(token_ids)
                )
        except LocalEmbeddingError:
            raise
        except (TypeError, ValueError, OverflowError):
            raise LocalEmbeddingError("Tokenizer produced invalid token ids.") from None

        width = max(len(row) for row in ids)
        batch_ids = np.full((len(ids), width), self._manifest.pad_token_id, dtype=np.int64)
        attention = np.zeros((len(ids), width), dtype=np.int64)
        batch_types = np.zeros((len(ids), width), dtype=np.int64)
        for index, row in enumerate(ids):
            batch_ids[index, : len(row)] = row
            attention[index, : len(row)] = 1
            batch_types[index, : len(row)] = type_ids[index]

        payload = {"input_ids": batch_ids}
        if "attention_mask" in input_names:
            payload["attention_mask"] = attention
        if "token_type_ids" in input_names:
            payload["token_type_ids"] = batch_types
        return payload, attention

    def _pool(self, output: np.ndarray, attention: np.ndarray) -> np.ndarray:
        if output.size > _MAX_HIDDEN_ELEMENTS:
            raise LocalEmbeddingError("Embedding output exceeds the local inference memory budget.")
        if output.ndim == 2:
            if output.shape[0] != attention.shape[0]:
                raise LocalEmbeddingError("Embedding output batch size is invalid.")
            vectors = output
        elif output.ndim == 3:
            if output.shape[:2] != attention.shape:
                raise LocalEmbeddingError("Embedding output token shape is invalid.")
            if self._manifest.pooling == "cls":
                vectors = output[:, 0, :]
            else:
                weights = attention.astype(np.float32)[..., None]
                denominator = np.maximum(weights.sum(axis=1), 1.0)
                vectors = (output.astype(np.float32) * weights).sum(axis=1) / denominator
        else:
            raise LocalEmbeddingError("Unsupported ONNX embedding output shape.")
        if vectors.ndim != 2 or vectors.shape[1] != self._manifest.dimension:
            raise LocalEmbeddingError("Embedding output dimension does not match the manifest.")
        vectors = vectors.astype(np.float32, copy=False)
        if not np.isfinite(vectors).all():
            raise LocalEmbeddingError("Embedding model produced non-finite values.")
        if self._manifest.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if np.any(norms <= 0.0) or not np.isfinite(norms).all():
                raise LocalEmbeddingError("Embedding model produced a zero-length vector.")
            vectors = vectors / norms
        return vectors

    def embed_texts(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise ValueError("texts must be a sequence of strings.")
        if not 1 <= len(texts) <= _MAX_BATCH:
            raise ValueError(f"Embedding batch size must be between 1 and {_MAX_BATCH}.")
        if len(texts) * self._manifest.max_length * self._manifest.dimension > _MAX_HIDDEN_ELEMENTS:
            raise ValueError("Embedding request exceeds the local inference memory budget.")
        clean = [self._validate_text(text) for text in texts]
        session, tokenizer = self._load()
        input_names = {getattr(item, "name", "") for item in session.get_inputs()}
        payload, attention = self._inputs(tokenizer, clean, input_names)
        if not self._output_name:
            raise LocalEmbeddingError("Local embedding runtime has no selected output.")
        requested_outputs = [self._output_name]
        try:
            raw = session.run(requested_outputs, payload)
        except Exception as exc:
            raise LocalEmbeddingError(
                f"Local embedding inference failed ({type(exc).__name__})."
            ) from None
        if not isinstance(raw, (list, tuple)) or not raw:
            raise LocalEmbeddingError("Local embedding inference returned no output.")
        vectors = self._pool(np.asarray(raw[0]), attention)
        return tuple(tuple(float(value) for value in row) for row in vectors)

    def embed(self, text: str) -> tuple[float, ...]:
        return self.embed_texts((text,))[0]
