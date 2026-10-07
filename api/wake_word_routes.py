"""Loopback wake-word routes for the installed desktop product.

Routes are installed into the existing bearer-token protected local Brain API.
They expose configuration and activation only; no route dispatches a JARVIS
action or converts a wake hit into authorization.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.wake_word_session import WakeWordSession, WakeWordSessionError


class WakeWordConfigureRequest(BaseModel):
    model_dir: str = Field(min_length=1, max_length=4096)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class WakeWordEnabledRequest(BaseModel):
    enabled: bool


class WakeWordFrameRequest(BaseModel):
    pcm16_base64: str = Field(min_length=1, max_length=240000)


def install_wake_word_routes(
    app: FastAPI,
    session: WakeWordSession,
    require_ui: Callable[..., Any],
) -> None:
    """Install wake-word endpoints on the existing local Brain application."""

    @app.get("/v1/wake-word/status", dependencies=[Depends(require_ui)])
    def wake_word_status() -> dict[str, Any]:
        return session.status()

    @app.post("/v1/wake-word/configure", dependencies=[Depends(require_ui)])
    def wake_word_configure(request: WakeWordConfigureRequest) -> dict[str, Any]:
        try:
            return session.configure(request.model_dir, threshold=request.threshold)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except WakeWordSessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    @app.post("/v1/wake-word/enabled", dependencies=[Depends(require_ui)])
    def wake_word_enabled(request: WakeWordEnabledRequest) -> dict[str, Any]:
        try:
            return session.set_enabled(request.enabled)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except WakeWordSessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    @app.post("/v1/wake-word/frame", dependencies=[Depends(require_ui)])
    def wake_word_frame(request: WakeWordFrameRequest) -> dict[str, Any]:
        try:
            return session.process_base64_pcm(request.pcm16_base64)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except WakeWordSessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
