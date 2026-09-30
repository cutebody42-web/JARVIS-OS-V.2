"""Standalone signed NEXUS device-sync HTTP service.

This is intentionally separate from the hosted multi-user API. A device node
must opt in and bind this app to loopback or its chosen Tailscale interface.
Semantic payloads require application-level Ed25519 authentication.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import json
import threading

from fastapi import FastAPI, HTTPException, Request, Response

from core.nexus.pairing import PairingRequest
from core.nexus.peer_auth import MAX_ENVELOPE_BYTES
from core.nexus.signed_transport import MEDIA_TYPE, SignedSyncEndpoint
from core.nexus.sync_node import NexusSyncNode


async def _bounded_body(request: Request, *, limit: int = MAX_ENVELOPE_BYTES) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="Request body exceeds byte budget")
        chunks.append(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="Empty sync envelope")
    return b"".join(chunks)


def create_sync_app(
    node: NexusSyncNode,
    *,
    run_scheduler: bool = True,
    scheduler_poll_seconds: float = 0.25,
) -> FastAPI:
    if not isinstance(node, NexusSyncNode):
        raise TypeError("node must be NexusSyncNode")

    stop_event = threading.Event()
    scheduler_thread: threading.Thread | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal scheduler_thread
        await asyncio.to_thread(node.recover)
        if run_scheduler:
            scheduler_thread = threading.Thread(
                target=node.scheduler.run,
                args=(stop_event,),
                kwargs={"poll_seconds": scheduler_poll_seconds},
                name="nexus-sync-scheduler",
                daemon=True,
            )
            scheduler_thread.start()
        try:
            yield
        finally:
            stop_event.set()
            if scheduler_thread is not None:
                await asyncio.to_thread(scheduler_thread.join, 3)

    app = FastAPI(
        title="NEXUS Device Sync",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/nexus/sync/v1/health")
    def health() -> dict:
        return {
            "ok": True,
            "device_id": node.identity.device_id,
            "fingerprint": node.signer.fingerprint,
            "protocol": 1,
        }

    @app.get("/nexus/sync/v1/identity")
    def identity() -> dict:
        # Public key material is intentionally shareable for explicit pairing.
        return {
            "device_id": node.identity.device_id,
            "public_key": node.identity.public_key,
            "fingerprint": node.signer.fingerprint,
        }

    @app.post("/nexus/pair/v1/request", status_code=202)
    async def pairing_request(request: Request) -> dict:
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise HTTPException(status_code=415, detail="Pairing request must be JSON")
        body = await _bounded_body(request, limit=16 * 1024)
        try:
            value = json.loads(body)
            candidate = PairingRequest.from_dict(value)
            pending = await asyncio.to_thread(node.pairing.receive_request, candidate)
        except PermissionError:
            raise HTTPException(status_code=403, detail="Pairing proof was rejected") from None
        except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
            raise HTTPException(status_code=400, detail="Invalid pairing request") from None
        return {
            "accepted": True,
            "pairing_id": pending.pairing_id,
            "candidate_device": pending.candidate_device,
            "owner_approval_required": True,
        }

    @app.post("/nexus/sync/v1/batch")
    async def receive_batch(request: Request) -> Response:
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != MEDIA_TYPE:
            raise HTTPException(status_code=415, detail="Unsupported sync media type")
        body = await _bounded_body(request)
        try:
            signed_ack = await asyncio.to_thread(node.endpoint.handle_batch, body)
        except PermissionError:
            raise HTTPException(status_code=403, detail="Peer authentication failed") from None
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Invalid sync envelope") from None
        except Exception:
            # Do not leak SQLite paths, keys, peer names or semantic payloads.
            raise HTTPException(status_code=503, detail="Sync node could not accept the batch") from None
        return Response(content=signed_ack, media_type=MEDIA_TYPE)

    return app
