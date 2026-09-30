"""Authenticated owner channel, deliberately absent from model tool declarations."""
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from api.auth import get_current_user
from api.models import User
from core.mission_store import MissionConflict

router = APIRouter(prefix="/missions", tags=["owner missions"])


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GoalInput(StrictInput):
    goal: str = Field(min_length=1, max_length=4000)


class ActionInput(StrictInput):
    tool: str = Field(min_length=1, max_length=120)
    arguments: dict


class StateInput(StrictInput):
    session_id: str = Field(min_length=1, max_length=100)
    version: int = Field(ge=0)


class RunInput(StateInput):
    ticket_id: str | None = Field(default=None, min_length=1, max_length=100)


class ApproveInput(StrictInput):
    session_id: str = Field(min_length=1, max_length=100)
    expected_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    ttl_seconds: int = Field(default=120, gt=0, le=300)


class RevokeInput(StrictInput):
    session_id: str = Field(min_length=1, max_length=100)
    ticket_id: str = Field(min_length=1, max_length=100)


def service(request: Request):
    value = getattr(request.app.state, "missions", None)
    if value is None:
        raise HTTPException(503, "Durable missions are not enabled on this host")
    return value


def call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except KeyError:
        raise HTTPException(404, "Mission not found") from None
    except (MissionConflict, PermissionError):
        raise HTTPException(409, "Mission, session or consent changed; review again") from None
    except (ValueError, TypeError):
        raise HTTPException(422, "Invalid or unsupported mission request") from None


@router.post("")
def create_goal(payload: GoalInput, user: User = Depends(get_current_user), host=Depends(service)):
    return call(host.create_goal, user.id, payload.goal)


@router.post("/actions")
def create_action(payload: ActionInput, user: User = Depends(get_current_user), host=Depends(service)):
    return call(host.create_action, user.id, payload.tool, payload.arguments)


@router.get("")
def list_missions(user: User = Depends(get_current_user), host=Depends(service)):
    return host.store.list(user.id)


@router.get("/{mission_id}")
def get_mission(mission_id: str, user: User = Depends(get_current_user), host=Depends(service)):
    return call(host.view, user.id, mission_id)


@router.post("/{mission_id}/run")
def run_mission(mission_id: str, payload: RunInput, user: User = Depends(get_current_user), host=Depends(service)):
    return call(host.run, user.id, mission_id, payload.session_id, payload.version, ticket_id=payload.ticket_id)


@router.post("/{mission_id}/approve")
def approve_mission(mission_id: str, payload: ApproveInput, user: User = Depends(get_current_user), host=Depends(service)):
    return call(host.approve, user.id, mission_id, payload.session_id, payload.expected_digest, ttl_seconds=payload.ttl_seconds)


@router.post("/{mission_id}/revoke")
def revoke_mission(mission_id: str, payload: RevokeInput, user: User = Depends(get_current_user), host=Depends(service)):
    return {"revoked": call(host.revoke, user.id, mission_id, payload.session_id, payload.ticket_id)}


@router.post("/{mission_id}/cancel")
def cancel_mission(mission_id: str, payload: StateInput, user: User = Depends(get_current_user), host=Depends(service)):
    return call(host.cancel, user.id, mission_id, payload.session_id, payload.version)
