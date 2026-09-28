"""Minimal, application-owned Phase 1 capability boundary, not a full sandbox.

No policy mutation tool, environment bypass, generated code or generic shell.
Only these read operations are admitted until per-action permissions/verifiers
exist. Legacy Live/direct action entry points are outside this spike's boundary.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str


def authorize_action(tool: str, parameters: dict) -> PolicyDecision:
    if not isinstance(parameters, dict):
        return PolicyDecision(False, "Action parameters must be an object.")
    if tool == "system_time" and not parameters:
        return PolicyDecision(True, "Read the local system clock.")
    if tool == "web_search" and set(parameters) == {"query"}:
        query = parameters["query"]
        if isinstance(query, str) and 0 < len(query.strip()) <= 2000:
            return PolicyDecision(True, "Read-only search; output remains unverified.")
    if tool == "weather_report" and set(parameters) == {"city"}:
        city = parameters["city"]
        if isinstance(city, str) and 0 < len(city.strip()) <= 200:
            return PolicyDecision(True, "Read-only weather lookup; output remains unverified.")
    return PolicyDecision(False, "Capability is not admitted by the Phase 1 owner policy.")
