"""An exact, bounded local route. Compound or ambiguous requests do not match."""

import re


_TIME_COMMANDS = frozenset({
    "what time is it", "what is the time", "what's the time", "time",
    "الساعة كام", "الساعه كام", "كم الساعة",
})


def reflex_plan(goal: str) -> dict | None:
    normalized = re.sub(r"\s+", " ", str(goal or "").strip().casefold()).rstrip("?؟.").strip()
    if normalized not in _TIME_COMMANDS:
        return None
    return {"goal": goal, "steps": [{
        "step": 1, "tool": "system_time", "description": "Read the local system clock",
        "parameters": {}, "critical": True,
    }]}
