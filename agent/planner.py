import json
import re

from core.model_provider import ModelProvider, ModelRequest, ModelTier, default_provider


PLANNER_PROMPT = """You are the NEXUS planning module. Propose at most five independent steps.
Application-owned policy, not this plan, decides which actions may run.
Phase 1 admits only these read operations:
- system_time: parameters {} (read this machine's local system clock)
- web_search: parameters {"query": "..."} (read-only search)
- weather_report: parameters {"city": "..."} (read-only weather lookup)
Never propose generated code, shell execution, privilege escalation, replication,
policy changes or mutations. Do not claim completion or invent evidence.
If no admitted tool can satisfy the goal, return an empty steps array.
Return only JSON: {"goal": "...", "steps": [{"step": 1, "tool": "...",
"description": "...", "parameters": {}, "critical": true}]}.
"""


def validate_plan(value: object, goal: str) -> dict:
    """Validate shape and bounds, never accept model-provided receipts or policy."""
    if not isinstance(value, dict) or set(value) - {"goal", "steps"}:
        raise ValueError("Invalid plan object")
    steps = value.get("steps")
    if not isinstance(steps, list) or len(steps) > 5:
        raise ValueError("A plan must contain at most five steps")
    seen = set()
    clean = []
    for step in steps:
        if not isinstance(step, dict) or set(step) - {"step", "tool", "description", "parameters", "critical"}:
            raise ValueError("Invalid step object")
        number = step.get("step")
        if type(number) is not int or number < 1 or number in seen:
            raise ValueError("Step IDs must be unique positive integers")
        if not isinstance(step.get("tool"), str) or not step["tool"].strip():
            raise ValueError("A tool name is required")
        if not isinstance(step.get("parameters"), dict):
            raise ValueError("Parameters must be an object")
        if not isinstance(step.get("description", ""), str) or type(step.get("critical", True)) is not bool:
            raise ValueError("Invalid step metadata")
        seen.add(number)
        clean.append({**step, "critical": step.get("critical", True)})
    # Strict JSON also rejects NaN and snapshots nested parameters.
    return json.loads(json.dumps({"goal": goal, "steps": clean}, allow_nan=False))


def _request_plan(prompt: str, goal: str, provider: ModelProvider | None, tier: ModelTier) -> dict:
    try:
        response = (provider if provider is not None else default_provider()).generate(
            ModelRequest(prompt, PLANNER_PROMPT, tier, json_output=True)
        )
        text = response.text.strip()
        if text.startswith("```json") and text.endswith("```"):
            text = text[7:-3].strip()
        elif text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
        return validate_plan(json.loads(text), goal)
    except Exception:
        # No arbitrary fallback search, generated execution, or error/secret echo.
        return {"goal": goal, "steps": []}


def create_plan(goal: str, context: str = "", *, provider: ModelProvider | None = None) -> dict:
    from agent.reflex import reflex_plan

    local = reflex_plan(goal)
    if local is not None:
        return local
    prompt = f"Goal: {goal}"
    if context:
        prompt += f"\n\nContext: {context}"
    return _request_plan(prompt, goal, provider, ModelTier.FAST)


def _fallback_plan(goal: str) -> dict:
    print("[Planner] 🔄 Fallback plan")
    if re.search(r"\b(email|emails|e-mail|inbox)\b", goal, re.I):
        action = "inbox"
        parameters = {"action": action, "limit": 10}
        if re.search(r"\b(send|compose|write)\b", goal, re.I):
            parameters = {"action": "prepare", "provider": "gmail", "browser": "chrome"}
            address = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", goal)
            subject = re.search(r"(?i)\bsubject\s+(.+?)(?=\s+\b(?:saying|body)\b|$)", goal)
            body = re.search(r"(?i)\b(?:saying|body)\s+(.+)$", goal)
            if address:
                parameters["to"] = address.group(0)
            if subject:
                parameters["subject"] = subject.group(1).strip(" .\"'")
            if body:
                parameters["body"] = body.group(1).strip(" .\"'")
        elif re.search(r"\bunread\b", goal, re.I):
            parameters["action"] = "unread"
        search_match = re.search(r"(?i)\b(?:search|find)\s+(?:my\s+)?(?:email|emails|inbox)\s+(?:for|from)\s+(.+)$", goal)
        if search_match and parameters["action"] != "prepare":
            parameters = {"action": "search", "query": search_match.group(1).strip(" .")}
        return {
            "goal": goal,
            "steps": [{
                "step": 1,
                "tool": "email_control",
                "description": "Access the requested email information.",
                "parameters": parameters,
                "critical": True,
            }],
        }
    media_match = re.search(
        r"\b(spotify|apple\s+music|youtube\s+music|music|song|track|playback)\b",
        goal,
        re.I,
    )
    media_command = re.search(
        r"\b(play|resume|pause|stop|toggle|skip|next|previous|back)\b",
        goal,
        re.I,
    )
    if media_match and media_command:
        spoken_action = media_command.group(1).lower()
        action = {
            "resume": "play",
            "skip": "next",
            "back": "previous",
        }.get(spoken_action, spoken_action)
        platform_name = "spotify"
        if re.search(r"\bapple\s+music\b", goal, re.I):
            platform_name = "apple_music"
        elif re.search(r"\byoutube\s+music\b", goal, re.I):
            platform_name = "youtube_music"
        parameters = {"action": action, "platform": platform_name}
        if action == "play":
            query = re.sub(
                r"(?i)^.*?\bplay\b|\b(?:on|in|using)\s+(?:spotify|apple\s+music|youtube\s+music)\b.*$",
                "",
                goal,
            )
            query = re.sub(r"(?i)\bplease\b", "", query).strip(" .,")
            if query and query.lower() not in {"music", "song", "a song", "spotify"}:
                parameters["action"] = "play_query"
                parameters["query"] = query
        return {
            "goal": goal,
            "steps": [{
                "step": 1,
                "tool": "media_control",
                "description": "Control the requested music playback.",
                "parameters": parameters,
                "critical": True,
            }],
        }
    if re.search(r"\b(power\s*point|powerpoint|presentation|slide\s+deck|pitch\s+deck|slideshow)\b", goal, re.I):
        count_match = re.search(r"\b(\d{1,2})[ -]slide\b", goal, re.I)
        parameters = {"topic": goal, "execution_mode": "ask"}
        if re.search(r"\b(?:dark|black background|midnight)\b", goal, re.I):
            parameters["appearance"] = "dark"
        elif re.search(r"\b(?:light|white background|bright|airy)\b", goal, re.I):
            parameters["appearance"] = "light"
        if re.search(r"\b(?:3d|three[- ]dimensional)\b.*\bmodel|\bmodel.*\b(?:3d|powerpoint)\b", goal, re.I):
            parameters["use_native_3d"] = True
        if count_match:
            parameters["slide_count"] = max(3, min(50, int(count_match.group(1))))
        return {
            "goal": goal,
            "steps": [
                {
                    "step": 1,
                    "tool": "create_presentation",
                    "description": "Create the requested editable PowerPoint presentation.",
                    "parameters": parameters,
                    "critical": True,
                }
            ],
        }
    if re.search(r"\b(deep|thorough|comprehensive|source[- ]backed)\s+(?:web\s+)?research\b", goal, re.I):
        depth = "deep" if re.search(r"\b(deep|comprehensive)\b", goal, re.I) else "standard"
        return {
            "goal": goal,
            "steps": [
                {
                    "step": 1,
                    "tool": "deep_research",
                    "description": "Ask how the source-grounded deep research should be displayed.",
                    "parameters": {"question": goal, "depth": depth, "execution_mode": "ask"},
                    "critical": True,
                }
            ],
        }
    return {
        "goal": goal,
        "steps": [
            {
                "step": 1,
                "tool": "web_search",
                "description": f"Search for: {goal}",
                "parameters": {"query": goal},
                "critical": True
            }
        ]
    }


def replan(goal: str, completed_steps: list, failed_step: dict, error: str,
           *, provider: ModelProvider | None = None) -> dict:
    prompt = (f"Goal: {goal}\nVerified completed steps: {json.dumps(completed_steps)}\n"
              f"Failed step: {json.dumps(failed_step)}\nError: {error[:500]}\n"
              "Propose remaining work only. Do not repeat verified steps.")
    return _request_plan(prompt, goal, provider, ModelTier.STANDARD)
