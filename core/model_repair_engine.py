"""Provider-assisted repair proposals with host-owned validation.

A model may suggest replacement text for a small, explicitly supplied set of
repository files. The model cannot choose repository state, fabricate base
hashes, create files outside the supplied context, execute code, dispatch
actions, or approve its own proposal. The host binds every accepted edit to the
SHA-256 of the exact context it supplied, then the existing SelfHealPolicy,
GitHub repair coordinator, CI gates, and owner approval remain authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

from core.model_provider import ModelProvider, ModelRequest, ModelTier
from core.self_heal import FileEdit, Incident, RepairPatch, repository_path


_CONTEXT_SCHEMA = "jarvis.repair-context.v1"
_OUTPUT_SCHEMA = "jarvis.repair-proposal.v1"
_MAX_CONTEXT_FILES = 12
_MAX_CONTEXT_BYTES = 512_000
_MAX_EDIT_FILES = 4
_MAX_REPLACEMENT_BYTES = 256_000
_MAX_RATIONALE = 4000
_MAX_TESTS = 16


class ModelRepairError(RuntimeError):
    """Raised when a model repair proposal is malformed or exceeds bounds."""


@dataclass(frozen=True)
class RepairContextFile:
    path: str
    sha256: str
    content: str

    def __post_init__(self) -> None:
        repository_path(self.path)
        if not isinstance(self.content, str):
            raise TypeError("repair context content must be text")
        digest = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.sha256 != digest:
            raise ValueError("repair context SHA-256 does not match content")


def build_repair_context(files: Mapping[str, str]) -> str:
    """Create the only repository context accepted by ``ModelRepairEngine``.

    Context is deliberately explicit and bounded. Models receive only these
    files and can propose edits only to these exact paths.
    """
    if not isinstance(files, Mapping) or not files:
        raise ValueError("at least one repair context file is required")
    if len(files) > _MAX_CONTEXT_FILES:
        raise ValueError("repair context contains too many files")

    rows: list[dict[str, str]] = []
    total = 0
    seen: set[str] = set()
    for path, content in files.items():
        clean_path = repository_path(path)
        folded = clean_path.casefold()
        if folded in seen:
            raise ValueError("repair context contains duplicate or case-colliding paths")
        seen.add(folded)
        if not isinstance(content, str):
            raise TypeError("repair context content must be text")
        encoded = content.encode("utf-8")
        total += len(encoded)
        if total > _MAX_CONTEXT_BYTES:
            raise ValueError("repair context is too large")
        rows.append(
            {
                "path": clean_path,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "content": content,
            }
        )

    return json.dumps(
        {"schema": _CONTEXT_SCHEMA, "files": rows},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_context(raw: str) -> tuple[RepairContextFile, ...]:
    if not isinstance(raw, str) or not raw:
        raise ModelRepairError("repository context must be a non-empty JSON document")
    if len(raw.encode("utf-8")) > _MAX_CONTEXT_BYTES * 2:
        raise ModelRepairError("repository context is too large")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelRepairError("repository context is not valid JSON") from exc
    if not isinstance(value, dict) or value.get("schema") != _CONTEXT_SCHEMA:
        raise ModelRepairError("repository context has an unsupported schema")
    rows = value.get("files")
    if not isinstance(rows, list) or not 1 <= len(rows) <= _MAX_CONTEXT_FILES:
        raise ModelRepairError("repository context has an invalid file list")

    parsed: list[RepairContextFile] = []
    seen: set[str] = set()
    total = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "content"}:
            raise ModelRepairError("repository context file entry is malformed")
        try:
            item = RepairContextFile(
                path=row["path"], sha256=row["sha256"], content=row["content"]
            )
        except (TypeError, ValueError) as exc:
            raise ModelRepairError("repository context file entry failed validation") from exc
        folded = item.path.casefold()
        if folded in seen:
            raise ModelRepairError("repository context contains duplicate paths")
        seen.add(folded)
        total += len(item.content.encode("utf-8"))
        if total > _MAX_CONTEXT_BYTES:
            raise ModelRepairError("repository context is too large")
        parsed.append(item)
    return tuple(parsed)


_SYSTEM = """You are a code-repair proposal generator inside JARVIS NEXUS.
You have zero execution or permission authority. Return one JSON object only.
Never use markdown fences. Never invent files, hashes, commands, approvals,
credentials, secrets, shell steps, workflow permission changes, or repository
settings. You may propose complete replacement text only for files explicitly
present in the supplied repair context.

Required output schema:
{
  "schema": "jarvis.repair-proposal.v1",
  "rationale": "short explanation",
  "edits": [{"path": "exact/context/path", "replacement": "full replacement text"}],
  "requested_tests": ["specific test name"]
}
"""


class ModelRepairEngine:
    """Strict ``RepairEngine`` implementation over the neutral ModelProvider."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        tier: ModelTier = ModelTier.STANDARD,
    ) -> None:
        if not isinstance(provider, ModelProvider):
            raise TypeError("provider must implement ModelProvider")
        if not isinstance(tier, ModelTier):
            raise TypeError("tier must be ModelTier")
        self.provider = provider
        self.tier = tier

    def propose(self, incident: Incident, repository_context: str) -> RepairPatch:
        if not isinstance(incident, Incident):
            raise TypeError("incident must be Incident")
        files = _parse_context(repository_context)
        by_path = {item.path: item for item in files}

        prompt = json.dumps(
            {
                "incident": {
                    "id": incident.id,
                    "kind": incident.kind.value,
                    "summary": incident.summary,
                    "component": incident.component,
                    "evidence": dict(incident.evidence),
                },
                "repair_context": json.loads(repository_context),
                "limits": {
                    "max_edit_files": _MAX_EDIT_FILES,
                    "max_total_replacement_bytes": _MAX_REPLACEMENT_BYTES,
                    "authority": "proposal_only",
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        response = self.provider.generate(
            ModelRequest(
                prompt=prompt,
                system_instruction=_SYSTEM,
                tier=self.tier,
                json_output=True,
            )
        )
        if not isinstance(response.text, str) or not response.text.strip():
            raise ModelRepairError("model returned an empty repair proposal")
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise ModelRepairError("model repair proposal is not valid JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "schema", "rationale", "edits", "requested_tests"
        }:
            raise ModelRepairError("model repair proposal has unexpected fields")
        if payload.get("schema") != _OUTPUT_SCHEMA:
            raise ModelRepairError("model repair proposal has an unsupported schema")

        rationale = payload.get("rationale")
        if not isinstance(rationale, str) or not 1 <= len(rationale.strip()) <= _MAX_RATIONALE:
            raise ModelRepairError("model repair rationale is invalid")
        rows = payload.get("edits")
        if not isinstance(rows, list) or not 1 <= len(rows) <= _MAX_EDIT_FILES:
            raise ModelRepairError("model repair edit count is invalid")

        edits: list[FileEdit] = []
        seen: set[str] = set()
        total = 0
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"path", "replacement"}:
                raise ModelRepairError("model repair edit is malformed")
            path = row.get("path")
            replacement = row.get("replacement")
            if not isinstance(path, str) or path not in by_path:
                raise ModelRepairError("model proposed a file outside the supplied context")
            folded = path.casefold()
            if folded in seen:
                raise ModelRepairError("model proposed duplicate edits")
            seen.add(folded)
            if not isinstance(replacement, str):
                raise ModelRepairError("model replacement must be text")
            total += len(replacement.encode("utf-8"))
            if total > _MAX_REPLACEMENT_BYTES:
                raise ModelRepairError("model repair replacements are too large")
            source = by_path[path]
            if replacement == source.content:
                raise ModelRepairError("model repair contains a no-op replacement")
            edits.append(FileEdit(path, source.sha256, replacement))

        tests = payload.get("requested_tests")
        if not isinstance(tests, list) or len(tests) > _MAX_TESTS:
            raise ModelRepairError("model repair test list is invalid")
        requested: list[str] = []
        for test in tests:
            if not isinstance(test, str) or not 1 <= len(test.strip()) <= 160:
                raise ModelRepairError("model repair test name is invalid")
            requested.append(test.strip())

        return RepairPatch(
            incident_id=incident.id,
            edits=tuple(edits),
            rationale=rationale.strip(),
            requested_tests=tuple(dict.fromkeys(requested)),
        )
