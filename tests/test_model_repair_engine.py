import hashlib
import json
import unittest

from core.model_provider import ModelResponse
from core.model_repair_engine import (
    ModelRepairEngine,
    ModelRepairError,
    build_repair_context,
)
from core.self_heal import Incident, IncidentKind


class FakeProvider:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        text = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return ModelResponse(text=text, provider="test", model="repair-test")


def proposal(path="core/example.py", replacement="VALUE = 2\n"):
    return {
        "schema": "jarvis.repair-proposal.v1",
        "rationale": "Correct the bounded regression.",
        "edits": [{"path": path, "replacement": replacement}],
        "requested_tests": ["tests.test_example"],
    }


class ModelRepairEngineTests(unittest.TestCase):
    def setUp(self):
        self.incident = Incident.create(
            IncidentKind.CODE_FAILURE,
            "VALUE should be two",
            "core/example",
            {"test": "tests.test_example"},
        )
        self.original = "VALUE = 1\n"
        self.context = build_repair_context({"core/example.py": self.original})

    def test_host_binds_edit_to_supplied_content_hash(self):
        provider = FakeProvider(proposal())
        patch = ModelRepairEngine(provider).propose(self.incident, self.context)
        self.assertEqual(patch.incident_id, self.incident.id)
        self.assertEqual(patch.edits[0].path, "core/example.py")
        self.assertEqual(
            patch.edits[0].expected_sha256,
            hashlib.sha256(self.original.encode()).hexdigest(),
        )
        self.assertEqual(patch.edits[0].replacement, "VALUE = 2\n")
        request = provider.requests[0]
        self.assertTrue(request.json_output)
        self.assertIn("zero execution or permission authority", request.system_instruction)

    def test_model_cannot_edit_file_not_in_context(self):
        provider = FakeProvider(proposal(path="core/owner_kernel.py"))
        with self.assertRaises(ModelRepairError):
            ModelRepairEngine(provider).propose(self.incident, self.context)

    def test_model_cannot_supply_its_own_hash_or_extra_fields(self):
        payload = proposal()
        payload["edits"][0]["expected_sha256"] = "0" * 64
        provider = FakeProvider(payload)
        with self.assertRaises(ModelRepairError):
            ModelRepairEngine(provider).propose(self.incident, self.context)

    def test_markdown_or_invalid_json_is_rejected(self):
        provider = FakeProvider("```json\n{}\n```")
        with self.assertRaises(ModelRepairError):
            ModelRepairEngine(provider).propose(self.incident, self.context)

    def test_no_op_duplicate_and_oversize_replacements_are_rejected(self):
        with self.assertRaises(ModelRepairError):
            ModelRepairEngine(FakeProvider(proposal(replacement=self.original))).propose(
                self.incident, self.context
            )

        duplicate = proposal()
        duplicate["edits"].append(dict(duplicate["edits"][0]))
        with self.assertRaises(ModelRepairError):
            ModelRepairEngine(FakeProvider(duplicate)).propose(self.incident, self.context)

        huge = proposal(replacement="x" * 256_001)
        with self.assertRaises(ModelRepairError):
            ModelRepairEngine(FakeProvider(huge)).propose(self.incident, self.context)

    def test_context_hash_mismatch_and_path_escape_are_rejected(self):
        parsed = json.loads(self.context)
        parsed["files"][0]["sha256"] = "0" * 64
        with self.assertRaises(ModelRepairError):
            ModelRepairEngine(FakeProvider(proposal())).propose(
                self.incident, json.dumps(parsed)
            )
        with self.assertRaises(ValueError):
            build_repair_context({"../outside.py": "bad"})

    def test_context_is_bounded_and_has_exact_schema(self):
        with self.assertRaises(ValueError):
            build_repair_context({f"core/f{i}.py": "x" for i in range(13)})
        invalid = json.loads(self.context)
        invalid["files"][0]["extra"] = True
        with self.assertRaises(ModelRepairError):
            ModelRepairEngine(FakeProvider(proposal())).propose(
                self.incident, json.dumps(invalid)
            )

    def test_requested_tests_are_bounded_text(self):
        payload = proposal()
        payload["requested_tests"] = ["x" * 161]
        with self.assertRaises(ModelRepairError):
            ModelRepairEngine(FakeProvider(payload)).propose(self.incident, self.context)


if __name__ == "__main__":
    unittest.main()
