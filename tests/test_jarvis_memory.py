"""Cross-device continuity tests for the single JARVIS memory."""

from pathlib import Path
import tempfile
import unittest

from core.jarvis_memory import JarvisMemory
from core.nexus.event_store import EventStore
from core.nexus.merge_applier import MergeApplier
from core.nexus.sync_daemon import SyncDaemon


class Loopback:
    def __init__(self, remote):
        self.remote = remote

    def send_batch(self, peer_id, batch):
        if peer_id != self.remote.device_id:
            raise ValueError("wrong peer")
        return self.remote.receive_batch(batch)


class MemoryNode:
    def __init__(self, root, device):
        self.store = EventStore(root, device)
        self.applier = MergeApplier(self.store)
        self.memory = JarvisMemory(self.store, self.applier)
        self.sync = SyncDaemon(self.store, self.applier)


class JarvisMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.hp = MemoryNode(root / "hp", "hp")
        self.phone = MemoryNode(root / "phone", "phone")

    def tearDown(self):
        self.tmp.cleanup()

    def sync_hp_to_phone(self):
        while self.hp.sync.pending_for_peer_count("phone"):
            self.hp.sync.sync_peer("phone", Loopback(self.phone.sync))

    def test_project_handoff_and_turns_follow_owner_to_phone(self):
        self.hp.memory.checkpoint_project(
            "jarvis-v2",
            summary="Building one JARVIS Brain",
            phase="productization",
            open_tasks=["mobile shell", "self-heal"],
            decisions=["local-first", "one identity"],
        )
        self.hp.memory.write_handoff(
            project_id="jarvis-v2",
            summary="Continue mobile shell after memory continuity.",
            next_actions=["build Tauri shell"],
            session_id="desktop-session",
        )
        self.hp.memory.append_turn(
            "desktop-session",
            "user",
            "Continue the JARVIS project",
            project_id="jarvis-v2",
        )
        self.hp.memory.append_turn(
            "desktop-session",
            "assistant",
            "Continuing from productization.",
            project_id="jarvis-v2",
        )

        self.sync_hp_to_phone()

        handoff = self.phone.memory.current_handoff()
        self.assertEqual(handoff["project_id"], "jarvis-v2")
        self.assertIn("mobile shell", handoff["summary"].lower())

        project = self.phone.memory.project_state("jarvis-v2")
        self.assertEqual(project.fields["phase"], "productization")
        self.assertIn("one identity", project.fields["decisions"])

        turns = self.phone.memory.recent_turns(project_id="jarvis-v2")
        self.assertEqual(
            [turn.role for turn in turns],
            ["user", "assistant"],
        )
        self.assertEqual(turns[-1].content, "Continuing from productization.")

    def test_fact_updates_are_lww_and_syncable(self):
        self.hp.memory.remember_fact("preferred_name", "Sir")
        self.sync_hp_to_phone()
        self.assertEqual(self.phone.memory.get_fact("preferred_name"), "Sir")

    def test_project_fields_merge_independently(self):
        self.hp.memory.set_project_field("p", "summary", "from hp")
        self.phone.memory.set_project_field("p", "phase", "from phone")

        self.hp.sync.sync_peer("phone", Loopback(self.phone.sync))
        self.phone.sync.sync_peer("hp", Loopback(self.hp.sync))

        self.assertEqual(self.hp.memory.project_state("p").fields, {
            "phase": "from phone",
            "summary": "from hp",
        })
        self.assertEqual(self.phone.memory.project_state("p").fields, {
            "phase": "from phone",
            "summary": "from hp",
        })

    def test_continuity_context_contains_handoff_project_and_turns(self):
        self.hp.memory.set_project_field("alpha", "summary", "Project Alpha")
        self.hp.memory.write_handoff(
            project_id="alpha",
            summary="Resume Alpha",
            next_actions=["next step"],
        )
        self.hp.memory.append_turn("s", "user", "Where were we?", project_id="alpha")

        context = self.hp.memory.continuity_context()
        self.assertIn("Resume Alpha", context)
        self.assertIn("Project Alpha", context)
        self.assertIn("Where were we?", context)


if __name__ == "__main__":
    unittest.main()
