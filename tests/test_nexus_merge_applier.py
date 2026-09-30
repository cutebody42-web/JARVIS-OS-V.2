"""Transactional tests for the NEXUS merge applier."""

from pathlib import Path
import tempfile
import unittest

from core.nexus.conflict_resolver import materialize_orset
from core.nexus.event_store import EventStore, SyncEvent
from core.nexus.merge_applier import ApplyStatus, MergeApplier


def remote_event(
    *,
    event_id,
    device="dell",
    entity_id="memory:1",
    entity_type="memory.fact",
    event_type="memory.upsert",
    vclock=None,
    payload=None,
    timestamp="2026-09-30T14:00:00Z",
    tombstone=False,
):
    return SyncEvent(
        id=event_id,
        device=device,
        entity_id=entity_id,
        entity_type=entity_type,
        type=event_type,
        vclock=vclock or {device: 1},
        payload=payload or {},
        timestamp=timestamp,
        tombstone=tombstone,
    )


class MergeApplierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = EventStore(self.root, "hp")

    def tearDown(self):
        self.tmp.cleanup()

    def test_local_event_is_pending_until_materialized_then_exported(self):
        event = self.store.create_event(
            "memory.upsert",
            "memory:local",
            {"value": "local"},
            entity_type="memory.fact",
        )
        applier = MergeApplier(self.store)

        self.assertEqual(self.store.pending_local_count(), 1)
        self.assertEqual(list((self.root / "outbox").glob("*.jsonl")), [])

        result = applier.apply_local_event(event)

        self.assertEqual(result.status, ApplyStatus.APPLIED)
        self.assertEqual(self.store.pending_local_count(), 0)
        self.assertEqual(applier.get_snapshot("memory:local").value, "local")
        self.assertEqual(len(list((self.root / "outbox").glob("*.jsonl"))), 1)

    def test_first_apply_mutates_state_and_marks_event_applied_atomically(self):
        event = remote_event(
            event_id="dell-1",
            payload={"value": "physics"},
        )
        self.assertTrue(self.store.accept_remote(event))

        applier = MergeApplier(self.store)
        result = applier.apply_event(event)

        self.assertEqual(result.status, ApplyStatus.APPLIED)
        self.assertEqual(applier.get_apply_state(event.id), "applied")
        self.assertEqual(self.store.pending_apply_count(), 0)
        snapshot = applier.get_snapshot(event.entity_id)
        self.assertEqual(snapshot.value, "physics")
        self.assertEqual(snapshot.last_event_id, event.id)
        evidence = applier.get_evidence(event.id)
        self.assertEqual(evidence["action"], "apply")

    def test_replay_after_success_is_exactly_once_for_local_materialization(self):
        event = remote_event(
            event_id="dell-once",
            payload={"value": "once"},
        )
        self.store.accept_remote(event)
        applier = MergeApplier(self.store)
        first = applier.apply_event(event)
        second = applier.apply_event(event)

        self.assertEqual(first.status, ApplyStatus.APPLIED)
        self.assertEqual(second.status, ApplyStatus.ALREADY_APPLIED)
        self.assertEqual(applier.get_snapshot(event.entity_id).value, "once")

    def test_crash_before_commit_rolls_back_state_evidence_and_apply_flag(self):
        event = remote_event(
            event_id="dell-crash",
            payload={"value": "survives"},
        )
        self.store.accept_remote(event)

        def crash(_db, _event, _decision):
            raise RuntimeError("simulated process death before commit")

        applier = MergeApplier(self.store, before_commit_hook=crash)
        with self.assertRaises(RuntimeError):
            applier.apply_event(event)

        # The same SQLite transaction protects all three writes.
        self.assertEqual(applier.get_apply_state(event.id), "pending")
        self.assertIsNone(applier.get_snapshot(event.entity_id))
        self.assertIsNone(applier.get_evidence(event.id))

        # Restart/replay materializes the event once.
        restarted_store = EventStore(self.root, "hp")
        restarted = MergeApplier(restarted_store)
        results = restarted.apply_pending()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, ApplyStatus.APPLIED)
        self.assertEqual(restarted.get_apply_state(event.id), "applied")
        self.assertEqual(restarted.get_snapshot(event.entity_id).value, "survives")
        self.assertEqual(restarted_store.pending_apply_count(), 0)

        self.assertEqual(
            restarted.apply_event(event).status,
            ApplyStatus.ALREADY_APPLIED,
        )

    def test_concurrent_remote_events_merge_through_materialized_state(self):
        hp_skill = remote_event(
            event_id="hp-skill",
            device="hp2",
            entity_id="skills:user",
            entity_type="skills.learned",
            event_type="skills.add",
            vclock={"hp2": 1},
            payload={"element": "python"},
        )
        dell_skill = remote_event(
            event_id="dell-skill",
            device="dell",
            entity_id="skills:user",
            entity_type="skills.learned",
            event_type="skills.add",
            vclock={"dell": 1},
            payload={"element": "rust"},
        )
        self.store.accept_remote(hp_skill)
        self.store.accept_remote(dell_skill)

        applier = MergeApplier(self.store)
        results = applier.apply_pending()
        self.assertEqual(
            {result.status for result in results},
            {ApplyStatus.APPLIED, ApplyStatus.MERGED},
        )
        self.assertEqual(
            materialize_orset(applier.get_snapshot("skills:user")),
            ("python", "rust"),
        )

    def test_integrity_reject_marks_failed_without_mutating_materialized_state(self):
        first = remote_event(
            event_id="event-a",
            payload={"value": "safe"},
            vclock={"dell": 1},
        )
        collision = remote_event(
            event_id="event-b",
            payload={"value": "corrupt"},
            vclock={"dell": 1},
        )
        self.store.accept_remote(first)
        self.store.accept_remote(collision)

        applier = MergeApplier(self.store)
        self.assertEqual(applier.apply_event(first).status, ApplyStatus.APPLIED)
        rejected = applier.apply_event(collision)

        self.assertEqual(rejected.status, ApplyStatus.REJECTED)
        self.assertEqual(applier.get_apply_state(collision.id), "failed")
        self.assertEqual(applier.get_snapshot(first.entity_id).value, "safe")
        evidence = applier.get_evidence(collision.id)
        self.assertEqual(evidence["action"], "reject")

    def test_unknown_entity_policy_stays_pending_for_future_policy_update(self):
        event = remote_event(
            event_id="future-1",
            entity_type="future.unregistered",
            payload={"value": 1},
        )
        self.store.accept_remote(event)
        applier = MergeApplier(self.store)

        with self.assertRaises(KeyError):
            applier.apply_event(event)

        self.assertEqual(applier.get_apply_state(event.id), "pending")
        self.assertIsNone(applier.get_snapshot(event.entity_id))

    def test_tombstone_is_materialized_transactionally(self):
        create = remote_event(
            event_id="create-1",
            payload={"value": "old"},
            vclock={"dell": 1},
        )
        delete = remote_event(
            event_id="delete-2",
            event_type="memory.delete",
            payload={},
            vclock={"dell": 2},
            tombstone=True,
            timestamp="2026-09-30T14:01:00Z",
        )
        self.store.accept_remote(create)
        self.store.accept_remote(delete)

        applier = MergeApplier(self.store)
        applier.apply_pending()
        snapshot = applier.get_snapshot(create.entity_id)
        self.assertTrue(snapshot.tombstone)
        self.assertIsNone(snapshot.value)
        self.assertEqual(snapshot.last_event_id, delete.id)


if __name__ == "__main__":
    unittest.main()
