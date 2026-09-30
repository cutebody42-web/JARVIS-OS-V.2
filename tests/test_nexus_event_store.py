"""Offline tests for the NEXUS semantic event journal."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core.nexus.event_store import (
    ClockRelation,
    EventStore,
    SyncEvent,
    compare_vector_clocks,
)


class VectorClockTests(unittest.TestCase):
    def test_relations(self):
        self.assertEqual(compare_vector_clocks({"hp": 1}, {"hp": 1}), ClockRelation.EQUAL)
        self.assertEqual(compare_vector_clocks({"hp": 1}, {"hp": 2}), ClockRelation.BEFORE)
        self.assertEqual(compare_vector_clocks({"hp": 3}, {"hp": 2}), ClockRelation.AFTER)
        self.assertEqual(
            compare_vector_clocks({"hp": 2, "dell": 1}, {"hp": 1, "dell": 2}),
            ClockRelation.CONCURRENT,
        )
        self.assertEqual(
            compare_vector_clocks({"hp": 1}, {"hp": 1, "dell": 2}),
            ClockRelation.BEFORE,
        )

    def test_invalid_counter_is_rejected(self):
        for bad in (-1, 1.2, True):
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                compare_vector_clocks({"hp": bad}, {})


class SyncEventTests(unittest.TestCase):
    def test_round_trip_and_payload_detachment(self):
        payload = {"facts": ["a", {"nested": 1}]}
        event = SyncEvent(
            id="evt-1",
            device="hp",
            entity_id="memory:user",
            entity_type="memory.fact",
            type="memory.add",
            vclock={"hp": 1},
            payload=payload,
            timestamp="2026-09-30T14:01:23+00:00",
        )
        payload["facts"][1]["nested"] = 99
        restored = SyncEvent.from_json_line(event.to_json_line())
        self.assertEqual(restored.to_dict()["payload"]["facts"][1]["nested"], 1)
        self.assertEqual(restored.to_dict(), event.to_dict())

    def test_schema_is_strict_and_origin_clock_is_required(self):
        with self.assertRaises(ValueError):
            SyncEvent.from_dict({
                "id": "evt", "device": "hp", "entity_id": "x",
                "entity_type": "memory.fact", "type": "memory.add",
                "vclock": {"dell": 1}, "payload": {}, "timestamp": "2026-09-30T14:01:23Z",
            })
        with self.assertRaises(ValueError):
            SyncEvent.from_dict({
                "id": "evt", "device": "hp", "entity_id": "x",
                "entity_type": "memory.fact", "type": "memory.add",
                "vclock": {"hp": 1}, "payload": {}, "timestamp": "2026-09-30T14:01:23Z",
                "unexpected": "field",
            })

    def test_v1_event_can_be_read_as_legacy_unknown_type(self):
        legacy = SyncEvent.from_dict({
            "schema_version": 1,
            "id": "legacy-1",
            "device": "hp",
            "entity_id": "x",
            "type": "memory.add",
            "vclock": {"hp": 1},
            "payload": {"value": 1},
            "timestamp": "2026-09-30T14:01:23Z",
        })
        self.assertEqual(legacy.entity_type, "legacy.unknown")


class EventStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_local_event_is_durable_pending_and_counter_survives_restart(self):
        store = EventStore(self.root, "hp")
        first = store.create_event(
            "memory.add", "pref:theme", {"value": "dark"},
            entity_type="user.preference",
        )
        self.assertTrue(store.seen(first.id))
        self.assertEqual(store.pending_local_count(), 1)
        self.assertEqual(store.pending_count(), 1)
        self.assertEqual(list((self.root / "outbox").glob("*.jsonl")), [])

        restarted = EventStore(self.root, "hp")
        self.assertEqual(restarted.pending_local()[0].id, first.id)
        second = restarted.create_event(
            "memory.add", "pref:lang", {"value": "ar"},
            entity_type="user.preference",
        )
        self.assertEqual(second.vclock["hp"], first.vclock["hp"] + 1)

    def test_interrupted_export_is_recoverable_after_local_materialization(self):
        store = EventStore(self.root, "hp")
        event = store.create_event(
            "memory.add", "fact:1", {"value": 1},
            entity_type="memory.fact",
            export=False,
        )
        with store._connect() as db:
            db.execute(
                "UPDATE nexus_sync_events SET apply_state = 'applied' WHERE event_id = ?",
                (event.id,),
            )

        with patch.object(store, "_write_event_file", side_effect=OSError("disk interruption")):
            with self.assertRaises(OSError):
                store.flush_pending()
        self.assertTrue(store.seen(event.id))
        self.assertEqual(store.pending_count(), 1)

        restarted = EventStore(self.root, "hp")
        self.assertEqual(restarted.flush_pending(), 1)
        self.assertEqual(restarted.pending_count(), 0)
        self.assertEqual(len(list((self.root / "outbox").glob("*.jsonl"))), 1)

    def test_remote_dedupe_updates_causality_for_next_local_event(self):
        hp = EventStore(self.root / "hp", "hp")
        remote = SyncEvent(
            id="dell-7",
            device="dell",
            entity_id="memory:study",
            entity_type="memory.fact",
            type="memory.add",
            vclock={"dell": 7},
            payload={"value": "physics"},
            timestamp="2026-09-30T14:02:00Z",
        )
        self.assertTrue(hp.accept_remote(remote))
        self.assertFalse(hp.accept_remote(remote))
        local = hp.create_event(
            "memory.add", "memory:chemistry", {"value": "organic"},
            entity_type="memory.fact",
        )
        self.assertEqual(local.vclock["dell"], 7)
        self.assertEqual(local.vclock["hp"], 1)

    def test_inbound_event_stays_pending_across_restart_until_merge_applier_commits(self):
        hp = EventStore(self.root / "hp", "hp")
        remote = SyncEvent(
            id="dell-pending",
            device="dell",
            entity_id="memory:study",
            entity_type="memory.fact",
            type="memory.add",
            vclock={"dell": 1},
            payload={"value": "physics"},
            timestamp="2026-09-30T14:02:00Z",
        )
        self.assertTrue(hp.accept_remote(remote))
        self.assertEqual(hp.pending_apply_count(), 1)
        self.assertEqual(hp.pending_inbound()[0].id, remote.id)

        restarted = EventStore(self.root / "hp", "hp")
        self.assertEqual(restarted.pending_apply_count(), 1)
        self.assertEqual(restarted.pending_inbound()[0].id, remote.id)

    def test_loopback_event_is_not_reinserted_as_inbound(self):
        store = EventStore(self.root, "hp")
        local = store.create_event(
            "memory.add", "x", {"value": 1}, entity_type="memory.fact"
        )
        self.assertFalse(store.accept_remote(local))

    def test_tombstone_survives_spool_round_trip_after_apply(self):
        store = EventStore(self.root, "hp")
        event = store.create_event(
            "memory.delete", "memory:old", {},
            entity_type="memory.fact", tombstone=True,
            export=False,
        )
        with store._connect() as db:
            db.execute(
                "UPDATE nexus_sync_events SET apply_state = 'applied' WHERE event_id = ?",
                (event.id,),
            )
        self.assertEqual(store.flush_pending(), 1)
        recovered = SyncEvent.from_json_line(
            next((self.root / "outbox").glob("*.jsonl")).read_text("utf-8")
        )
        self.assertTrue(recovered.tombstone)
        self.assertEqual(recovered.id, event.id)


if __name__ == "__main__":
    unittest.main()
