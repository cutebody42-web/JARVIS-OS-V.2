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
                "id": "evt", "device": "hp", "entity_id": "x", "type": "memory.add",
                "vclock": {"dell": 1}, "payload": {}, "timestamp": "2026-09-30T14:01:23Z",
            })
        with self.assertRaises(ValueError):
            SyncEvent.from_dict({
                "id": "evt", "device": "hp", "entity_id": "x", "type": "memory.add",
                "vclock": {"hp": 1}, "payload": {}, "timestamp": "2026-09-30T14:01:23Z",
                "unexpected": "field",
            })


class EventStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_local_event_is_durable_exported_and_counter_survives_restart(self):
        store = EventStore(self.root, "hp")
        first = store.create_event("memory.add", "pref:theme", {"value": "dark"})
        self.assertTrue(store.seen(first.id))
        self.assertEqual(store.pending_count(), 0)
        files = list((self.root / "outbox").glob("*.jsonl"))
        self.assertEqual(len(files), 1)
        self.assertEqual(SyncEvent.from_json_line(files[0].read_text("utf-8")).id, first.id)

        restarted = EventStore(self.root, "hp")
        second = restarted.create_event("memory.add", "pref:lang", {"value": "ar"})
        self.assertEqual(second.vclock["hp"], first.vclock["hp"] + 1)

    def test_interrupted_export_is_recoverable_without_event_loss(self):
        store = EventStore(self.root, "hp")
        with patch.object(store, "_write_event_file", side_effect=OSError("disk interruption")):
            with self.assertRaises(OSError):
                event = store.create_event("memory.add", "fact:1", {"value": 1})
        # Retrieve the durable event because create_event intentionally raised during export.
        with store._connect() as db:
            event_id = db.execute(
                "SELECT event_id FROM nexus_sync_events WHERE exported = 0"
            ).fetchone()["event_id"]
        self.assertTrue(store.seen(event_id))
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
            type="memory.add",
            vclock={"dell": 7},
            payload={"value": "physics"},
            timestamp="2026-09-30T14:02:00Z",
        )
        self.assertTrue(hp.accept_remote(remote))
        self.assertFalse(hp.accept_remote(remote))
        local = hp.create_event("memory.add", "memory:chemistry", {"value": "organic"})
        self.assertEqual(local.vclock["dell"], 7)
        self.assertEqual(local.vclock["hp"], 1)

    def test_loopback_event_is_not_reinserted_as_inbound(self):
        store = EventStore(self.root, "hp")
        local = store.create_event("memory.add", "x", {"v": 1})
        self.assertFalse(store.accept_remote(local))

    def test_tombstone_survives_spool_round_trip(self):
        store = EventStore(self.root, "hp")
        event = store.create_event("memory.delete", "memory:old", {}, tombstone=True)
        recovered = SyncEvent.from_json_line(next((self.root / "outbox").glob("*.jsonl")).read_text("utf-8"))
        self.assertTrue(recovered.tombstone)
        self.assertEqual(recovered.id, event.id)


if __name__ == "__main__":
    unittest.main()
