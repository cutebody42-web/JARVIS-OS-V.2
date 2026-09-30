"""Two-node convergence tests for the transport-neutral NEXUS sync daemon."""

from pathlib import Path
import tempfile
import unittest

from core.nexus.event_store import EventStore
from core.nexus.merge_applier import MergeApplier
from core.nexus.sync_daemon import SyncAck, SyncBatch, SyncDaemon


class LoopbackTransport:
    def __init__(self, remote: SyncDaemon, *, drop_first_ack: bool = False):
        self.remote = remote
        self.drop_first_ack = drop_first_ack
        self.calls = 0
        self.batches = []

    def send_batch(self, peer_id: str, batch: SyncBatch) -> SyncAck:
        self.calls += 1
        self.batches.append(batch)
        if peer_id != self.remote.device_id:
            raise ValueError("wrong peer")
        ack = self.remote.receive_batch(batch)
        if self.drop_first_ack and self.calls == 1:
            raise ConnectionError("simulated ACK loss after durable remote receive")
        return ack


class Node:
    def __init__(self, root: Path, device_id: str, *, batch_events=50):
        self.store = EventStore(root, device_id)
        self.applier = MergeApplier(self.store)
        self.sync = SyncDaemon(
            self.store,
            self.applier,
            max_batch_events=batch_events,
        )

    def author(
        self,
        *,
        event_type="memory.upsert",
        entity_id="memory:1",
        entity_type="memory.fact",
        payload=None,
        tombstone=False,
        timestamp=None,
    ):
        event = self.store.create_event(
            event_type,
            entity_id,
            payload or {},
            entity_type=entity_type,
            tombstone=tombstone,
            timestamp=timestamp,
        )
        self.applier.apply_local_event(event)
        return event


class SyncDaemonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.hp = Node(root / "hp", "hp")
        self.dell = Node(root / "dell", "dell")

    def tearDown(self):
        self.tmp.cleanup()

    def test_local_event_is_not_sync_eligible_until_materialized(self):
        event = self.hp.store.create_event(
            "memory.upsert",
            "memory:gate",
            {"value": "draft"},
            entity_type="memory.fact",
        )
        self.assertEqual(self.hp.sync.pending_for_peer_count("dell"), 0)

        self.hp.applier.apply_local_event(event)

        self.assertEqual(self.hp.sync.pending_for_peer_count("dell"), 1)

    def test_basic_delivery_acks_only_after_remote_durable_accept(self):
        event = self.hp.author(payload={"value": "physics"})
        transport = LoopbackTransport(self.dell.sync)

        report = self.hp.sync.sync_peer("dell", transport)

        self.assertEqual(report.selected, 1)
        self.assertEqual(report.acked, 1)
        self.assertEqual(report.remaining, 0)
        state = self.hp.sync.delivery_state("dell", event.id)
        self.assertEqual(state["state"], "acked")
        self.assertEqual(state["attempts"], 1)
        self.assertEqual(
            self.dell.applier.get_snapshot(event.entity_id).value,
            "physics",
        )

    def test_lost_ack_causes_safe_retransmission_without_double_apply(self):
        event = self.hp.author(
            entity_id="usage:1",
            entity_type="usage.stats",
            event_type="usage.p",
            payload={"value": 3},
        )
        transport = LoopbackTransport(self.dell.sync, drop_first_ack=True)

        first = self.hp.sync.sync_peer("dell", transport)
        self.assertEqual(first.acked, 0)
        self.assertIsNotNone(first.error)
        self.assertEqual(self.hp.sync.pending_for_peer_count("dell"), 1)

        second = self.hp.sync.sync_peer("dell", transport)
        self.assertEqual(second.acked, 1)
        self.assertEqual(second.remaining, 0)

        state = self.hp.sync.delivery_state("dell", event.id)
        self.assertEqual(state["attempts"], 2)
        remote = self.dell.applier.get_snapshot("usage:1")
        self.assertEqual(remote.value["p"]["hp"], 3)

    def test_two_node_concurrent_lww_converges_after_bidirectional_sync(self):
        hp_event = self.hp.author(
            entity_id="pref:theme",
            entity_type="user.preference",
            payload={"value": "dark"},
            timestamp="2026-09-30T12:00:00Z",
        )
        dell_event = self.dell.author(
            entity_id="pref:theme",
            entity_type="user.preference",
            payload={"value": "light"},
            timestamp="2026-09-30T12:00:00Z",
        )
        self.assertNotEqual(hp_event.id, dell_event.id)

        self.hp.sync.sync_peer("dell", LoopbackTransport(self.dell.sync))
        self.dell.sync.sync_peer("hp", LoopbackTransport(self.hp.sync))

        hp_state = self.hp.applier.get_snapshot("pref:theme")
        dell_state = self.dell.applier.get_snapshot("pref:theme")
        self.assertEqual(hp_state.value, dell_state.value)
        self.assertEqual(hp_state.last_event_id, dell_state.last_event_id)
        self.assertEqual(hp_state.vclock, {"dell": 1, "hp": 1})
        self.assertEqual(dell_state.vclock, {"dell": 1, "hp": 1})

    def test_two_node_orset_adds_converge(self):
        self.hp.author(
            entity_id="skills:user",
            entity_type="skills.learned",
            event_type="skills.add",
            payload={"element": "python"},
        )
        self.dell.author(
            entity_id="skills:user",
            entity_type="skills.learned",
            event_type="skills.add",
            payload={"element": "rust"},
        )

        self.hp.sync.sync_peer("dell", LoopbackTransport(self.dell.sync))
        self.dell.sync.sync_peer("hp", LoopbackTransport(self.hp.sync))

        hp_value = self.hp.applier.get_snapshot("skills:user").value
        dell_value = self.dell.applier.get_snapshot("skills:user").value
        self.assertEqual(hp_value, dell_value)

    def test_backpressure_sends_only_one_bounded_batch_per_call(self):
        root = Path(self.tmp.name)
        sender = Node(root / "small-batch", "sender", batch_events=2)
        receiver = Node(root / "receiver", "receiver")
        for index in range(5):
            sender.author(
                entity_id=f"memory:{index}",
                payload={"value": index},
            )

        transport = LoopbackTransport(receiver.sync)
        first = sender.sync.sync_peer("receiver", transport)
        self.assertEqual(first.selected, 2)
        self.assertEqual(first.acked, 2)
        self.assertEqual(first.remaining, 3)

        second = sender.sync.sync_peer("receiver", transport)
        self.assertEqual(second.selected, 2)
        self.assertEqual(second.remaining, 1)

        third = sender.sync.sync_peer("receiver", transport)
        self.assertEqual(third.selected, 1)
        self.assertEqual(third.remaining, 0)

    def test_unknown_policy_is_durably_acked_but_left_pending_remote(self):
        event = self.hp.store.create_event(
            "future.set",
            "future:1",
            {"value": 1},
            entity_type="future.unregistered",
        )
        # Unknown policy cannot be locally materialized, so force a direct remote
        # batch to test receiver durability semantics independent of local routing.
        batch = SyncBatch(
            sender_device="hp",
            batch_id="future-batch",
            events=(event,),
        )
        ack = self.dell.sync.receive_batch(batch)

        self.assertEqual(ack.durable_event_ids, (event.id,))
        self.assertEqual(ack.pending_event_ids, (event.id,))
        self.assertEqual(self.dell.applier.get_apply_state(event.id), "pending")

    def test_bad_ack_does_not_mark_delivery_complete(self):
        event = self.hp.author(payload={"value": "x"})

        class BadAckTransport:
            def send_batch(self, peer_id, batch):
                return SyncAck(
                    receiver_device=peer_id,
                    batch_id=batch.batch_id,
                    durable_event_ids=("not-sent",),
                )

        report = self.hp.sync.sync_peer("dell", BadAckTransport())
        self.assertEqual(report.acked, 0)
        self.assertIsNotNone(report.error)
        self.assertEqual(self.hp.sync.delivery_state("dell", event.id)["state"], "pending")


if __name__ == "__main__":
    unittest.main()
