"""Pure offline tests for deterministic NEXUS merge policy."""

import unittest

from core.nexus.conflict_resolver import (
    ConflictResolver,
    MergeAction,
    materialize_orset,
    materialize_pn_counter,
)
from core.nexus.event_store import ClockRelation, SyncEvent
from core.nexus.merge_policy import UnknownMergePolicyError


def event(
    *,
    event_id,
    device,
    entity_id="entity:1",
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


class ConflictResolverTests(unittest.TestCase):
    def setUp(self):
        self.resolver = ConflictResolver()

    def test_causal_order_beats_wall_clock_for_lww_register(self):
        first = event(
            event_id="hp-1",
            device="hp",
            vclock={"hp": 1},
            payload={"value": "dark"},
            timestamp="2026-09-30T15:00:00Z",
        )
        successor = event(
            event_id="hp-2",
            device="hp",
            vclock={"hp": 2},
            payload={"value": "light"},
            timestamp="2026-09-30T14:00:00Z",
        )
        local = self.resolver.resolve(None, first).snapshot
        decision = self.resolver.resolve(local, successor)
        self.assertEqual(decision.action, MergeAction.APPLY)
        self.assertEqual(decision.evidence.relation, ClockRelation.AFTER)
        self.assertEqual(decision.snapshot.value, "light")

    def test_concurrent_lww_is_deterministic_in_both_arrival_orders(self):
        hp = event(
            event_id="hp-write",
            device="hp",
            payload={"value": "dark"},
            timestamp="2026-09-30T14:00:00Z",
        )
        dell = event(
            event_id="dell-write",
            device="dell",
            payload={"value": "light"},
            timestamp="2026-09-30T14:00:00Z",
        )

        hp_first = self.resolver.resolve(None, hp).snapshot
        result_a = self.resolver.resolve(hp_first, dell)

        dell_first = self.resolver.resolve(None, dell).snapshot
        result_b = self.resolver.resolve(dell_first, hp)

        self.assertEqual(result_a.evidence.relation, ClockRelation.CONCURRENT)
        self.assertEqual(result_b.evidence.relation, ClockRelation.CONCURRENT)
        self.assertEqual(result_a.snapshot.value, result_b.snapshot.value)
        self.assertEqual(result_a.snapshot.last_event_id, result_b.snapshot.last_event_id)
        self.assertEqual(result_a.snapshot.vclock, {"dell": 1, "hp": 1})

    def test_concurrent_tombstone_wins_for_delete_wins_register(self):
        update = event(
            event_id="hp-update",
            device="hp",
            payload={"value": "keep"},
        )
        deletion = event(
            event_id="dell-delete",
            device="dell",
            event_type="memory.delete",
            payload={},
            tombstone=True,
            timestamp="2026-09-30T13:00:00Z",
        )
        local = self.resolver.resolve(None, update).snapshot
        decision = self.resolver.resolve(local, deletion)
        self.assertTrue(decision.snapshot.tombstone)
        self.assertIsNone(decision.snapshot.value)
        self.assertEqual(decision.snapshot.vclock, {"dell": 1, "hp": 1})

        deleted_first = self.resolver.resolve(None, deletion).snapshot
        reverse = self.resolver.resolve(deleted_first, update)
        self.assertTrue(reverse.snapshot.tombstone)
        self.assertEqual(reverse.snapshot.last_event_id, deletion.id)

    def test_concurrent_orset_adds_converge(self):
        hp = event(
            event_id="hp-python",
            device="hp",
            entity_type="skills.learned",
            event_type="skills.add",
            payload={"element": "python"},
        )
        dell = event(
            event_id="dell-rust",
            device="dell",
            entity_type="skills.learned",
            event_type="skills.add",
            payload={"element": "rust"},
        )

        state_a = self.resolver.resolve(None, hp).snapshot
        state_a = self.resolver.resolve(state_a, dell).snapshot

        state_b = self.resolver.resolve(None, dell).snapshot
        state_b = self.resolver.resolve(state_b, hp).snapshot

        self.assertEqual(materialize_orset(state_a), ("python", "rust"))
        self.assertEqual(materialize_orset(state_b), ("python", "rust"))
        self.assertEqual(state_a.value, state_b.value)

    def test_orset_remove_only_removes_observed_add_tags(self):
        add_python = event(
            event_id="hp-python",
            device="hp",
            entity_type="skills.learned",
            event_type="skills.add",
            payload={"element": "python"},
        )
        add_rust = event(
            event_id="dell-rust",
            device="dell",
            entity_type="skills.learned",
            event_type="skills.add",
            payload={"element": "rust"},
        )
        state = self.resolver.resolve(None, add_python).snapshot
        state = self.resolver.resolve(state, add_rust).snapshot

        remove_python = event(
            event_id="hp-remove-python",
            device="hp",
            entity_type="skills.learned",
            event_type="skills.remove",
            vclock={"hp": 2, "dell": 1},
            payload={"element": "python", "observed_tags": ["hp-python"]},
        )
        state = self.resolver.resolve(state, remove_python).snapshot
        self.assertEqual(materialize_orset(state), ("rust",))

    def test_lww_map_merges_disjoint_concurrent_keys(self):
        theme = event(
            event_id="hp-theme",
            device="hp",
            entity_type="config.ui_settings",
            event_type="config.set",
            payload={"key": "theme", "value": "dark"},
        )
        density = event(
            event_id="dell-density",
            device="dell",
            entity_type="config.ui_settings",
            event_type="config.set",
            payload={"key": "density", "value": "compact"},
        )
        state = self.resolver.resolve(None, theme).snapshot
        state = self.resolver.resolve(state, density).snapshot
        entries = state.value["entries"]
        self.assertEqual(entries["theme"]["value"], "dark")
        self.assertEqual(entries["density"]["value"], "compact")

    def test_pn_counter_components_merge_by_max_and_duplicate_is_skipped(self):
        hp = event(
            event_id="hp-p2",
            device="hp",
            entity_type="usage.stats",
            event_type="usage.p",
            payload={"value": 2},
        )
        dell = event(
            event_id="dell-p3",
            device="dell",
            entity_type="usage.stats",
            event_type="usage.p",
            payload={"value": 3},
        )
        state = self.resolver.resolve(None, hp).snapshot
        state = self.resolver.resolve(state, dell).snapshot
        self.assertEqual(materialize_pn_counter(state), 5)

        duplicate = self.resolver.resolve(state, dell)
        self.assertEqual(duplicate.action, MergeAction.SKIP)
        self.assertEqual(materialize_pn_counter(duplicate.snapshot), 5)

    def test_equal_clock_with_different_event_id_is_rejected(self):
        first = event(
            event_id="event-a",
            device="hp",
            payload={"value": 1},
        )
        local = self.resolver.resolve(None, first).snapshot
        collision = event(
            event_id="event-b",
            device="hp",
            payload={"value": 2},
        )
        decision = self.resolver.resolve(local, collision)
        self.assertEqual(decision.action, MergeAction.REJECT)
        self.assertIn("same vector clock", decision.evidence.reason)

    def test_unknown_entity_type_fails_closed(self):
        unknown = event(
            event_id="hp-unknown",
            device="hp",
            entity_type="future.unregistered",
            payload={"value": 1},
        )
        with self.assertRaises(UnknownMergePolicyError):
            self.resolver.resolve(None, unknown)


if __name__ == "__main__":
    unittest.main()
