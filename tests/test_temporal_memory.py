from datetime import datetime, timezone

import pytest

from core.temporal_memory import TemporalMemoryError, TemporalMemoryStore


def test_claims_preserve_provenance_and_do_not_auto_verify(tmp_path):
    store = TemporalMemoryStore(tmp_path / "memory.db")
    claim = store.add_claim(
        subject="project:jarvis",
        predicate="status",
        value="alpha",
        source="owner:chat",
        observed_at="2026-10-07T10:00:00+00:00",
        confidence=0.8,
        metadata={"kind": "owner_statement"},
    )

    assert claim.value == "alpha"
    assert claim.source == "owner:chat"
    assert claim.verified is False
    assert claim.active is True
    assert claim.metadata["kind"] == "owner_statement"


def test_supersede_closes_old_validity_without_erasing_history(tmp_path):
    store = TemporalMemoryStore(tmp_path / "memory.db")
    old = store.add_claim(
        subject="device:laptop",
        predicate="state",
        value="offline",
        source="sensor:local",
        observed_at="2026-10-07T10:00:00+00:00",
        verified=True,
    )
    new = store.supersede(
        old.claim_id,
        value="online",
        source="sensor:local",
        observed_at="2026-10-07T11:00:00+00:00",
        confidence=0.95,
        verified=True,
    )

    old_after = store.get(old.claim_id)
    assert old_after.active is False
    assert old_after.valid_to == "2026-10-07T11:00:00.000000+00:00"
    assert old_after.superseded_by == new.claim_id
    assert new.active is True
    assert store.active_claims(subject="device:laptop")[0].value == "online"
    assert [item.value for item in store.history(subject="device:laptop")] == ["online", "offline"]


def test_claims_at_reconstructs_past_state(tmp_path):
    store = TemporalMemoryStore(tmp_path / "memory.db")
    first = store.add_claim(
        subject="car",
        predicate="mode",
        value="parked",
        source="owner:input",
        observed_at="2026-10-07T09:00:00Z",
    )
    store.supersede(
        first.claim_id,
        value="driving",
        source="owner:input",
        observed_at="2026-10-07T10:00:00Z",
    )

    past = store.claims_at("2026-10-07T09:30:00Z", subject="car", predicate="mode")
    now = store.claims_at("2026-10-07T10:30:00Z", subject="car", predicate="mode")
    assert [item.value for item in past] == ["parked"]
    assert [item.value for item in now] == ["driving"]


def test_search_can_require_verified_active_claims(tmp_path):
    store = TemporalMemoryStore(tmp_path / "memory.db")
    store.add_claim(
        subject="jarvis",
        predicate="capability",
        value="structured UI observation",
        source="test:verified",
        verified=True,
        confidence=0.9,
    )
    store.add_claim(
        subject="jarvis",
        predicate="capability",
        value="unverified UI rumor",
        source="test:unverified",
        verified=False,
        confidence=1.0,
    )

    result = store.search("UI", verified_only=True)
    assert len(result) == 1
    assert result[0].value == "structured UI observation"


def test_double_supersede_fails_closed(tmp_path):
    store = TemporalMemoryStore(tmp_path / "memory.db")
    first = store.add_claim(
        subject="x",
        predicate="p",
        value="one",
        source="test:source",
    )
    store.supersede(first.claim_id, value="two", source="test:source")
    with pytest.raises(TemporalMemoryError):
        store.supersede(first.claim_id, value="three", source="test:source")
