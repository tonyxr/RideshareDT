from datetime import datetime, timezone

import numpy as np
import pytest

from digital_twin.ledger import EventLedger
from digital_twin.observation import ClosedLoopFeatures, V30ObservationContract
from digital_twin.policy_service import ShadowPolicy, ShadowPolicyService


def _base(contract):
    return {name: float(index) / 100 for index, name in enumerate(contract.base_names)}


def _closed_loop():
    return ClosedLoopFeatures(100, 110, 80, 76, 0.2, 300, 86400, 900, 1200)


def test_shadow_policy_is_deterministic_audited_and_non_deploying(tmp_path):
    contract = V30ObservationContract(shock_features=True)
    ledger = EventLedger(tmp_path / "events.sqlite")
    policy = ShadowPolicy(
        policy_id="v30-candidate-7",
        action_names=("hold", "base_up", "per_mile_down"),
        probability_policy=lambda observation: np.asarray([0.1, 0.8, 0.1]),
        observation_schema_sha256=contract.schema_sha256,
    )

    def transition(action, before):
        result = dict(before)
        if action == "base_up":
            result["base"] += 0.1
        return result

    service = ShadowPolicyService(
        policy=policy, observation_contract=contract, ledger=ledger,
        tariff_transition=transition,
    )
    kwargs = dict(
        event_id="decision:1", request_id="request:1",
        event_time=datetime(2026, 1, 1, tzinfo=timezone.utc), market="chicago",
        base_observation=_base(contract), closed_loop=_closed_loop(),
        tariff_before={"base": 1.0, "per_mile": 1.0},
    )
    first = service.recommend(**kwargs)
    second = service.recommend(**kwargs)
    assert first.action_name == second.action_name
    assert first.tariff_after["base"] == pytest.approx(
        1.1 if first.action_name == "base_up" else 1.0
    )
    assert first.ledger_inserted is True and second.ledger_inserted is False
    row = ledger.events(event_type="decision")[0]
    assert row["context"]["deployment_authorized"] is False
    assert row["action"]["shadow_only"] is True
    assert row["observation_schema"] == contract.schema_sha256
    assert row["context"]["policy_mode"] == "sample_raw"
    counts = {name: 0 for name in policy.action_names}
    for index in range(500):
        result = service.recommend(**{
            **kwargs, "event_id": f"sample:{index}", "request_id": f"sample:{index}",
        })
        counts[result.action_name] += 1
        assert result.action_probabilities == first.action_probabilities
    assert 350 < counts["base_up"] < 450
    assert counts["hold"] > 20 and counts["per_mile_down"] > 20


def test_shadow_service_rejects_schema_and_probability_mismatch(tmp_path):
    contract = V30ObservationContract(shock_features=False)
    ledger = EventLedger(tmp_path / "events.sqlite")
    with pytest.raises(ValueError):
        ShadowPolicyService(
            policy=ShadowPolicy("p", ("hold",), lambda _: [1], "wrong"),
            observation_contract=contract, ledger=ledger,
            tariff_transition=lambda _, tariff: tariff,
        )
    service = ShadowPolicyService(
        policy=ShadowPolicy("p", ("a", "b"), lambda _: [0, 0], contract.schema_sha256),
        observation_contract=contract, ledger=ledger,
        tariff_transition=lambda _, tariff: tariff,
    )
    with pytest.raises(ValueError):
        service.recommend(
            event_id="d", request_id="r", event_time="2026-01-01T00:00:00Z",
            market="x", base_observation=_base(contract), closed_loop=_closed_loop(),
            tariff_before={"base": 1.0},
        )
