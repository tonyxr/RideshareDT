import numpy as np
import pytest

from digital_twin.observation import ClosedLoopFeatures, V30ObservationContract


def closed_loop(**changes):
    values = dict(
        demand_now=100, demand_forecast=120, supply_now=80, supply_forecast=72,
        state_uncertainty=0.25, data_age_seconds=60, decision_interval_seconds=3600,
        episode_steps_remaining=900, episode_horizon_steps=1200,
    )
    values.update(changes)
    return ClosedLoopFeatures(**values)


def test_production_and_simulator_vectors_are_byte_equal():
    contract = V30ObservationContract(shock_features=True)
    base = np.linspace(-1.0, 1.0, len(contract.base_names), dtype=np.float32)
    production = dict(zip(contract.base_names, base.tolist()))
    simulated = contract.from_simulator(base, closed_loop())
    served = contract.from_production(production, closed_loop())
    assert simulated.tobytes() == served.tobytes()


def test_v29_compact_mdp_is_preserved_with_loop_features_and_episode_clock():
    contract = V30ObservationContract(shock_features=True)
    assert contract.base_names[0:4] == ("hour_sin", "hour_cos", "day_sin", "day_cos")
    assert len(contract.names) == len(contract.base_names) + 6
    assert contract.manifest()["base_contract"] == "v29_compact_foresight_preserved"


def test_forecast_ratios_are_monotone_and_unaliased():
    contract = V30ObservationContract(shock_features=False)
    base = np.zeros(len(contract.base_names), dtype=np.float32)
    low = contract.from_simulator(base, closed_loop(demand_forecast=100))[-6]
    high = contract.from_simulator(base, closed_loop(demand_forecast=10000))[-6]
    extreme = contract.from_simulator(base, closed_loop(demand_forecast=1e9))[-6]
    assert low < high < extreme


def test_schema_hash_changes_with_shock_layout_and_is_stable():
    first = V30ObservationContract(shock_features=True)
    second = V30ObservationContract(shock_features=True)
    without_shocks = V30ObservationContract(shock_features=False)
    assert first.schema_sha256 == second.schema_sha256
    assert first.schema_sha256 != without_shocks.schema_sha256


def test_missing_extra_and_wrong_width_fail_closed():
    contract = V30ObservationContract(shock_features=False)
    base = {name: 0.0 for name in contract.base_names}
    base.pop(contract.base_names[-1])
    with pytest.raises(ValueError, match="schema mismatch"):
        contract.from_production(base, closed_loop())
    with pytest.raises(ValueError, match="width mismatch"):
        contract.from_simulator(np.zeros(len(contract.base_names) + 1), closed_loop())


@pytest.mark.parametrize(
    "changes",
    [
        {"demand_now": -1}, {"state_uncertainty": -1},
        {"data_age_seconds": -1}, {"decision_interval_seconds": 0},
        {"supply_forecast": float("nan")},
    ],
)
def test_invalid_closed_loop_measurements_are_rejected(changes):
    with pytest.raises(ValueError):
        closed_loop(**changes).encoded()
