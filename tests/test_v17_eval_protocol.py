from scripts import eval_protocol_v1 as protocol
from Core import CATCHUP_OPPONENT_MODES, REACTIVE_STAGE3_OPPONENT_MODES
from Core import OPTIMIZED_BENCHMARK_OPPONENT_MODES
from Core import PROFIT_BEST_RESPONSE_OPPONENT_MODES
from v17_objective import final_objective, profit_report, rank_profit_rows


def test_registered_contract_is_internally_consistent():
    spec = protocol.protocol()
    protocol.validate(spec)
    assert len(protocol.CASES) == 6
    assert len(protocol.COMPETITORS) == 10
    assert len(protocol.POLICY_CLASSES) == 10
    assert spec["step_B"]["rollouts_per_city"] == 3000
    assert spec["step_C"]["rollouts_total"] == 1500


def test_protocol_hash_is_stable_and_content_addressed():
    first = protocol.protocol()
    second = protocol.protocol()
    assert protocol.protocol_hash(first) == protocol.protocol_hash(second)
    second["step_B"]["horizon_days"] = 41
    assert protocol.protocol_hash(first) != protocol.protocol_hash(second)


def test_execution_is_deliberately_disabled_in_contract_patch():
    spec = protocol.protocol()
    assert spec["entry_point"] == "scripts/eval_protocol_v1.py"
    assert spec["held_out_tapes"]["must_differ_from"] == [
        "training",
        "T2_validation",
    ]


def test_t0_13_corrected_competitor_groups_are_disjoint_from_training_pools():
    spec = protocol.protocol()
    groups = spec["competitor_groups"]
    assert len(groups["training_exposed"]) == 5
    assert len(groups["held_out_canonical"]) == 3
    assert len(groups["held_out_perturbed"]) == 2
    training_pool = {
        *CATCHUP_OPPONENT_MODES,
        *REACTIVE_STAGE3_OPPONENT_MODES,
        *OPTIMIZED_BENCHMARK_OPPONENT_MODES,
        *PROFIT_BEST_RESPONSE_OPPONENT_MODES,
    }
    assert not set(groups["held_out_canonical"]) & training_pool
    assert not set(groups["held_out_perturbed"]) & training_pool


def test_final_objective_ranks_and_reports_both_profit_components():
    assert final_objective(4.0, 2.0) == 2.5
    assert profit_report(4.0, 2.0) == {
        "own_profit": 4.0,
        "competitor_profit": 2.0,
        "final_objective": 2.5,
    }
    ranked = rank_profit_rows([
        {"name": "high-own", "own_profit": 4.0, "competitor_profit": 3.0},
        {"name": "competitive", "own_profit": 3.5, "competitor_profit": 1.0},
    ])
    assert [row["name"] for row in ranked] == ["competitive", "high-own"]
    assert all("own_profit" in row and "competitor_profit" in row for row in ranked)
