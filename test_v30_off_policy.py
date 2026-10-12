import numpy as np
import pytest

from digital_twin.off_policy import LoggedDecision, OffPolicyEvaluator


def test_on_policy_estimators_equal_sample_mean():
    rows = [
        LoggedDecision(reward=value, behavior_probability=0.5, target_probability=0.5)
        for value in (1, 2, 3, 4)
    ]
    result = OffPolicyEvaluator().evaluate(rows)
    assert result.ips == pytest.approx(2.5)
    assert result.snips == pytest.approx(2.5)
    assert result.effective_sample_size == pytest.approx(4)
    assert result.doubly_robust is None


def test_ips_and_snips_use_exact_unclipped_behavior_propensity():
    rows = [
        LoggedDecision(10, 0.1, 0.5),
        LoggedDecision(0, 0.9, 0.5),
    ]
    result = OffPolicyEvaluator().evaluate(rows)
    weights = np.asarray([5, 5 / 9])
    assert result.ips == pytest.approx(np.mean(weights * [10, 0]))
    assert result.snips == pytest.approx(np.sum(weights * [10, 0]) / weights.sum())
    assert result.maximum_importance_weight == 5


def test_doubly_robust_is_exact_with_correct_q_model():
    rows = [
        LoggedDecision(
            reward=reward, behavior_probability=behavior, target_probability=target,
            q_logged_action=reward, v_target_policy=7.0,
        )
        for reward, behavior, target in ((2, 0.2, 0.8), (9, 0.8, 0.2))
    ]
    result = OffPolicyEvaluator().evaluate(rows)
    assert result.doubly_robust == pytest.approx(7.0)


def test_low_overlap_is_reported_not_silently_clipped():
    rows = [LoggedDecision(1, 1e-6, 1.0)] + [
        LoggedDecision(1, 1.0, 0.0) for _ in range(99)
    ]
    result = OffPolicyEvaluator(minimum_effective_fraction=0.1).evaluate(rows)
    assert result.maximum_importance_weight == 1e6
    assert result.support_warning is True
    assert result.effective_sample_size < 2


def test_cluster_bootstrap_is_deterministic_and_nonempty():
    rows = [
        LoggedDecision(
            reward=float(index), behavior_probability=0.5, target_probability=0.5,
            q_logged_action=float(index), v_target_policy=float(index),
            cluster_id=f"day:{index // 3}",
        )
        for index in range(30)
    ]
    evaluator = OffPolicyEvaluator()
    first = evaluator.cluster_bootstrap(rows, replicates=200, seed=30)
    second = evaluator.cluster_bootstrap(rows, replicates=200, seed=30)
    assert first == second
    assert first[0] < first[1]


@pytest.mark.parametrize(
    "row",
    [
        LoggedDecision(1, 0, 1),
        LoggedDecision(1, -1, 1),
        LoggedDecision(1, 1, 1.1),
        LoggedDecision(float("nan"), 1, 1),
        LoggedDecision(1, 1, 1, q_logged_action=1),
    ],
)
def test_invalid_logged_contract_is_rejected(row):
    with pytest.raises(ValueError):
        OffPolicyEvaluator().evaluate([row])
