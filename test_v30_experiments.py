from collections import Counter

import pytest

from digital_twin.experiments import ExperimentArm, StableExperimentAssigner


def _assigner():
    return StableExperimentAssigner(
        experiment_id="chicago-price-2026q4", salt="test-secret",
        arms=(ExperimentArm("control", 0.7), ExperimentArm("candidate", 0.3)),
    )


def test_assignment_is_stable_stratified_and_has_exact_propensity():
    assigner = _assigner()
    first = assigner.assign("rider_hash_123", stratum="airport")
    assert assigner.assign("rider_hash_123", stratum="airport") == first
    assert first.assignment_probability in (0.7, 0.3)
    assert first.assignment_digest
    assert assigner.assign("rider_hash_123", stratum="commute").assignment_digest != first.assignment_digest


def test_assignment_distribution_matches_preregistered_weights():
    assigner = _assigner()
    counts = Counter(assigner.assign(f"rider_hash_{index}").arm for index in range(20_000))
    assert counts["candidate"] / 20_000 == pytest.approx(0.3, abs=0.015)


@pytest.mark.parametrize("unit_id", ["", "person@example.com", "+1-415-555-0100"])
def test_direct_pii_or_empty_unit_is_rejected(unit_id):
    with pytest.raises(ValueError):
        _assigner().assign(unit_id)


def test_invalid_arm_probabilities_are_rejected():
    with pytest.raises(ValueError):
        StableExperimentAssigner(
            experiment_id="e", salt="s",
            arms=(ExperimentArm("a", 0.8), ExperimentArm("b", 0.3)),
        )
