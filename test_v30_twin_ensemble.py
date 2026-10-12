import numpy as np
import pytest

from digital_twin.twin_ensemble import TwinMember, VersionedTwinEnsemble


def members():
    return [
        TwinMember("low-elasticity", {"elasticity": -1.0}, "market-v30"),
        TwinMember("middle", {"elasticity": -2.0}, "market-v30"),
        TwinMember("high-elasticity", {"elasticity": -4.0}, "market-v30"),
    ]


def predictions():
    return {
        "low-elasticity": {"demand": 90, "profit": 10},
        "middle": {"demand": 70, "profit": 8},
        "high-elasticity": {"demand": 30, "profit": 2},
    }


def test_ensemble_reports_mean_and_structural_uncertainty():
    ensemble = VersionedTwinEnsemble(members())
    result = ensemble.predict(predictions())
    assert result.mean["demand"] == pytest.approx((90 + 70 + 30) / 3)
    assert result.variance["demand"] > 0
    assert sum(result.weights.values()) == pytest.approx(1)


def test_likelihood_update_moves_weight_toward_supported_twin():
    ensemble = VersionedTwinEnsemble(members())
    for _ in range(3):
        result = ensemble.update_weights(
            {"demand": 69, "profit": 8.2}, predictions(),
            observation_variance={"demand": 4, "profit": 1},
        )
    assert result.weights["middle"] > 0.99


def test_member_sampling_matches_posterior():
    ensemble = VersionedTwinEnsemble(members(), weights=[0, 1, 0])
    rng = np.random.default_rng(30)
    assert {ensemble.sample_member(rng).member_id for _ in range(20)} == {"middle"}


def test_snapshot_restore_and_fingerprint_are_exact():
    ensemble = VersionedTwinEnsemble(members())
    ensemble.update_weights(
        {"demand": 70, "profit": 8}, predictions(),
        observation_variance={"demand": 4, "profit": 1},
    )
    restored = VersionedTwinEnsemble.restore(ensemble.snapshot())
    assert restored.snapshot() == ensemble.snapshot()
    assert restored.predict(predictions()) == ensemble.predict(predictions())


def test_incomplete_or_inconsistent_predictions_fail():
    ensemble = VersionedTwinEnsemble(members())
    bad = predictions()
    bad.pop("middle")
    with pytest.raises(ValueError, match="every ensemble member"):
        ensemble.predict(bad)
    bad = predictions()
    bad["middle"] = {"demand": 70}
    with pytest.raises(ValueError, match="same outputs"):
        ensemble.predict(bad)


def test_invalid_weights_and_member_parameters_fail():
    with pytest.raises(ValueError):
        VersionedTwinEnsemble(members(), weights=[0, 0, 0])
    with pytest.raises(ValueError):
        VersionedTwinEnsemble([TwinMember("bad", {"x": float("nan")}, "v")])
