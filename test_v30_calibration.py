import numpy as np
import pytest

from digital_twin.calibration import RecursiveLinearCalibrator, ResidualMonitor


def test_recursive_calibrator_recovers_known_elasticity():
    rng = np.random.default_rng(30)
    model = RecursiveLinearCalibrator(["log_price", "wait"], forgetting_factor=1.0)
    for _ in range(500):
        price = rng.normal()
        wait = rng.normal()
        target = 2.0 - 1.5 * price - 0.25 * wait + rng.normal(scale=0.03)
        model.update({"log_price": price, "wait": wait}, target, expected_noise_std=0.03)
    params = model.parameter_dict()
    assert params["intercept"] == pytest.approx(2.0, abs=0.02)
    assert params["log_price"] == pytest.approx(-1.5, abs=0.02)
    assert params["wait"] == pytest.approx(-0.25, abs=0.02)


def test_prediction_uncertainty_shrinks_with_identifying_data():
    model = RecursiveLinearCalibrator(["x"], forgetting_factor=1.0)
    before = model.predict({"x": 1})[1]
    for value in np.linspace(-2, 2, 40):
        model.update({"x": float(value)}, 3 + 2 * value)
    after = model.predict({"x": 1})[1]
    assert after < before / 100


def test_cusum_detects_persistent_shift_not_centered_noise():
    rng = np.random.default_rng(31)
    monitor = ResidualMonitor(window=64, cusum_threshold=10)
    for value in rng.normal(scale=1, size=100):
        report = monitor.update(value, 0, expected_noise_std=1)
    assert report.drift_detected is False
    for value in np.full(20, 2.0):
        report = monitor.update(value, 0, expected_noise_std=1)
    assert report.drift_detected is True
    assert report.cusum_positive >= 10


def test_residual_report_has_bias_mae_rmse_and_scale():
    monitor = ResidualMonitor(window=8)
    for residual in (1, -1, 2, -2):
        report = monitor.update(residual, 0, expected_noise_std=1)
    assert report.bias == pytest.approx(0)
    assert report.mae == pytest.approx(1.5)
    assert report.rmse == pytest.approx(np.sqrt(2.5))
    assert report.residual_std > 0


def test_calibration_snapshot_restore_is_exact():
    model = RecursiveLinearCalibrator(["x"])
    for x in range(10):
        model.update({"x": x}, 1 + 3 * x)
    restored = RecursiveLinearCalibrator.restore(model.snapshot())
    assert restored.snapshot() == model.snapshot()
    assert restored.predict({"x": 4}) == model.predict({"x": 4})


def test_feature_contract_and_invalid_values_fail():
    model = RecursiveLinearCalibrator(["x"])
    with pytest.raises(ValueError, match="mismatch"):
        model.update({}, 1)
    with pytest.raises(ValueError):
        model.update({"x": float("nan")}, 1)
    with pytest.raises(ValueError):
        model.update({"x": 1}, float("inf"))
