"""Online calibration and residual/drift reporting for v30 twins."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from typing import Mapping, Optional, Sequence

import numpy as np

from .contracts import CALIBRATION_SCHEMA_VERSION


@dataclass(frozen=True)
class ResidualReport:
    count: int
    bias: float
    mae: float
    rmse: float
    residual_std: float
    cusum_positive: float
    cusum_negative: float
    drift_detected: bool
    schema_version: str = CALIBRATION_SCHEMA_VERSION


class ResidualMonitor:
    """Fixed-window residual diagnostics with two-sided standardized CUSUM."""

    def __init__(self, window: int = 128, *, cusum_slack: float = 0.5, cusum_threshold: float = 8.0):
        if window < 8:
            raise ValueError("residual window must be at least 8")
        if cusum_slack < 0 or cusum_threshold <= 0:
            raise ValueError("CUSUM parameters are invalid")
        self.window = int(window)
        self.cusum_slack = float(cusum_slack)
        self.cusum_threshold = float(cusum_threshold)
        self.residuals: deque[float] = deque(maxlen=self.window)
        self.reference_scale: Optional[float] = None
        self.cusum_positive = 0.0
        self.cusum_negative = 0.0

    def update(self, observed: float, predicted: float, *, expected_noise_std: Optional[float] = None) -> ResidualReport:
        residual = float(observed) - float(predicted)
        if not math.isfinite(residual):
            raise ValueError("residual inputs must be finite")
        self.residuals.append(residual)
        empirical = float(np.std(self.residuals, ddof=1)) if len(self.residuals) > 1 else 0.0
        supplied = 0.0 if expected_noise_std is None else float(expected_noise_std)
        if supplied < 0 or not math.isfinite(supplied):
            raise ValueError("expected_noise_std must be finite and non-negative")
        if self.reference_scale is None:
            self.reference_scale = max(supplied, empirical, 1e-9)
        scale = max(self.reference_scale, supplied, 1e-9)
        standardized = residual / scale
        self.cusum_positive = max(0.0, self.cusum_positive + standardized - self.cusum_slack)
        self.cusum_negative = min(0.0, self.cusum_negative + standardized + self.cusum_slack)
        return self.report()

    def report(self) -> ResidualReport:
        values = np.asarray(self.residuals, dtype=float)
        if values.size == 0:
            return ResidualReport(0, math.nan, math.nan, math.nan, math.nan, 0.0, 0.0, False)
        return ResidualReport(
            count=int(values.size), bias=float(values.mean()),
            mae=float(np.abs(values).mean()), rmse=float(np.sqrt(np.mean(values * values))),
            residual_std=float(values.std(ddof=1)) if values.size > 1 else 0.0,
            cusum_positive=self.cusum_positive, cusum_negative=self.cusum_negative,
            drift_detected=(
                self.cusum_positive >= self.cusum_threshold
                or self.cusum_negative <= -self.cusum_threshold
            ),
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "window": self.window, "cusum_slack": self.cusum_slack,
            "cusum_threshold": self.cusum_threshold,
            "residuals": list(self.residuals), "reference_scale": self.reference_scale,
            "cusum_positive": self.cusum_positive, "cusum_negative": self.cusum_negative,
        }

    @classmethod
    def restore(cls, snapshot: Mapping[str, object]) -> "ResidualMonitor":
        if snapshot.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise ValueError("residual-monitor schema mismatch")
        monitor = cls(
            int(snapshot["window"]), cusum_slack=float(snapshot["cusum_slack"]),
            cusum_threshold=float(snapshot["cusum_threshold"]),
        )
        monitor.residuals.extend(float(value) for value in snapshot["residuals"])
        value = snapshot.get("reference_scale")
        monitor.reference_scale = None if value is None else float(value)
        monitor.cusum_positive = float(snapshot["cusum_positive"])
        monitor.cusum_negative = float(snapshot["cusum_negative"])
        return monitor


class RecursiveLinearCalibrator:
    """Recursive least squares for continuously updated structural parameters."""

    def __init__(
        self,
        feature_names: Sequence[str],
        *,
        forgetting_factor: float = 0.995,
        initial_variance: float = 1_000.0,
    ):
        if not feature_names or len(set(feature_names)) != len(feature_names):
            raise ValueError("feature_names must be non-empty and unique")
        if not 0.9 <= forgetting_factor <= 1.0:
            raise ValueError("forgetting_factor must be in [0.9, 1]")
        if initial_variance <= 0:
            raise ValueError("initial_variance must be positive")
        self.feature_names = tuple(str(value) for value in feature_names)
        self.forgetting_factor = float(forgetting_factor)
        width = len(self.feature_names) + 1
        self.coefficients = np.zeros(width, dtype=float)
        self.covariance = np.eye(width, dtype=float) * float(initial_variance)
        self.updates = 0
        self.monitor = ResidualMonitor()

    def _vector(self, features: Mapping[str, float]) -> np.ndarray:
        missing = set(self.feature_names) - set(features)
        extra = set(features) - set(self.feature_names)
        if missing or extra:
            raise ValueError(f"calibration feature mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
        vector = np.asarray([1.0] + [float(features[name]) for name in self.feature_names])
        if not np.all(np.isfinite(vector)):
            raise ValueError("calibration features must be finite")
        return vector

    def predict(self, features: Mapping[str, float]) -> tuple[float, float]:
        vector = self._vector(features)
        prediction = float(vector @ self.coefficients)
        variance = float(max(vector @ self.covariance @ vector, 0.0))
        return prediction, variance

    def update(
        self,
        features: Mapping[str, float],
        target: float,
        *,
        expected_noise_std: Optional[float] = None,
    ) -> dict[str, object]:
        target_value = float(target)
        if not math.isfinite(target_value):
            raise ValueError("calibration target must be finite")
        vector = self._vector(features)
        prediction = float(vector @ self.coefficients)
        covariance_vector = self.covariance @ vector
        denominator = self.forgetting_factor + float(vector @ covariance_vector)
        gain = covariance_vector / denominator
        residual = target_value - prediction
        self.coefficients = self.coefficients + gain * residual
        self.covariance = (
            self.covariance - np.outer(gain, vector) @ self.covariance
        ) / self.forgetting_factor
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.updates += 1
        report = self.monitor.update(target_value, prediction, expected_noise_std=expected_noise_std)
        return {
            "prediction_before_update": prediction,
            "residual": residual,
            "coefficients": self.parameter_dict(),
            "residual_report": asdict(report),
        }

    def parameter_dict(self) -> dict[str, float]:
        return {
            "intercept": float(self.coefficients[0]),
            **{
                name: float(self.coefficients[index + 1])
                for index, name in enumerate(self.feature_names)
            },
        }

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "feature_names": list(self.feature_names),
            "forgetting_factor": self.forgetting_factor,
            "coefficients": self.coefficients.tolist(),
            "covariance": self.covariance.tolist(),
            "updates": self.updates,
            "monitor": self.monitor.snapshot(),
        }

    @classmethod
    def restore(cls, snapshot: Mapping[str, object]) -> "RecursiveLinearCalibrator":
        if snapshot.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise ValueError("calibrator schema mismatch")
        model = cls(
            snapshot["feature_names"], forgetting_factor=float(snapshot["forgetting_factor"])
        )
        model.coefficients = np.asarray(snapshot["coefficients"], dtype=float)
        model.covariance = np.asarray(snapshot["covariance"], dtype=float)
        model.updates = int(snapshot["updates"])
        model.monitor = ResidualMonitor.restore(snapshot["monitor"])
        return model
