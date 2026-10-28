import numpy as np
import torch

from platform_mdp import (
    MOMENTUM_FEATURE_DIM,
    OPPONENT_FEATURE_DIM,
    SEGMENTS,
    ObservationConfig,
    PlatformObservationModel,
)
from PPOAgent import PPOAgent


COEFS = {
    "base_fare": 3.0,
    "per_minute": 0.3,
    "per_mile": 1.2,
    "booking_fee": 2.0,
    "airport_fee": 0.0,
}


def _model(**kw):
    cfg = ObservationConfig(
        learning_corrections=True,
        regime_features=True,
        persistence_features=True,
        telemetry_delay_steps=0,
        market_share_noise=0.0,
        **kw,
    )
    return PlatformObservationModel(1, cfg)


def _obs(m):
    return m.build_observation(
        hour=9,
        day_of_week=1,
        weather="clear",
        own_coefficients=COEFS,
        anchor_coefficients=COEFS,
        regime_features={},
        persistence_features={},
    )


def _ingest(m, fulfil, n):
    for _ in range(n):
        m.ingest(
            own_metrics={
                "fulfillment_rate": fulfil,
                "chosen_share": 0.3,
                "completed_share": 0.3,
                "profit_per_request": 2.0,
                "wait_minutes": 5.0,
            },
            supply_metrics={"utilization": 0.6},
            crowd_stats={},
        )


def test_flags_off_keep_revision12_width():
    assert _model().observation_dim == 34
    assert _obs(_model()).size == 34


def test_widths_with_flags_on():
    m = _model(momentum_features=True, opponent_features=True)
    assert (
        m.observation_dim
        == 34 + MOMENTUM_FEATURE_DIM + OPPONENT_FEATURE_DIM
        == 60
    )
    assert _obs(m).size == 60
    groups = m.regime_feature_groups()
    assert tuple(range(49, 60)) == tuple(
        groups["opponent"][-OPPONENT_FEATURE_DIM:]
    )


def test_momentum_step_change():
    m = _model(momentum_features=True, momentum_day_steps=6)
    _ingest(m, 0.95, 42)
    v = _obs(m)
    assert np.allclose(v[34:48], 0.0, atol=1e-6) and v[48] == 1.0
    _ingest(m, 0.80, 6)
    v = _obs(m)
    assert np.isclose(v[34], np.arcsinh(-3.0), atol=1e-5)
    assert np.isclose(v[41], np.arcsinh(-3.0), atol=1e-5)


def test_opponent_move_detection():
    m = _model(opponent_features=True, momentum_day_steps=6)
    for _ in range(12):
        m.record_rival_index(0.10, 0.20)
    m.record_rival_index(0.00, 0.20)
    assert m._rival_last_move_step is None
    m.record_rival_index(0.00, 0.20)
    assert np.isclose(m._rival_last_move, -0.10)
    for _ in range(4):
        m.record_rival_index(0.00, 0.20)
    s = _obs(m)[-OPPONENT_FEATURE_DIM:]
    assert np.isclose(s[0], 0.0)
    assert np.isclose(s[1], np.arcsinh(-0.10 / 0.03), atol=1e-5)
    assert np.isclose(s[7], np.arcsinh(-0.10 / 0.03), atol=1e-5)
    assert np.isclose(s[8], 4 / 42.0)
    assert np.isclose(s[9], 1 / 7.0)
    assert np.isclose(s[10], np.arcsinh(2.0), atol=1e-5)


def test_single_outlier_is_not_a_move():
    m = _model(opponent_features=True)
    for value in [0.10] * 10 + [0.16] + [0.10] * 5:
        m.record_rival_index(value, 0.10)
    assert m._rival_last_move_step is None


def test_snapshot_round_trip():
    m = _model(momentum_features=True, opponent_features=True)
    _ingest(m, 0.9, 20)
    for value in (0.1, 0.1, 0.0, 0.0):
        m.record_rival_index(value, 0.1)
    snap, before = m.snapshot(), _obs(m)
    _ingest(m, 0.5, 10)
    m.record_rival_index(0.3, 0.1)
    m.record_rival_index(0.3, 0.1)
    m.restore(snap)
    assert np.allclose(_obs(m), before)


def test_hold_baseline_rms_advantage():
    v = torch.tensor([[0.0, 0.4, float("nan")], [0.0, -0.2, 0.1]])
    adv, valid = PPOAgent._counterfactual_actor_advantages(
        v,
        torch.isfinite(v),
        torch.tensor([1, 2]),
        baseline="provided",
        scale="rms",
    )
    rms = float(np.sqrt((0.4 ** 2 + 0.1 ** 2) / 2))
    assert valid.tolist() == [True, True]
    assert torch.allclose(
        adv, torch.tensor([0.4 / rms, 0.1 / rms]), atol=1e-5
    )


def test_default_mode_matches_revision12():
    v = torch.tensor([[0.3, 0.5, 0.1], [0.2, -0.1, 0.0]])
    adv, _ = PPOAgent._counterfactual_actor_advantages(
        v, torch.isfinite(v), torch.tensor([1, 0])
    )
    raw = torch.tensor([0.5 - 0.3, 0.2 - 0.1 / 3])
    expected = (raw - raw.mean()) / (raw.std(unbiased=False) + 1e-8)
    assert torch.allclose(adv, expected, atol=1e-5)
