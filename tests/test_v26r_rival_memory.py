"""Rival with memory: lagged retaliation that outlives the undercut, and accommodation of a sustained premium."""
import numpy as np
from pricing_models import FirmMemoryBestResponsePricer, FirmAggressiveAdaptiveBestResponsePricer, FirmMetrics

ANCHORS = dict(city_base=2.5, city_pmin=0.35, city_pmile=1.5, city_booking=2.0, city_airport=5.0)


def _level(pricer):
    return float(pricer.overrides.base_fare) / ANCHORS["city_base"]


def _run(pricer, gaps, start=True):
    """Alternate act/update with the given Firm2-minus-Firm1 gaps; return the base-fare level path."""
    levels = []
    if start:
        pricer.act(hour=12, weather="clear", **ANCHORS)
    for gap in gaps:
        pricer.update(FirmMetrics(), price_gap_mean=float(gap))
        pricer.act(hour=12, weather="clear", **ANCHORS)
        levels.append(_level(pricer))
    return levels


def test_undercut_triggers_lagged_retaliation_that_persists():
    pricer = FirmMemoryBestResponsePricer(seed=0, lag_steps=6, hold_mean_steps=30)
    before = _run(pricer, [0.0] * 6)
    assert pricer.retaliation_left == 0 and pricer.memory_state == "calm"
    during = _run(pricer, [3.0] * 5, start=False)           # five steps: below the lag, no retaliation yet
    assert pricer.retaliation_episodes == 0
    _run(pricer, [3.0] * 3, start=False)                     # lag reached
    assert pricer.retaliation_episodes == 1 and pricer.memory_state == "retaliation"
    matched = _run(pricer, [3.0] * 10, start=False)
    assert matched[-1] < before[-1] - 1e-9                   # stepped down to match
    hold = pricer.retaliation_left
    after = _run(pricer, [0.0] * 3, start=False)             # the undercut ends; the hold continues
    assert pricer.retaliation_left == max(0, hold - 3) and pricer.memory_state == "retaliation"
    assert after[-1] <= matched[-1] + 1e-9


def test_sustained_premium_is_accommodated_and_calm_is_the_base_rule():
    pricer = FirmMemoryBestResponsePricer(seed=1, accommodation_steps=12)
    base = _run(pricer, [0.0] * 3)
    _run(pricer, [-3.0] * 11, start=False)
    moves_before = pricer.accommodation_moves
    path = _run(pricer, [-3.0] * 6, start=False)
    assert pricer.accommodation_moves > moves_before and pricer.memory_state == "accommodation"
    assert path[-1] > base[-1] + 1e-9
    assert isinstance(pricer, FirmAggressiveAdaptiveBestResponsePricer.__mro__[1])   # shares the base rule
    assert pricer.alpha_share < FirmAggressiveAdaptiveBestResponsePricer(seed=0).alpha_share


def test_geometric_hold_is_seeded():
    a = FirmMemoryBestResponsePricer(seed=7); b = FirmMemoryBestResponsePricer(seed=7)
    for pr in (a, b):
        _run(pr, [3.0] * 8)
    assert a.retaliation_left == b.retaliation_left > 0
