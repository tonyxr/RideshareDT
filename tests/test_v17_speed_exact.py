"""WP-S1: the speed changes must not change any number (exactness contracts)."""
import math
import random

import numpy as np

from fast_scalar import sclip
from v17_branch_store import BranchReplayStore


def _same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)


def test_sclip_matches_np_clip_value_type_nan_and_signed_zero():
    rng = random.Random(17)
    xs = [0.3, -1.0, 2.0, float("nan"), -0.0, 0.0, 1e308, -1e308,
          float("inf"), -float("inf")] + [rng.uniform(-200, 200) for _ in range(20000)]
    bounds = [(0.0, 1.0), (0, 1), (-60.0, 60.0), (5.0, 120.0), (0.25, 8.0), (1, 4.0)]
    for x in xs:
        for lo, hi in bounds:
            for value in (x, np.float64(x)):
                assert _same(sclip(value, lo, hi), np.clip(value, lo, hi)), (value, lo, hi)


def test_sclip_falls_back_to_np_clip_for_arrays_ints_and_other_dtypes():
    arr = np.array([-1.0, 0.5, 3.0])
    assert _same(sclip(arr, 0.0, 1.0), np.clip(arr, 0.0, 1.0))
    assert _same(sclip(np.float32(2.5), 0.0, 1.0), np.clip(np.float32(2.5), 0.0, 1.0))
    assert sclip(5, 0, 3) == np.clip(5, 0, 3)
    assert type(sclip(5, 0, 3)) is type(np.clip(5, 0, 3))


def _row(i):
    return {"observation_window": [float(i + k) for k in range(20)],
            "action_credits": [float(i)] * 4, "regime_label": "calm" if i % 3 else "shortage"}


def test_branch_store_copy_rows_false_draws_the_same_rows():
    a = BranchReplayStore(50, seed=5)
    b = BranchReplayStore(50, seed=5)
    for i in range(80):
        a.add(_row(i))
        b.add(_row(i))
    for _ in range(25):
        assert a.sample(16, regime_balanced=True) == b.sample(16, regime_balanced=True, copy_rows=False)
    assert a.state_dict()["rng_state"] == b.state_dict()["rng_state"]
