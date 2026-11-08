"""Exact scalar fast paths for numpy calls made one rider at a time.

``sclip(x, lo, hi)`` returns exactly what ``np.clip(x, lo, hi)`` returns
(same value, same NaN and signed-zero behaviour, same numpy scalar type) for
Python-float and numpy-float64 inputs, without numpy's per-call dispatch cost
(about 2 microseconds). Any other input (arrays, ints, other dtypes) is
passed to ``np.clip`` unchanged.
"""
import numpy as _np

_F64 = _np.float64


def sclip(x, lo, hi):
    tx = type(x)
    if (tx is float or tx is _F64) and type(lo) in (float, int) and type(hi) in (float, int):
        # numpy.clip == minimum(maximum(x, lo), hi).  On this Mac/NumPy build,
        # the bound wins an equality tie, which matters for the sign bit of
        # zero.  Explicit comparisons preserve that behavior while retaining
        # NaN propagation (both comparisons are false for NaN).
        v = lo if x <= lo else x
        v = hi if v >= hi else v
        return _F64(v)
    return _np.clip(x, lo, hi)
