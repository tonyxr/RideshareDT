from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 21:25:19 2026

@author: Xiaoru Shi
"""

"""Numerical-library startup defaults used before importing NumPy/Torch.

Intel oneMKL prints a noisy SSE4.2 deprecation banner when its dispatcher is
forced onto the legacy SSE4.2 path.  Import this module before NumPy, Torch,
SciPy, or scikit-learn in executable paths.
"""

import os

# Respect explicit user choices while using the minimum instruction set
# supported by current/future oneMKL releases.  Forcing SSE4_2 here caused the
# deprecation warning this module was intended to suppress.
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX")
os.environ.setdefault("KMP_WARNINGS", "0")
