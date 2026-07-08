from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 21:25:19 2026

@author: Xiaoru Shi
"""

"""Numerical-library startup defaults used before importing NumPy/Torch.

Intel oneMKL prints a noisy SSE4.2 deprecation banner on older CPUs unless the
instruction policy is set before the native library is loaded.  Import this
module before NumPy, Torch, SciPy, or scikit-learn in executable paths.
"""

import os

# Respect explicit user choices while providing quiet defaults for local runs.
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("KMP_WARNINGS", "0")