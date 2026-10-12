#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 21:29:28 2026

@author: Xiaoru Shi
"""

"""Process-wide startup defaults for local experiment runs.

Python imports ``sitecustomize`` before user modules when this repository is on
``PYTHONPATH``/the working directory.  Keep native numerical-library settings
here so they are applied before NumPy, Torch, SciPy, or scikit-learn can load
Intel oneMKL and emit noisy CPU-instruction deprecation banners.
"""

from __future__ import annotations

import os

# Match ``mkl_config.py`` and avoid forcing oneMKL onto the deprecated SSE4.2
# dispatcher. Respect an explicit experiment-level override.
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX")
os.environ.setdefault("KMP_WARNINGS", "0")
