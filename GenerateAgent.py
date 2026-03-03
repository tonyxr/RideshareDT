from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi
Customer profile sampler (city-conditioned extension can be added later).

- Fully rewritten Feb 6, 2026

"""

import numpy as np
from typing import Optional, Dict, Any

class GenerateAgent:
    # Approximate city demographic priors (can be replaced by external data source later)
    CITY_DEMOGRAPHICS: Dict[str, Dict[str, Any]] = {
        "General": {
            "age_mean": 36.0,
            "age_std": 12.0,
            "income_probs": [0.15, 0.35, 0.30, 0.20],
            "marital_probs": [0.50, 0.35, 0.10, 0.05],
            "gender_probs": [0.51, 0.49],
            "household_lambda": 2.0,
            "p_new": 0.30,
        },
        "Seattle": {
            "age_mean": 37.0,
            "age_std": 11.5,
            "income_probs": [0.14, 0.31, 0.32, 0.23],
            "marital_probs": [0.48, 0.37, 0.10, 0.05],
            "gender_probs": [0.50, 0.50],
            "household_lambda": 2.1,
            "p_new": 0.27,
        },
        "New York City": {
            "age_mean": 36.0,
            "age_std": 12.5,
            "income_probs": [0.21, 0.34, 0.29, 0.16],
            "marital_probs": [0.54, 0.31, 0.10, 0.05],
            "gender_probs": [0.48, 0.52],
            "household_lambda": 1.9,
            "p_new": 0.34,
        },
        "Chicago": {
            "age_mean": 38.0,
            "age_std": 12.0,
            "income_probs": [0.19, 0.36, 0.29, 0.16],
            "marital_probs": [0.50, 0.35, 0.10, 0.05],
            "gender_probs": [0.49, 0.51],
            "household_lambda": 2.2,
            "p_new": 0.31,
        },
    }

    def __init__(
        self,
        seed: Optional[int] = None,
        total_customers: int = 20000,
        city_name: str = "Seattle",
        loyalty_strength_range: tuple[float, float] = (0.4, 1.0),
    ):
        
        self.rng = np.random.default_rng(seed)
        self.total_customers = int(total_customers)
        self.loy_lo, self.loy_hi = float(loyalty_strength_range[0]), float(loyalty_strength_range[1])

        self.income_names = ['<50k', '50k-100k', '100k-200k', '200k+']
        
        self.marital_names = ['Single', 'Married', 'Divorced', 'Widowed']

        self.gender_names = ['Male', 'Female']
        
        self.city_name = city_name if city_name in self.CITY_DEMOGRAPHICS else "General"
        self.demographics = self.CITY_DEMOGRAPHICS[self.city_name]

        self.age_mean = float(self.demographics["age_mean"])
        self.age_std = float(self.demographics["age_std"])
        self.income_probs = np.array(self.demographics["income_probs"], dtype=float)
        self.marital_probs = np.array(self.demographics["marital_probs"], dtype=float)
        self.gender_probs = np.array(self.demographics["gender_probs"], dtype=float)
        self.household_lambda = float(self.demographics["household_lambda"])
        self.p_new = float(self.demographics["p_new"])

        # Create a compact static “population” pool so loyalty is fixed per rider across timesteps
        self._build_population()
    
    @staticmethod
    def _normalize_probs(probs: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.array(probs, dtype=float), 1e-6, None)
        total = float(np.sum(clipped))
        if total <= 0.0:
            return np.ones_like(clipped, dtype=float) / float(len(clipped))
        return clipped / total

    def apply_probability_variation(self, jitter_scale: float = 0.05) -> None:
        """Apply slight run-level variation to demographic priors, then rebuild pool."""
        j = float(max(0.0, jitter_scale))
        base = self.CITY_DEMOGRAPHICS[self.city_name]

        self.age_mean = float(base["age_mean"] + self.rng.normal(0.0, 1.0 * j))
        self.age_std = float(max(5.0, base["age_std"] * (1.0 + self.rng.normal(0.0, 0.15 * j))))
        self.income_probs = self._normalize_probs(
            np.array(base["income_probs"], dtype=float) + self.rng.normal(0.0, j, len(base["income_probs"]))
        )
        self.marital_probs = self._normalize_probs(
            np.array(base["marital_probs"], dtype=float) + self.rng.normal(0.0, j, len(base["marital_probs"]))
        )
        self.gender_probs = self._normalize_probs(
            np.array(base["gender_probs"], dtype=float) + self.rng.normal(0.0, j, len(base["gender_probs"]))
        )
        self.household_lambda = float(max(1.1, base["household_lambda"] * (1.0 + self.rng.normal(0.0, 0.1 * j))))
        self.p_new = float(np.clip(base["p_new"] + self.rng.normal(0.0, 0.08 * j), 0.05, 0.95))

        self._build_population()


    def _build_population(self) -> None:
        """Build a compact vectorized customer pool to reduce RAM usage."""
        n = int(max(0, self.total_customers))
        self._ages = np.clip(self.rng.normal(self.age_mean, self.age_std, size=n), 18, 80).astype(np.int8)
        self._income_idx = self.rng.choice(len(self.income_names), size=n, p=self.income_probs).astype(np.int8)
        self._household = np.clip(self.rng.poisson(self.household_lambda, size=n), 1, 6).astype(np.int8)
        self._marital_idx = self.rng.choice(len(self.marital_names), size=n, p=self.marital_probs).astype(np.int8)

        self._employment_idx = np.empty(n, dtype=np.int8)
        young_mask = self._ages < 22
        working_mask = (self._ages >= 22) & (self._ages < 65)
        senior_mask = self._ages >= 65

        self._employment_idx[young_mask] = self.rng.choice(
            [0, 1], size=int(np.sum(young_mask)), p=[0.7, 0.3]
        ).astype(np.int8)
        self._employment_idx[working_mask] = self.rng.choice(
            [1, 2], size=int(np.sum(working_mask)), p=[0.9, 0.1]
        ).astype(np.int8)
        self._employment_idx[senior_mask] = self.rng.choice(
            [3, 1], size=int(np.sum(senior_mask)), p=[0.75, 0.25]
        ).astype(np.int8)

        self._gender_idx = self.rng.choice(len(self.gender_names), size=n, p=self.gender_probs).astype(np.int8)

        is_returning = self.rng.random(n) >= self.p_new
        self._is_returning = is_returning
        self._loyalty_firm_idx = np.full(n, -1, dtype=np.int8)
        self._loyalty_firm_idx[is_returning] = self.rng.integers(0, 2, size=int(np.sum(is_returning)), dtype=np.int8)
        self._loyalty_strength = np.zeros(n, dtype=np.float32)
        self._loyalty_strength[is_returning] = self.rng.uniform(
            self.loy_lo, self.loy_hi, size=int(np.sum(is_returning))
        ).astype(np.float32)

    def sample_profiles(self, n: int) -> list[Dict[str, Any]]:
        count = int(max(0, n))
        if count == 0:
            return []
        idxs = self.rng.integers(0, self.total_customers, size=count)
        employment_names = ["Student", "Employed", "Unemployed", "Retired"]
        loyalty_firms = ["Firm1", "Firm2"]
        return [
            {
                "Age": int(self._ages[int(i)]),
                "IncomeBracket": self.income_names[int(self._income_idx[int(i)])],
                "HouseholdSize": int(self._household[int(i)]),
                "MaritalStatus": self.marital_names[int(self._marital_idx[int(i)])],
                "EmploymentStatus": employment_names[int(self._employment_idx[int(i)])],
                "Gender": self.gender_names[int(self._gender_idx[int(i)])],
                "LoyaltyType": "Returning" if bool(self._is_returning[int(i)]) else "New",
                "LoyaltyFirm": loyalty_firms[int(self._loyalty_firm_idx[int(i)])] if self._loyalty_firm_idx[int(i)] >= 0 else None,
                "LoyaltyStrength": float(self._loyalty_strength[int(i)]),
            }
            for i in idxs
        ]
        