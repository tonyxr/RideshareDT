from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi
Customer profile sampler (city-conditioned extension can be added later).

- Fully rewritten Feb 6, 2026

"""

import numpy as np
from typing import Optional, Dict, Any, List

class GenerateAgent:
    def __init__(
        self,
        seed: Optional[int] = 42,
        total_customers: int = 20000,
        p_new: float = 0.30,
        loyalty_strength_range: tuple[float, float] = (0.4, 1.0),
    ):
        self.rng = np.random.default_rng(seed)
        self.total_customers = int(total_customers)
        self.p_new = float(p_new)
        self.loy_lo, self.loy_hi = float(loyalty_strength_range[0]), float(loyalty_strength_range[1])

        self.income_names = ['<50k', '50k-100k', '100k-200k', '200k+']
        self.income_probs = np.array([0.15, 0.35, 0.30, 0.20], dtype=float)

        self.marital_names = ['Single', 'Married', 'Divorced', 'Widowed']
        self.marital_probs = np.array([0.50, 0.35, 0.10, 0.05], dtype=float)

        self.gender_names = ['Male', 'Female']
        self.gender_probs = np.array([0.51, 0.49], dtype=float)

        # Create a static “population” pool so loyalty is fixed per rider across timesteps
        self._population = [self._draw_profile_static(i) for i in range(self.total_customers)]

    def _draw_profile_static(self, _i: int) -> Dict[str, Any]:
        age = int(np.clip(self.rng.normal(36, 12), 18, 80))
        income = str(self.rng.choice(self.income_names, p=self.income_probs))
        household = int(np.clip(self.rng.poisson(2), 1, 5))
        marital = str(self.rng.choice(self.marital_names, p=self.marital_probs))

        if age < 22:
            employment = str(self.rng.choice(['Student', 'Employed'], p=[0.7, 0.3]))
        elif age < 65:
            employment = str(self.rng.choice(['Employed', 'Unemployed'], p=[0.9, 0.1]))
        else:
            employment = str(self.rng.choice(['Retired', 'Employed'], p=[0.75, 0.25]))

        gender = str(self.rng.choice(self.gender_names, p=self.gender_probs))
        loyalty_type = str(self.rng.choice(['New', 'Returning'], p=[self.p_new, 1.0 - self.p_new]))

        loyalty_firm = None
        loyalty_strength = 0.0
        if loyalty_type == "Returning":
            # uniform split across firm1/firm2
            loyalty_firm = "Firm1" if (self.rng.random() < 0.5) else "Firm2"
            loyalty_strength = float(self.rng.uniform(self.loy_lo, self.loy_hi))

        return {
            "Age": age,
            "IncomeBracket": income,
            "HouseholdSize": household,
            "MaritalStatus": marital,
            "EmploymentStatus": employment,
            "Gender": gender,
            "LoyaltyType": loyalty_type,          # New / Returning
            "LoyaltyFirm": loyalty_firm,          # Firm1 / Firm2 / None
            "LoyaltyStrength": loyalty_strength,  # numeric
        }

    def sample_profile(self) -> Dict[str, Any]:
        # sample a rider from the static population pool
        idx = int(self.rng.integers(0, self.total_customers))
        return dict(self._population[idx])  # copy to prevent mutation bugs
        