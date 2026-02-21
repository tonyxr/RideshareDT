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
        seed: Optional[int] = 42,
        total_customers: int = 20000,
        city_name: str = "General",
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

        # Create a static “population” pool so loyalty is fixed per rider across timesteps
        self._population = [self._draw_profile_static(i) for i in range(self.total_customers)]

    def _draw_profile_static(self, _i: int) -> Dict[str, Any]:
        age = int(np.clip(self.rng.normal(self.age_mean, self.age_std), 18, 80))
        income = str(self.rng.choice(self.income_names, p=self.income_probs))
        household = int(np.clip(self.rng.poisson(self.household_lambda), 1, 6))
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
            "LoyaltyType": loyalty_type,
            "LoyaltyFirm": loyalty_firm,
            "LoyaltyStrength": loyalty_strength,
        }

    def sample_profile(self) -> Dict[str, Any]:
        # sample a rider from the static population pool
        idx = int(self.rng.integers(0, self.total_customers))
        return dict(self._population[idx])  
        