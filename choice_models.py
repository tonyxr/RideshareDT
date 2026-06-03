from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

- Fully rewritten Feb 6, 2026
- Rewritten again on Feb 8, 2026
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ChoiceResult:
    choice: str  # "Firm1" or "Firm2"
    reason_codes: List[str]
    short_reason: str


class BaseChoiceModel:
    def choose(self, profile: Dict[str, Any], scenario: Dict[str, Any], price1: float, price2: float) -> ChoiceResult:
        raise NotImplementedError
        
    def apply_calibration(self, params: Dict[str, float]) -> None:
        del params
        return None

class ParametricChoiceModel(BaseChoiceModel):
    """Simple baseline model: price + loyalty + random noise."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.base_price_beta = 0.35
        self.loyalty_bias = 0.80
        self.noise_std = 0.20
        
    def apply_calibration(self, params: Dict[str, float]) -> None:
        ps = float(params.get("price_sensitivity_scale", 1.0))
        ls = float(params.get("loyalty_scale", 1.0))
        self.base_price_beta = float(np.clip(0.35 * ps, 0.05, 1.20))
        self.loyalty_bias = float(np.clip(0.80 * ls, 0.10, 2.00))

    @staticmethod
    def _income_score(income: str) -> float:
        return {"<50k": 0.0, "50k-100k": 0.3, "100k-200k": 0.7, "200k+": 1.0}.get(income, 0.3)

    def choose(self, profile: Dict[str, Any], scenario: Dict[str, Any], price1: float, price2: float) -> ChoiceResult:
        income_score = self._income_score(profile.get("IncomeBracket", "50k-100k"))
        price_gap = float(price2 - price1)

        # lower income => higher price sensitivity
        beta = self.base_price_beta * (1.25 - 0.5 * income_score)

        # static firm loyalty
        loyalty_firm = profile.get("LoyaltyFirm", None)
        strength = float(profile.get("LoyaltyStrength", 0.0))
        loy = 0.0
        if loyalty_firm == "Firm1":
            loy = +self.loyalty_bias * strength
        elif loyalty_firm == "Firm2":
            loy = -self.loyalty_bias * strength

        eps = float(self.rng.normal(0.0, self.noise_std))
        delta = beta * price_gap + loy + eps

        choice = "Firm1" if delta >= 0 else "Firm2"

        reasons: List[str] = []
        if abs(price_gap) >= 1.0:
            reasons.append("PRICE")
        if loyalty_firm is not None and strength > 0.2:
            reasons.append("LOYALTY")
        if not reasons:
            reasons = ["NOISE"]

        return ChoiceResult(choice=choice, reason_codes=reasons, short_reason="Parametric choice (price + static loyalty).")


class CognitiveChoiceModel(BaseChoiceModel):
    """
    Higher-fidelity behavioral model for rider platform choice.

    Utility(Firm1 - Firm2) combines:
    - Economic utility: nonlinear price pain with rider heterogeneity and rider-level price thresholds
    - Habit/loyalty inertia: persistent preference for prior platform
    - Reliability-risk sensitivity: stronger in airport, bad weather, rush periods
    - Convenience effects: trip urgency and service context

    Choice is probabilistic via logistic response, with temperature tuned by context
    so close offers remain stochastic while large utility gaps become decisive.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.price_sensitivity_scale = 1.0
        self.loyalty_scale = 1.0
        self.reliability_scale = 1.0

    def apply_calibration(self, params: Dict[str, float]) -> None:
        self.price_sensitivity_scale = float(np.clip(params.get("price_sensitivity_scale", 1.0), 0.30, 2.80))
        self.loyalty_scale = float(np.clip(params.get("loyalty_scale", 1.0), 0.30, 2.80))
        self.reliability_scale = float(np.clip(params.get("reliability_scale", 1.0), 0.30, 2.80))

    @staticmethod
    def _income_score(income: str) -> float:
        return {"<50k": 0.0, "50k-100k": 0.3, "100k-200k": 0.7, "200k+": 1.0}.get(str(income), 0.3)
    
    @staticmethod
    def _employment_flexibility(employment: str) -> float:
        # Larger => more schedule-constrained / less flexible.
        table = {
            "Student": 0.35,
            "Employed": 0.70,
            "Unemployed": 0.25,
            "Retired": 0.30,
        }
        return float(table.get(str(employment), 0.45))

    def choose(self, profile: Dict[str, Any], scenario: Dict[str, Any], price1: float, price2: float) -> ChoiceResult:
        # ---- Profile traits ----
        age = int(profile.get("Age", 35) or 35)
        income_score = self._income_score(profile.get("IncomeBracket", "50k-100k"))
        household = int(profile.get("HouseholdSize", 1) or 1)
        employment_flex = self._employment_flexibility(profile.get("EmploymentStatus", "Employed"))
        
        loyalty_firm = profile.get("LoyaltyFirm")
        loyalty_strength = float(profile.get("LoyaltyStrength", 0.0) or 0.0)
        
        # ---- Scenario traits ----
        hour = int(scenario.get("Hour", 12) or 12)
        
        weather = str(scenario.get("Weather", "clear") or "clear").lower()
        airport = bool(scenario.get("Airport", False))
        service = str(scenario.get("Service", "economy") or "economy").lower()
        distance = float(scenario.get("DistanceMiles", 4.0) or 4.0)
        day = int(scenario.get("DayOfWeek", 2) or 2)

        rush = (7 <= hour < 10) or (16 <= hour < 19)
        weekend = day >= 5
        bad_weather = weather in {"rain", "snow", "storm"}
        
        price_threshold = float(np.clip(float(profile.get("PriceThreshold", 1.50) or 1.50), 0.25, 8.00))

        # ---- Economic utility ----
        # Low income + larger household -> stronger price pain.
        price_sensitivity = (0.30 + 0.45 * (1.0 - income_score) + 0.05 * min(max(household - 1, 0), 3)) * self.price_sensitivity_scale
        # Urgent contexts compress cross-firm effective differences (reduced shopping intensity).
        urgency = 0.40 * float(rush) + 0.35 * float(bad_weather) + 0.30 * float(airport)
        shopping_intensity = max(0.45, 1.0 - 0.45 * urgency)

        # Nonlinear price pain: people react more strongly once a fare gap clears
        # their rider-specific threshold. Smaller gaps still matter, but are less salient.
        price_gap = float(price2 - price1)
        threshold_ratio = abs(price_gap) / max(price_threshold, 1e-6)
        price_salience = 0.35 + 0.65 * min(threshold_ratio, 1.0) + 0.25 * max(threshold_ratio - 1.0, 0.0)
        price_salience = float(np.clip(price_salience, 0.20, 2.25))
        price_term = price_sensitivity * np.sign(price_gap) * (abs(price_gap) ** 0.92) * shopping_intensity * price_salience

        # ---- Habit / loyalty ----
        loyalty_term = 0.0
        if loyalty_firm == "Firm1":
            loyalty_term = +0.85 * self.loyalty_scale * loyalty_strength
        elif loyalty_firm == "Firm2":
            loyalty_term = -0.85 * self.loyalty_scale * loyalty_strength

        # Inertia for non-loyal users near ties.
        habit_inertia = 0.0
        if loyalty_firm is None and abs(price_gap) < 0.60:
            habit_inertia = -0.06 if weekend else -0.03
            
        # ---- Reliability/risk context ----
        # In stressful contexts, users weigh reliability and predictability more.
        risk_aversion = 0.20 + 0.30 * float(age >= 55) + 0.20 * employment_flex
        reliability_pressure = risk_aversion * (0.45 * float(bad_weather) + 0.35 * float(airport) + 0.20 * float(rush))
        
        # Small structural preference drift (can be calibrated later with data).
        reliability_bias_firm1 = 0.02
        reliability_term = reliability_pressure * reliability_bias_firm1 * self.reliability_scale
        
        # Convenience/service framing: premium riders less price-sensitive and more inertial.
        comfort_term = 0.0
        if service == "premium":
            comfort_term = 0.05 + 0.10 * income_score
        
        # Longer trips => stronger reaction to absolute gap.
        trip_scale = 1.0 + 0.06 * min(distance / 10.0, 2.0)

        latent = trip_scale * (price_term + loyalty_term + habit_inertia + reliability_term + comfort_term)

        # Choice stochasticity: lower temp in urgent contexts -> more deterministic.
        temperature = 0.70 - 0.15 * float(rush) - 0.10 * float(bad_weather) - 0.05 * float(airport)
        temperature = float(np.clip(temperature, 0.35, 0.90))

        p_f1 = 1.0 / (1.0 + float(np.exp(-latent / temperature)))
        choose_f1 = bool(self.rng.random() < p_f1)
        choice = "Firm1" if choose_f1 else "Firm2"
        
        reasons: List[str] = []
        if abs(price_gap) >= price_threshold:
            reasons.append("PRICE")
        if loyalty_firm is not None and loyalty_strength > 0.2:
            reasons.append("LOYALTY")
        
        if rush or airport:
            reasons.append("URGENCY")
        if bad_weather:
            reasons.append("WEATHER")
            
        if service == "premium":
            reasons.append("COMFORT")
        if habit_inertia != 0.0:
            reasons.append("HABIT")
        if reliability_pressure > 0.12:
            reasons.append("RISK")
        if not reasons:
            reasons = ["UTILITY"]
            
        return ChoiceResult(
            choice=choice,
            reason_codes=[f"COG_{r}" for r in reasons],
            short_reason="Cognitive utility choice (economic + loyalty + context risk, probabilistic).",
        )
    
        
class LLMChoiceModel(CognitiveChoiceModel):
    """
    Backward-compatible alias.

    GPT-assisted rider price-threshold generation is handled at profile bootstrap time in Core.
    At choice time this class consumes profile-level PriceThreshold values produced upstream.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None, seed: Optional[int] = None):
        self.model_name = model_name
        self.api_key = api_key
        super().__init__(seed=seed)
        print("[ChoiceModel] GPT price-threshold mode enabled (thresholds consumed from profile).")