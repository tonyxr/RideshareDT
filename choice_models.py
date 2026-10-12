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
import mkl_config  # set oneMKL env before NumPy/Torch
del mkl_config
import numpy as np

from gpt_threshold_utils import read_profile_price_threshold


@dataclass(frozen=True)
class ChoiceResult:
    choice: str  # "Firm1", "Firm2", or "NoRide"
    reason_codes: List[str]
    short_reason: str


class BaseChoiceModel:
    # v26 city price level.  Reservation fares and outside-option costs were
    # specified in NYC dollars; ``price_level_index`` rescales them to the
    # calibrated city's reference fare (NYC = 1.0).  ``outside_option_base_fare``
    # is the city's transit fare.  Defaults reproduce the legacy model.
    price_level_index: float = 1.0
    outside_option_base_fare: float = 2.90
    LEGACY_PRICE_LEVEL_INDEX = 1.0
    LEGACY_OUTSIDE_OPTION_BASE_FARE = 2.90
    experimental_unbounded_economics: bool = False

    def _profile_price_threshold(self, profile: Dict[str, Any]) -> float:
        """Read a rider threshold under the selected experiment contract."""
        return read_profile_price_threshold(
            profile,
            0.25,
            8.00,
            experimental_unbounded_economics=bool(
                getattr(self, "experimental_unbounded_economics", False)
            ),
        )

    def _apply_price_level(self, params: Dict[str, float]) -> None:
        """Record the calibrated city price level; it takes effect only after
        ``activate_price_level(True)`` (the v26 environment), so a legacy run
        that reads a v26 calibration file keeps the v25 NYC-dollar constants."""
        if not isinstance(params, dict):
            return
        pending = dict(getattr(self, "_calibrated_price_level", {}) or {})
        if params.get("price_level_index") is not None:
            pending["price_level_index"] = float(np.clip(float(params["price_level_index"]), 0.25, 4.0))
        if params.get("outside_option_base_fare") is not None:
            pending["outside_option_base_fare"] = float(np.clip(float(params["outside_option_base_fare"]), 0.0, 20.0))
        self._calibrated_price_level = pending
        if getattr(self, "_price_level_active", False):
            self.activate_price_level(True)

    def activate_price_level(self, enabled: bool) -> None:
        self._price_level_active = bool(enabled)
        pending = dict(getattr(self, "_calibrated_price_level", {}) or {})
        if enabled:
            self.price_level_index = float(pending.get("price_level_index", self.LEGACY_PRICE_LEVEL_INDEX))
            self.outside_option_base_fare = float(
                pending.get("outside_option_base_fare", self.LEGACY_OUTSIDE_OPTION_BASE_FARE)
            )
        else:
            self.price_level_index = self.LEGACY_PRICE_LEVEL_INDEX
            self.outside_option_base_fare = self.LEGACY_OUTSIDE_OPTION_BASE_FARE

    def choose(self, profile: Dict[str, Any], scenario: Dict[str, Any], price1: float, price2: float) -> ChoiceResult:
        raise NotImplementedError
        
    def apply_calibration(self, params: Dict[str, float]) -> None:
        self._apply_price_level(params)
        return None
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        # Stable without clipping the latent utility.  Clipping at +/-60
        # creates a small but strictly positive demand floor which is harmless
        # on the legacy bounded tariff grid, but makes profit diverge when a
        # v28 fare is deliberately allowed to grow without an economic cap.
        x = float(x)
        if x >= 0.0:
            return 1.0 / (1.0 + float(np.exp(-x)))
        exp_x = float(np.exp(x))
        return exp_x / (1.0 + exp_x)

    def _outside_probability_ceiling(self) -> float:
        """Legacy kept a 2% ride floor; v28 must permit demand to vanish."""
        return (
            1.0
            if bool(getattr(self, "experimental_unbounded_economics", False))
            else 0.98
        )

    @staticmethod
    def _apply_outside_odds_factor(probability: float, factor: float) -> float:
        """Apply rider-memory influence to odds, preserving p -> 1 tails."""
        probability = float(np.clip(probability, 0.0, 1.0))
        factor = float(factor)
        if not np.isfinite(factor) or factor <= 0.0:
            raise FloatingPointError(
                "outside-option reputation odds must be positive and finite"
            )
        numerator = probability * factor
        denominator = (1.0 - probability) + numerator
        return float(numerator / denominator) if denominator > 0.0 else 1.0

    @staticmethod
    def _income_score(income: str) -> float:
        return {"<50k": 0.0, "50k-100k": 0.3, "100k-200k": 0.7, "200k+": 1.0}.get(str(income), 0.3)

    def _reservation_price(self, profile: Dict[str, Any], scenario: Dict[str, Any]) -> float:
        """Estimate the maximum acceptable rideshare fare before outside options become attractive."""
        explicit = profile.get("ReservationPrice", profile.get("RideshareReservationPrice"))
        if explicit is not None:
            return float(np.clip(float(explicit), 5.0, 120.0))

        income_score = self._income_score(profile.get("IncomeBracket", "50k-100k"))
        household = int(profile.get("HouseholdSize", 1) or 1)
        loyalty_strength = float(profile.get("LoyaltyStrength", 0.0) or 0.0)
        returning = str(profile.get("LoyaltyType", "New")) == "Returning"
        distance = float(scenario.get("DistanceMiles", 4.0) or 4.0)
        duration = float(scenario.get("DurationMinutes", 15.0) or 15.0)
        service = str(scenario.get("Service", "economy") or "economy").lower()
        hour = int(scenario.get("Hour", 12) or 12)
        weather = str(scenario.get("Weather", "clear") or "clear").lower()
        airport = bool(scenario.get("Airport", False))
        rush = (7 <= hour < 10) or (16 <= hour < 19)
        bad_weather = weather in {"rain", "snow", "storm"}

        reservation = 11.0 + 26.0 * income_score
        reservation += 0.95 * distance + 0.10 * duration
        reservation += 8.0 if service == "premium" else 0.0
        reservation += 7.0 if airport else 0.0
        reservation += 2.5 if rush else 0.0
        reservation += 2.5 if bad_weather else 0.0
        reservation += (2.0 + 3.0 * loyalty_strength) if returning else 0.0
        reservation -= 1.25 * min(max(household - 1, 0), 4)
        level = float(self.price_level_index)
        return float(np.clip(reservation * level, 5.0 * level, 120.0 * level))

    def _outside_option_cost(self, profile: Dict[str, Any], scenario: Dict[str, Any]) -> float:
        explicit = profile.get("OutsideOptionCost", profile.get("AlternativeTransportCost"))
        if explicit is not None:
            return float(np.clip(float(explicit), 0.0, 80.0))
        distance = float(scenario.get("DistanceMiles", 4.0) or 4.0)
        airport = bool(scenario.get("Airport", False))
        level = float(self.price_level_index)
        return float(np.clip(
            float(self.outside_option_base_fare) + level * (0.20 * distance + (9.0 if airport else 0.0)),
            0.0,
            80.0 * level,
        ))

    def _outside_option_inconvenience(self, profile: Dict[str, Any], scenario: Dict[str, Any]) -> float:
        explicit = profile.get("OutsideOptionInconvenience")
        if explicit is not None:
            return float(np.clip(float(explicit), 0.0, 4.0))
        distance = float(scenario.get("DistanceMiles", 4.0) or 4.0)
        hour = int(scenario.get("Hour", 12) or 12)
        weather = str(scenario.get("Weather", "clear") or "clear").lower()
        service = str(scenario.get("Service", "economy") or "economy").lower()
        airport = bool(scenario.get("Airport", False))
        rush = (7 <= hour < 10) or (16 <= hour < 19)
        bad_weather = weather in {"rain", "snow", "storm"}
        income_score = self._income_score(profile.get("IncomeBracket", "50k-100k"))

        inconvenience = 0.35 + 0.05 * min(distance, 15.0)
        inconvenience += 0.45 if bad_weather else 0.0
        inconvenience += 0.55 if airport else 0.0
        inconvenience += 0.20 if rush else 0.0
        inconvenience += 0.25 + 0.25 * income_score if service == "premium" else 0.0
        return float(np.clip(inconvenience, 0.0, 4.0))

    def _no_ride_probability(self, profile: Dict[str, Any], scenario: Dict[str, Any], price1: float, price2: float) -> float:
        """Probability that the rider rejects both quotes in favor of no ride/outside transport."""
        best_price = float(min(price1, price2))
        reservation = self._reservation_price(profile, scenario)
        outside_cost = self._outside_option_cost(profile, scenario)
        inconvenience = self._outside_option_inconvenience(profile, scenario)
        income_score = self._income_score(profile.get("IncomeBracket", "50k-100k"))
        household = int(profile.get("HouseholdSize", 1) or 1)
        loyalty_strength = float(profile.get("LoyaltyStrength", 0.0) or 0.0)

        affordability_pressure = (best_price - reservation) / max(reservation, 1e-6)
        outside_savings = max(0.0, best_price - outside_cost) / max(reservation, 1e-6)
        household_pressure = 0.08 * min(max(household - 1, 0), 4)
        loyalty_pull = 0.35 * loyalty_strength
        income_buffer = 0.35 * income_score

        latent = (
            -2.15
            + 4.00 * affordability_pressure
            + 0.65 * outside_savings
            + household_pressure
            - 0.70 * inconvenience
            - loyalty_pull
            - income_buffer
        )
        return float(
            np.clip(
                self._sigmoid(latent),
                0.0,
                self._outside_probability_ceiling(),
            )
        )

class ParametricChoiceModel(BaseChoiceModel):
    """Simple baseline model: price + loyalty + random noise."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.base_price_beta = 0.35
        self.loyalty_bias = 0.80
        self.noise_std = 0.20
        
    def apply_calibration(self, params: Dict[str, float]) -> None:
        self._apply_price_level(params)
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
        
        scenario = dict(scenario or {})
        p_no_ride = self._no_ride_probability(profile, scenario, price1, price2)
        if bool(self.rng.random() < p_no_ride):
            reservation = self._reservation_price(profile, scenario)
            best_price = float(min(price1, price2))
            reasons = ["OUTSIDE_OPTION"]
            if best_price > reservation:
                reasons.append("TOO_EXPENSIVE")
            return ChoiceResult(
                choice="NoRide",
                reason_codes=reasons,
                short_reason="Parametric outside-option choice (both rideshare quotes rejected).",
            )

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

    A separate outside-option gate can reject both rideshare quotes when the best
    absolute fare exceeds the rider's reservation price enough to make no ride or
    alternative transportation attractive. Conditional on taking rideshare, choice
    is probabilistic via logistic response, with temperature tuned by context so
    close offers remain stochastic while large utility gaps become decisive.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.price_sensitivity_scale = 1.0
        self.loyalty_scale = 1.0
        self.reliability_scale = 1.0
        # Legacy firm-identity constants (v25): non-loyal riders lean toward
        # Firm 2 near price ties, and stressed riders lean slightly toward
        # Firm 1.  ``make_symmetric`` zeroes both so platform identity carries
        # no intrinsic utility (v26 symmetric firms).
        self.near_tie_habit_firm2 = 0.03
        self.reliability_bias_firm1 = 0.02
        # Optional v26 rider memory (market_dynamics.FirmReputationMemory).
        self.reputation = None
        # Optional identity-specific memory. It is populated by the sequential
        # vector market and kept separate from immutable profile traits.
        self.profile_memory = None

    def make_symmetric(self) -> None:
        self.near_tie_habit_firm2 = 0.0
        self.reliability_bias_firm1 = 0.0

    def apply_calibration(self, params: Dict[str, float]) -> None:
        self._apply_price_level(params)
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
        
        price_threshold = self._profile_price_threshold(profile)

        # ---- Economic utility ----
        # Low income + larger household -> stronger price pain.
        price_sensitivity = (0.30 + 0.45 * (1.0 - income_score) + 0.05 * min(max(household - 1, 0), 3)) * self.price_sensitivity_scale
        # Urgent contexts compress cross-firm effective differences (reduced shopping intensity).
        urgency = 0.40 * float(rush) + 0.35 * float(bad_weather) + 0.30 * float(airport)
        shopping_intensity = max(0.45, 1.0 - 0.45 * urgency)

        # Nonlinear price pain: people react more strongly once a fare gap clears
        # their rider-specific threshold. Smaller gaps still matter, but are less salient.
        price_gap = float(price2 - price1)
        wait_gap = float(scenario.get("WaitEstimateFirm2", 0.0) or 0.0) - float(scenario.get("WaitEstimateFirm1", 0.0) or 0.0)
        cancel_gap = float(scenario.get("CancelRiskFirm2", 0.0) or 0.0) - float(scenario.get("CancelRiskFirm1", 0.0) or 0.0)
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
        near_tie_habit = float(getattr(self, "near_tie_habit_firm2", 0.03))
        if loyalty_firm is None and abs(price_gap) < 0.60 and near_tie_habit != 0.0:
            habit_inertia = -near_tie_habit * (2.0 if weekend else 1.0)
            
        # ---- Reliability/risk context ----
        # In stressful contexts, users weigh reliability and predictability more.
        risk_aversion = 0.20 + 0.30 * float(age >= 55) + 0.20 * employment_flex
        reliability_pressure = risk_aversion * (0.45 * float(bad_weather) + 0.35 * float(airport) + 0.20 * float(rush))
        
        # Small structural preference drift (can be calibrated later with data).
        reliability_bias_firm1 = float(getattr(self, "reliability_bias_firm1", 0.02))
        wait_value = (0.035 + 0.035 * employment_flex + 0.025 * float(rush or airport)) * self.reliability_scale
        reliability_term = (
            reliability_pressure * reliability_bias_firm1 * self.reliability_scale
            + wait_value * wait_gap
            + 0.85 * reliability_pressure * cancel_gap
        )
        
        # Convenience/service framing: premium riders less price-sensitive and more inertial.
        comfort_term = 0.0
        if service == "premium":
            comfort_term = 0.05 + 0.10 * income_score
        
        # Longer trips => stronger reaction to absolute gap.
        trip_scale = 1.0 + 0.06 * min(distance / 10.0, 2.0)

        reputation = getattr(self, "reputation", None)
        memory_term = (
            float(reputation.choice_utility(price_sensitivity))
            if reputation is not None
            else 0.0
        )
        profile_memory = getattr(self, "profile_memory", None)
        profile_choice_memory = 0.0
        profile_outside_memory = 0.0
        if profile_memory is not None:
            choice_values, outside_values = profile_memory.features([profile])
            profile_choice_memory = float(choice_values[0])
            profile_outside_memory = float(outside_values[0])

        latent = (
            trip_scale
            * (price_term + loyalty_term + habit_inertia + reliability_term + comfort_term)
            + memory_term
            + profile_choice_memory
        )

        # Choice stochasticity: lower temp in urgent contexts -> more deterministic.
        temperature = 0.70 - 0.15 * float(rush) - 0.10 * float(bad_weather) - 0.05 * float(airport)
        temperature = float(np.clip(temperature, 0.35, 0.90))
        
        p_no_ride = self._no_ride_probability(profile, scenario, price1, price2)
        if profile_outside_memory != 0.0:
            odds = float(np.exp(np.clip(profile_outside_memory, -60.0, 60.0)))
            p_no_ride = self._apply_outside_odds_factor(p_no_ride, odds)
        if reputation is not None:
            if bool(getattr(self, "experimental_unbounded_economics", False)):
                adjusted = self._apply_outside_odds_factor(
                    p_no_ride, reputation.outside_option_factor()
                )
            else:
                # Preserve exact v27 probability-multiplier semantics.  The
                # odds transform is the v28 uncapped-tail correction only.
                adjusted = p_no_ride * reputation.outside_option_factor()
            p_no_ride = float(
                np.clip(
                    adjusted,
                    0.0,
                    self._outside_probability_ceiling(),
                )
            )
        if bool(self.rng.random() < p_no_ride):
            reservation = self._reservation_price(profile, scenario)
            best_price = float(min(price1, price2))
            reasons = ["OUTSIDE_OPTION"]
            if best_price > reservation:
                reasons.append("TOO_EXPENSIVE")
            if rush or airport:
                reasons.append("URGENCY")
            if bad_weather:
                reasons.append("WEATHER")
            return ChoiceResult(
                choice="NoRide",
                reason_codes=[f"COG_{r}" for r in reasons],
                short_reason="Cognitive outside-option choice (rideshare rejected for no ride/alternative transport).",
            )

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
        if abs(wait_gap) >= 3.0 or abs(cancel_gap) >= 0.15:
            reasons.append("RELIABILITY")
            
        if service == "premium":
            reasons.append("COMFORT")
        if habit_inertia != 0.0:
            reasons.append("HABIT")
        if abs(memory_term) >= 0.05:
            reasons.append("REPUTATION")
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
