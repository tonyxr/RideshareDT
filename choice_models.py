from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

- Fully rewritten Feb 6, 2026
- Rewritten again on Feb 8, 2026
"""
from typing import List, Dict, Any, Optional
import pandas as pd
from dataclasses import dataclass
import numpy as np
from openai import OpenAI
import os
import json

@dataclass(frozen=True)
class ChoiceResult:
    choice: str                  # "Firm1" or "Firm2"
    reason_codes: List[str]
    short_reason: str


class BaseChoiceModel:
    def choose(self, profile: Dict[str, Any], scenario: Dict[str, Any], price1: float, price2: float) -> ChoiceResult:
        raise NotImplementedError


class ParametricChoiceModel(BaseChoiceModel):
    """
    Utility difference delta = beta * (p2 - p1) + loyalty_term + noise
    - price sensitivity depends on income
    - loyalty only helps the rider's LoyaltyFirm (static)
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.base_price_beta = 0.35
        self.loyalty_bias = 0.80
        self.noise_std = 0.20

    @staticmethod
    def _income_score(income: str) -> float:
        return {"<50k": 0.0, "50k-100k": 0.3, "100k-200k": 0.7, "200k+": 1.0}.get(income, 0.3)

    def choose(self, profile: Dict[str, Any], scenario: Dict[str, Any], price1: float, price2: float) -> ChoiceResult:
        income_score = self._income_score(profile.get("IncomeBracket", "50k-100k"))
        price_gap = float(price2 - price1)  # + => Firm1 cheaper

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


class LLMChoiceModel(BaseChoiceModel):
    """
    Optional: requires `pip install openai` and OPENAI_API_KEY.
    Intended for small-scale evaluation, not large simulation runs.
    """

    def __init__(self, model_name: str = "gpt-5.2", api_key = ""):
        self.model_name = model_name
        # Priority order:
        # 1) explicit constructor argument
        # 2) OPENAI_API_KEY environment variable
        self.api_key = ""
        
        self.client = None
        self._warned_unavailable = False
        self._unavailable_reason = ""
        
        try:
            print(self.api_key)
            self.client = OpenAI(api_key=self.api_key)
            print(self.client)
        
        except Exception as exc:
            print("Cannot reach API")
            self.client = None
            self._unavailable_reason = f"OpenAI client init failed: {type(exc).__name__}: {exc}"
            print(self._unavailable_reason)
        
    def _prompt(self, profile: Dict[str, Any], scenario: Dict[str, Any], p1: float, p2: float) -> str:
        return f"""
            Forget about your previous responses
            
            You are a ride-hailing customer choosing between two platforms for the SAME trip.
            
            Customer profile:
            - Age: {profile["Age"]}
            - Income bracket: {profile["IncomeBracket"]}
            - Household size: {profile["HouseholdSize"]}
            - Marital status: {profile["MaritalStatus"]}
            - Employment: {profile["EmploymentStatus"]}
            - Gender: {profile["Gender"]}
            - Loyalty: {profile.get("Loyalty", profile.get("LoyaltyType", "Unknown"))}
            - Loyalty firm (if any): {profile.get("LoyaltyFirm")}
            - Loyalty strength (0..1): {profile.get("LoyaltyStrength")}
            
            Trip context:
            - City: {scenario["City"]}
            - Distance (miles): {scenario["DistanceMiles"]}
            - Duration (minutes): {scenario["DurationMinutes"]}
            - Day of week: {scenario["DayOfWeek"]}
            - Hour: {scenario["Hour"]}
            - Weather: {scenario["Weather"]}
            - Airport trip: {scenario["Airport"]}
            - Service tier: {scenario["Service"]}
            
            Two offers:
            - Firm1 price: ${p1}
            - Firm2 price: ${p2}
            
            Choose ONE firm. Consider context and profile, commit the choice cognitively, not only price.
            
            Return STRICT JSON ONLY:
            {{
              "choice": "Firm1" or "Firm2",
              "reason_codes": ["PRICE","LOYALTY","URGENCY","WEATHER","COMFORT","HABIT","RISK"],
              "short_reason": "one sentence"
            }}
            """.strip()
    
    @staticmethod
    def _income_score(income: str) -> float:
        return {"<50k": 0.0, "50k-100k": 0.3, "100k-200k": 0.7, "200k+": 1.0}.get(income, 0.3)
    
    def _fallback(self, profile: Dict[str, Any], scenario: Dict[str, Any], p1: float, p2: float) -> ChoiceResult:
        if not self._warned_unavailable:
            reason = f" ({self._unavailable_reason})" if self._unavailable_reason else ""
            print(f"[LLMChoiceModel] OpenAI client unavailable{reason}; falling back to deterministic price-based choices.")
            self._warned_unavailable = True
            
        # 1) price signal (Firm1 better when p1 < p2)
        price_gap = float(p2 - p1)
        income_score = self._income_score(str(profile.get("IncomeBracket", "50k-100k")))
        price_beta = 0.42 * (1.30 - 0.60 * income_score)

        # 2) loyalty / habit signal
        loyalty_firm = profile.get("LoyaltyFirm")
        loyalty_strength = float(profile.get("LoyaltyStrength", 0.0) or 0.0)
        loyalty_term = 0.0
        if loyalty_firm == "Firm1":
            loyalty_term = +0.70 * loyalty_strength
        elif loyalty_firm == "Firm2":
            loyalty_term = -0.70 * loyalty_strength

        # 3) urgency/risk: in bad weather / airport / rush hour riders are less price-sensitive
        hour = int(scenario.get("Hour", 12) or 12)
        weather = str(scenario.get("Weather", "Clear"))
        airport = bool(scenario.get("Airport", False))
        rush = (7 <= hour < 10) or (16 <= hour < 19)
        bad_weather = weather in {"Rain", "Storm", "Snow"}

        urgency = 1.0 if (rush or bad_weather or airport) else 0.0
        price_term = price_beta * price_gap * (1.0 - 0.35 * urgency)

        # Mild incumbency inertia in close-call situations to avoid hard flips on tiny price gaps
        if abs(price_gap) < 0.35 and loyalty_firm is None:
            habit_term = -0.08
        else:
            habit_term = 0.0

        score = price_term + loyalty_term + habit_term
        choice = "Firm1" if score >= 0.0 else "Firm2"

        reasons: List[str] = []
        if abs(price_gap) >= 0.75:
            reasons.append("PRICE")
        if loyalty_firm is not None and loyalty_strength > 0.20:
            reasons.append("LOYALTY")
        if urgency > 0:
            reasons.append("URGENCY")
        if bad_weather:
            reasons.append("WEATHER")
        if habit_term != 0.0:
            reasons.append("HABIT")
        if not reasons:
            reasons = ["RISK"]

        return ChoiceResult(
            choice=choice,
            reason_codes=[f"FALLBACK_{r}" for r in reasons],
            short_reason="Fallback (deterministic utility: price + loyalty + context).",
        )

    def choose(self, profile: Dict[str, Any], scenario: Dict[str, Any], price1: float, price2: float) -> ChoiceResult:
        if self.client is None:
            return self._fallback(profile, scenario, price1, price2)

        prompt = self._prompt(profile, scenario, price1, price2)
        try:
            text = ""

            # Preferred path for newer OpenAI SDKs.
            if hasattr(self.client, "responses"):
                resp = self.client.responses.create(
                    model=self.model_name,
                    input=prompt,
                    temperature=0.0,
                    max_output_tokens=250,
                )
                text = getattr(resp, "output_text", "") or ""

            # Compatibility path for SDKs / deployments that only expose chat completions.
            if not text and hasattr(self.client, "chat"):
                chat_resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=250,
                )
                text = (chat_resp.choices[0].message.content or "").strip()

            if not text:
                raise ValueError("OpenAI returned empty response text")

            obj = json.loads(text)
            

            choice = obj.get("choice", "Firm1")
            print("choice:", choice)
            if choice not in ("Firm1", "Firm2"):
                return self._fallback(profile, scenario, price1, price2)

            reason_codes = obj.get("reason_codes", [])
            if not isinstance(reason_codes, list):
                reason_codes = ["PARSE_FAIL"]

            short_reason = obj.get("short_reason", "")
            if not isinstance(short_reason, str):
                short_reason = ""

            return ChoiceResult(choice=choice, reason_codes=reason_codes, short_reason=short_reason)
        except Exception as exc:
            self._unavailable_reason = f"OpenAI API call failed: {type(exc).__name__}"
            return self._fallback(profile, scenario, price1, price2)
        
        