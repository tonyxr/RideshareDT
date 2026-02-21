from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

- Fully rewritten Feb 6, 2026
- Rewritten again on Feb 8, 2026
"""
from typing import List, Dict, Any, Optional
from openai import OpenAI
import pandas as pd
from dataclasses import dataclass
import numpy as np

import os
import json

OpenAI.api_key = "sk-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"  # <-- Replace this with your actual key before use
client = OpenAI(api_key = "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")

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

    def __init__(self, seed: int = 0):
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

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None

        if not self.api_key:
            return

        try:
            from openai import OpenAI  # lazy import
            self.client = OpenAI(api_key=self.api_key)
        except Exception:
            self.client = None

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
            - Loyalty: {profile["Loyalty"]}
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
            
            Choose ONE firm. Consider context and profile, not only price.
            
            Return STRICT JSON ONLY:
            {{
              "choice": "Firm1" or "Firm2",
              "reason_codes": ["PRICE","LOYALTY","URGENCY","WEATHER","COMFORT","HABIT","RISK"],
              "short_reason": "one sentence"
            }}
            """.strip()

    def _fallback(self, p1: float, p2: float) -> ChoiceResult:
        return ChoiceResult(
            choice="Firm1" if p1 <= p2 else "Firm2",
            reason_codes=["FALLBACK_PRICE"],
            short_reason="Fallback (LLM unavailable).",
        )

    def choose(self, profile: Dict[str, Any], scenario: Dict[str, Any], price1: float, price2: float) -> ChoiceResult:
        if self.client is None:
            return self._fallback(price1, price2)

        prompt = self._prompt(profile, scenario, price1, price2)
        try:
            resp = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                temperature=0.0,
                max_output_tokens=250,
            )
            text = getattr(resp, "output_text", "") or ""
            obj = json.loads(text)

            choice = obj.get("choice", "Firm1")
            if choice not in ("Firm1", "Firm2"):
                return self._fallback(price1, price2)

            reason_codes = obj.get("reason_codes", [])
            if not isinstance(reason_codes, list):
                reason_codes = ["PARSE_FAIL"]

            short_reason = obj.get("short_reason", "")
            if not isinstance(short_reason, str):
                short_reason = ""

            return ChoiceResult(choice=choice, reason_codes=reason_codes, short_reason=short_reason)
        except Exception:
            return self._fallback(price1, price2)
        
        