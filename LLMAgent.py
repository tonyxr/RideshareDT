from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

- Fully rewritten Feb 6, 2026
"""
from typing import Literal, Dict, Any, Optional
import openai
import pandas as pd
from openai import OpenAI


import os
import json

class LLMCustomerAgent:
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("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")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def build_choice_prompt(self, profile: Dict[str, Any], scenario: Dict[str, Any], offer_a: float, offer_b: float) -> str:
       return f""" You are a ride-hailing customer deciding between two platforms for ONE trip.
       
       You profile is as follows:
            - Age: {profile["Age"]}
            - Income Bracket: {profile["IncomeBracket"]}
            - Household Size: {profile["HouseholdSize"]}
            - Marital Status: {profile["MaritalStatus"]}
            - Employment: {profile["EmploymentStatus"]}
            - Gender: {profile["Gender"]}
            - Membership/loyalty: {profile["Loyalty"]}

        Context of your trip includes:
            - City: {scenario["City"]}
            - Distance (miles): {scenario["DistanceMiles"]}
            - Duration (minutes): {scenario["DurationMinutes"]}
            - Day of week (0=Mon): {scenario["DayOfWeek"]}
            - Hour of day: {scenario["Hour"]}
            - Weather: {scenario["Weather"]}
            - Airport trip: {scenario["Airport"]}
            - Service tier: {scenario["Service"]}
        
        You are seeing two TOTAL prices offered by separate platforms for the SAME trip:
            - Firm 1 price: ${offer_a}
            - Firm 2 price: ${offer_b}
            
        Task: 
            Choose ONE firm. Consider not only price but also your profile and the context (e.g., airport urgency, bad weather discomfort, loyalty).
            
        Return STRICT JSON ONLY (no extra text) with:
        {{
            "choice": "Firm1" or "Firm2",
            "reason_codes": ["PRICE", "LOYALTY", "URGENCY", "WEATHER", "COMFORT", "HABIT", "RISK"],
            "short_reason": "one sentence explanation"
        }}
        """.strip()
       
    def get_agent_response(self, prompt: str) -> str:
        if self.client is None:
            return json.dumps({"choice": "Firm1", "reason_codes": ["PRICE"], "short_reason": "Fallback choice."})
        
        resp = self.client.chat.completions.create(
            model = self.model_name,
            messages = [{"role": "user", "content": prompt}],
            temperature = 0.4,
            max_tokens = 200,
        )
        
        return resp.choices[0].message.content.strip()
        
    def parse_choice(self, response: str) -> Dict[str, Any]:
        try: 
            obj = json.loads(response)
            if obj.get("choice") not in ("Firm1", "Firm2"):
                raise ValueError("Invalid choice field.")
            if not isinstance(obj.get("reason_codes", []), list):
                obj["reason_codes"] = []
            if "short_reason" not in obj:
                obj["short_reason"] = ""
                
            return obj
        except Exception:
            return {"choice": "Firm1", "reason_codes": ["PARSE_FAIL"], "short_reason": response[:120]}
        
        