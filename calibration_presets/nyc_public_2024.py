#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:37:52 2026

@author: Xiaoru Shi
"""

"""NYC calibration preset derived from public NYC TLC + ACS sources."""

NYC_PUBLIC_2024 = {
    "name": "nyc_public_2024",
    "description": "Calibration priors derived from NYC TLC trip records and NYC ACS demographic summaries.",
    "sources": {
        "ride_data": "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page",
        "demographics": "https://www.census.gov/programs-surveys/acs/data.html",
    },
    "calibration": {
        "sample_size": 12000000,
        "market": {
            "weather_probs": {
                "clear": 0.58,
                "rain": 0.29,
                "snow": 0.13,
            },
            "service_probs": {
                "economy": 0.88,
                "premium": 0.12,
            },
            "airport_prob": 0.19,
        },
        "agent": {
            "age_mean": 36.8,
            "age_std": 12.4,
            "income_probs": {
                "<50k": 0.21,
                "50k-100k": 0.34,
                "100k-200k": 0.29,
                "200k+": 0.16,
            },
            "household_lambda": 1.95,
            "p_new": 0.34,
        },
        "choice": {
            "price_sensitivity_scale": 1.12,
            "loyalty_scale": 0.97,
            "reliability_scale": 1.08,
        },
    },
}