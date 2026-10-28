"""Regime channels in the rider choice model (neutral defaults leave v26 untouched)."""
import numpy as np
from choice_models import CognitiveChoiceModel


PROFILE = {"IncomeBracket": "50k-100k", "HouseholdSize": 2, "PriceThreshold": 1.5, "LoyaltyType": "New"}
BASE = {"City": "Boston", "DistanceMiles": 4.0, "DurationMinutes": 15.0, "DayOfWeek": 2, "Hour": 12,
        "Weather": "clear", "Airport": False, "Service": "economy",
        "WaitEstimateFirm1": 5.0, "WaitEstimateFirm2": 5.0, "CancelRiskFirm1": 0.0, "CancelRiskFirm2": 0.0}


def _no_ride_rate(model, scenario, p1=22.0, p2=23.0, n=600):
    model.rng = np.random.default_rng(0)
    return float(np.mean([model.choose(PROFILE, scenario, p1, p2).choice == "NoRide" for _ in range(n)]))


def test_defaults_are_neutral_and_transit_loss_lowers_no_ride():
    model = CognitiveChoiceModel(seed=0)
    base = _no_ride_rate(model, BASE)
    neutral = _no_ride_rate(model, {**BASE, "ShockReservationMultiplier": 1.0,
                                    "ShockPriceSensitivityMultiplier": 1.0, "ShockTransitAvailability": 1.0})
    assert abs(base - neutral) < 1e-9
    transit_gone = _no_ride_rate(model, {**BASE, "ShockTransitAvailability": 0.0})
    assert transit_gone < base


def test_lower_price_sensitivity_raises_reservation_and_demand():
    model = CognitiveChoiceModel(seed=0)
    hi_price = _no_ride_rate(model, BASE, p1=40.0, p2=41.0)
    inelastic = _no_ride_rate(model, {**BASE, "ShockPriceSensitivityMultiplier": 0.6,
                                      "ShockReservationMultiplier": 1.3}, p1=40.0, p2=41.0)
    assert inelastic < hi_price
