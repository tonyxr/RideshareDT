"""Rider loyalty v2: per-rider stocks moved by rides and stranding, used by the cognitive choice model."""
import numpy as np
from choice_models import CognitiveChoiceModel
from Core import Core as RideshareCore

PROFILE = {"IncomeBracket": "50k-100k", "HouseholdSize": 2, "PriceThreshold": 1.5, "LoyaltyType": "Returning",
           "LoyaltyFirm": "Firm2", "LoyaltyStrength": 0.6}
SCENARIO = {"City": "Boston", "DistanceMiles": 4.0, "DurationMinutes": 15.0, "DayOfWeek": 2, "Hour": 12,
            "Weather": "clear", "Airport": False, "Service": "economy",
            "WaitEstimateFirm1": 5.0, "WaitEstimateFirm2": 5.0, "CancelRiskFirm1": 0.0, "CancelRiskFirm2": 0.0}


def _firm1_rate(model, profile, p1=22.0, p2=22.0, n=800):
    model.rng = np.random.default_rng(0)
    return float(np.mean([model.choose(profile, SCENARIO, p1, p2).choice == "Firm1" for _ in range(n)]))


def test_choice_follows_loyalty_stock_and_v1_profile_untouched():
    model = CognitiveChoiceModel(seed=0)
    trait = _firm1_rate(model, PROFILE)                                   # fixed trait: loyal to Firm2
    loyal1 = _firm1_rate(model, {**PROFILE, "_LoyaltyV2": {"Firm1": 0.9, "Firm2": 0.0}})
    loyal2 = _firm1_rate(model, {**PROFILE, "_LoyaltyV2": {"Firm1": 0.0, "Firm2": 0.9}})
    neutral = _firm1_rate(model, {**PROFILE, "_LoyaltyV2": {"Firm1": 0.0, "Firm2": 0.0}})
    assert loyal1 > neutral > loyal2
    assert abs(trait - _firm1_rate(model, dict(PROFILE))) < 1e-12          # no stock key -> v26 behaviour
    model.loyalty_v2_weight = 3.0
    assert _firm1_rate(model, {**PROFILE, "_LoyaltyV2": {"Firm1": 0.9, "Firm2": 0.0}}) > loyal1


class _Stub:
    """Just enough of RideshareCore for the update rule."""
    loyalty_lambda = 0.3
    loyalty_overcharge_penalty = False
    _loyalty_record_outcome = RideshareCore._loyalty_record_outcome


def _update(stock, **kw):
    profile = {"_LoyaltyV2": stock}
    base = dict(choice="Firm1", matched_firm="Firm1", fulfilled=True, pickup_minutes=4.0, wait_estimate=5.0,
                paid=22.0, other_price=22.0, threshold=1.5)
    base.update(kw)
    _Stub()._loyalty_record_outcome(profile, **base)
    return stock


def test_update_rule_good_ride_strand_and_elsewhere():
    good = _update({"Firm1": 0.2, "Firm2": 0.2})
    assert good["Firm1"] > 0.2 and good["Firm2"] < 0.2 and good["rides_Firm1"] == 1.0
    stranded = _update({"Firm1": 0.2, "Firm2": 0.2}, matched_firm="", fulfilled=False)
    assert stranded["Firm1"] < 0.2 - 1e-9
    rescued = _update({"Firm1": 0.2, "Firm2": 0.2}, matched_firm="Firm2")
    assert rescued["Firm1"] < 0.2 and rescued["Firm2"] > 0.2
    slow = _update({"Firm1": 0.2, "Firm2": 0.2}, pickup_minutes=30.0)
    assert abs(slow["Firm1"] - 0.14) < 1e-9                               # x = 0: decays
    elsewhere = _update({"Firm1": 0.2, "Firm2": 0.2}, choice="Firm2", matched_firm="Firm2")
    assert elsewhere["Firm1"] < 0.2 and elsewhere["Firm2"] > 0.2
    stub = _Stub(); stub.loyalty_overcharge_penalty = True
    profile = {"_LoyaltyV2": {"Firm1": 0.2, "Firm2": 0.2}}
    stub._loyalty_record_outcome(profile, choice="Firm1", matched_firm="Firm1", fulfilled=True, pickup_minutes=4.0,
                                 wait_estimate=5.0, paid=30.0, other_price=22.0, threshold=1.5)
    assert profile["_LoyaltyV2"]["Firm1"] < 0.2                            # overcharged (opt-in penalty)


def test_lambda_zero_keeps_trait_initialisation():
    stock = _update({"Firm1": 0.0, "Firm2": 0.6}, )
    _Stub.loyalty_lambda = 0.0
    try:
        frozen = _update({"Firm1": 0.0, "Firm2": 0.6})
        assert frozen["Firm1"] == 0.0 and frozen["Firm2"] == 0.6
    finally:
        _Stub.loyalty_lambda = 0.3
    assert stock["Firm1"] > 0.0
