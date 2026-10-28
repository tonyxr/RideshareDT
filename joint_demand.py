"""Joint, heterogeneous logit demand; parameters are not selected using RL.

At the fixed city-anchor quote, retain the existing contextual/loyalty choice
probabilities as *priors*. Away from it, all three alternatives share one
normalizer. GPT thresholds supply relative heterogeneity, not an elasticity.
This does not identify outside-option levels or substitution from trip data.
"""
from __future__ import annotations
import copy
import numpy as np
from choice_models import CognitiveChoiceModel, ChoiceResult

CONTRACT = 'revision7-joint-contextual-logit-v1'
ALTERNATIVES = ('Firm1', 'Firm2', 'NoRide')


def positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def validate_spec(spec, city):
    if spec.get('format') != CONTRACT or spec.get('city') != city:
        raise ValueError('demand calibration format/city mismatch')
    for key in ('typical_fare', 'threshold_reference', 'beta'):
        if not isinstance(spec[key], (int, float)) or isinstance(spec[key], bool):
            raise ValueError(f'{key} must be numeric')
        positive(spec[key], key)
    if not spec.get('evidence'):
        raise ValueError('demand calibration must declare its evidence and limitations')


def softmax(logits):
    values = np.asarray(logits, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite demand utility')
    weights = np.exp(values - np.max(values, axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


class JointChoiceModel(CognitiveChoiceModel):
    """U_i = log(anchor probability_i) - beta*h*(price_i-anchor)/city fare.

    NoRide's utility is unchanged. All quotes affect participation and platform
    substitution jointly. There is no price clipping, action constraint, or
    fallback to the old demand response. Conditional IIA is an explicit model
    restriction; heterogeneous profiles yield a population mixture of logits.
    """
    @classmethod
    def from_legacy(cls, legacy, spec):
        validate_spec(spec, spec['city'])
        if isinstance(legacy, cls):
            raise ValueError('joint demand is already installed')
        if not isinstance(legacy, CognitiveChoiceModel):
            raise ValueError('joint demand requires the cognitive profile model')
        model = cls()
        model.__dict__.update(copy.deepcopy(legacy.__dict__))
        model.demand_spec = copy.deepcopy(spec)
        return model

    def components(self, profile, scenario):
        """Fixed-anchor prior plus marginal price coefficient (no RNG/state writes)."""
        reference = positive(scenario['ReferenceFare'], 'ReferenceFare')
        threshold = positive(profile['PriceThreshold'], 'PriceThreshold')
        # Use the unbound legacy method only at an exogenous equal-price anchor.
        # The posted quotes never enter its old min-price gate or salience term.
        prior = CognitiveChoiceModel.choose(self, profile, scenario, reference, reference,
                                            _probabilities=True)
        probabilities = np.array([prior[key] for key in ALTERNATIVES])
        if not np.isfinite(probabilities).all() or np.any(probabilities <= 0):
            raise ValueError('anchor prior must give positive finite choice probabilities')
        spec = self.demand_spec
        heterogeneity = spec['threshold_reference'] / threshold
        shock = positive(scenario.get('ShockPriceSensitivityMultiplier', 1.), 'shock sensitivity')
        if scenario.get('Airport', False):
            shock *= positive(scenario.get('ShockAirportPriceSensitivityMultiplier', 1.), 'airport sensitivity')
        return np.log(probabilities), heterogeneity * shock / spec['typical_fare'], reference

    def probabilities(self, profile, scenario, price1, price2):
        prices = np.asarray([price1, price2], dtype=float)
        if not np.isfinite(prices).all() or np.any(prices < 0):
            raise ValueError('posted fares must be finite and nonnegative')
        logits, slope, reference = self.components(profile, scenario)
        logits[:2] -= self.demand_spec['beta'] * slope * (prices-reference)
        return dict(zip(ALTERNATIVES, map(float, softmax(logits))))

    def choose(self, profile, scenario, price1, price2, *, _probabilities=False):
        probabilities = self.probabilities(profile, scenario, price1, price2)
        if _probabilities:
            return probabilities
        draw = float(self.rng.random())
        p1, p2 = probabilities['Firm1'], probabilities['Firm2']
        choice = 'Firm1' if draw < p1 else 'Firm2' if draw < p1+p2 else 'NoRide'
        return ChoiceResult(choice, ['JOINT_UTILITY'], 'Joint contextual choice among both firms and the outside option.')

    def retry_probability(self, profile, scenario, other_name, other_price):
        """Same utilities with the unavailable platform removed, not a new gate."""
        index = ALTERNATIVES.index(other_name)
        if index == 2 or not np.isfinite(other_price) or other_price < 0:
            raise ValueError('retry requires an available firm and nonnegative finite fare')
        logits, slope, reference = self.components(profile, scenario)
        utilities = [logits[index] - self.demand_spec['beta'] * slope * (other_price-reference), logits[2]]
        return float(softmax(utilities)[0])


def fit_local_own_elasticity(model, records, target):
    """Exact local log derivative at parity; no rewards, supply, or policy input.

    records: (profile, scenario, exogenous reference fare). The result matches
    the stated moment mathematically, NOT an independent empirical validation.
    """
    target = float(target)
    if not np.isfinite(target) or target >= 0 or not records:
        raise ValueError('need a negative finite elasticity target and calibration records')
    levels, derivatives = [], []
    for profile, scenario, fare in records:
        logits, slope, reference = model.components(profile, scenario)
        if float(fare) != reference:
            raise ValueError('calibration records must use their exogenous reference fare')
        p = softmax(logits)[0]
        levels.append(p)
        derivatives.append(slope * fare * p * (1-p))
    response = float(np.mean(derivatives) / np.mean(levels))
    return positive(-target / positive(response, 'local response'), 'fitted beta')
