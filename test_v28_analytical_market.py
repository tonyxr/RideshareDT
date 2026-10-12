import math
import unittest

import numpy as np

from analytical_market import (
    ExponentialMarketRegime,
    LaggedDriverSupply,
    best_anticipatory_incentive,
    price_tail,
)


class AnalyticalKnownAnswerTests(unittest.TestCase):
    def test_volume_only_shock_does_not_move_unconstrained_markup(self):
        low = ExponentialMarketRegime(1_000.0, 0.12, 3.0)
        high = ExponentialMarketRegime(4_000.0, 0.12, 3.0)
        self.assertAlmostEqual(low.optimal_price, high.optimal_price)
        self.assertAlmostEqual(low.optimal_price - 3.0, 1.0 / 0.12)

    def test_elasticity_and_capacity_shocks_move_the_optimum(self):
        normal = ExponentialMarketRegime(1_000.0, 0.12, 3.0)
        inelastic = ExponentialMarketRegime(1_000.0, 0.06, 3.0)
        scarce = ExponentialMarketRegime(1_000.0, 0.12, 3.0, capacity=80.0)
        self.assertGreater(inelastic.optimal_price, normal.optimal_price)
        self.assertGreater(scarce.optimal_price, normal.optimal_price)
        self.assertAlmostEqual(scarce.demand(scarce.capacity_clearing_price), 80.0)

    def test_uncapped_price_tail_is_economically_well_posed(self):
        regime = ExponentialMarketRegime(2_000.0, 0.10, 4.0)
        rows = price_tail(regime, [10.0, 20.0, 40.0, 80.0, 160.0])
        self.assertTrue(np.all(np.diff(rows[:, 1]) < 0.0))
        self.assertLess(rows[-1, 1], 1e-3)
        self.assertLess(rows[-1, 2], rows[-2, 2])
        self.assertTrue(math.isfinite(regime.optimal_price))

    def test_lagged_supply_can_make_pre_shortage_incentive_pay(self):
        supply = LaggedDriverSupply(
            adjustment_rate=0.55,
            incentive_elasticity=0.9,
            incentive_cost_per_driver=0.05,
        )
        common = dict(
            current_capacity=55.0,
            normal_capacity=100.0,
            shortage_capacity_multiplier=0.55,
            next_demand_scale=4_000.0,
            next_elasticity=0.08,
            marginal_cost=3.0,
            supply=supply,
        )
        best = best_anticipatory_incentive([1.0, 1.1, 1.25, 1.5, 2.0], **common)
        no_incentive = best_anticipatory_incentive([1.0], **common)
        self.assertGreater(best["incentive_multiplier"], 1.0)
        self.assertGreater(best["next_capacity"], no_incentive["next_capacity"])
        self.assertGreater(best["net_value"], no_incentive["net_value"])


if __name__ == "__main__":
    unittest.main()
