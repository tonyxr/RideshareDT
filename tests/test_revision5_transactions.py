"""Transactions must preserve both rider values and external stock aliases."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
from Core import Core


def riders():
    core = Core.__new__(Core)
    stock = dict(Firm1=.2, Firm2=.4, rides_Firm1=2., rides_Firm2=3.)
    profile = {"_RiderID": 0, "_LoyaltyV2": stock}
    core.rider_loyalty = {0: stock}
    core._active_rider_pool = [profile]
    return core, profile, stock


def test_rollback_restores_aliases_after_branch_and_validation_pool_replacement():
    core, profile, stock = riders()
    outside = {"_LoyaltyV2": {"Firm1": .7, "Firm2": .1}}
    snapshot = core._snapshot_rider_state([profile, profile, outside])
    before = copy.deepcopy(stock)
    for _ in range(3):
        stock["Firm1"] += .5
        stock["rides_Firm1"] += 1
        outside["_LoyaltyV2"]["Firm1"] = -1
        core.rider_loyalty = {99: {"Firm1": 1}}
        core._active_rider_pool = [{"_RiderID": 99}]
        core._restore_rider_state(snapshot)
        assert core.rider_loyalty[0] is profile["_LoyaltyV2"] is stock
        assert core._active_rider_pool[0] is profile
        assert stock == before
        assert outside["_LoyaltyV2"]["Firm1"] == .7


@pytest.mark.parametrize('evaluation', [False, True])
def test_returning_riders_use_the_requested_disjoint_partition(evaluation):
    core, _, _ = riders()
    core.synthetic_profile_pool = [{"name": "unused"}]
    core.training_profile_pool = [{"name": "train", "LoyaltyStrength": .4}]
    core.evaluation_profile_pool = [{"name": "heldout", "LoyaltyStrength": .7}]
    core.loyalty_pool_size = 4
    core._build_active_rider_pool(np.random.default_rng(1), evaluation=evaluation)
    assert core._active_rider_pool[0]['name'] == ('heldout' if evaluation else 'train')


@pytest.mark.parametrize('fail', [False, True])
def test_counterfactual_rollbacks_real_shared_stocks_and_exceptions(fail):
    core, profile, stock = riders()
    before = copy.deepcopy(stock)
    core.counterfactual_branches = 3
    core.counterfactual_action_counts = np.zeros(3, dtype=int)
    core.firm1 = SimpleNamespace(agent=SimpleNamespace(action_dim=3),
        apply_action=lambda action, market: None, last_action_magnitude=lambda: 0)
    core.market = None
    # Exercise the actual branch loop; lightweight doubles isolate state plumbing.
    core._snapshot_counterfactual_state = lambda profiles=(): core._snapshot_rider_state(profiles)
    core._restore_counterfactual_state = core._restore_rider_state
    metrics = SimpleNamespace(**{k: 0. for k in ('chosen_share', 'completed_share', 'rev_per_request',
        'profit_per_request', 'fulfillment_rate', 'avg_wait_minutes', 'driver_acceptance_rate')})
    seen = []
    def simulate(**kwargs):
        seen.append(copy.deepcopy(stock))
        stock['Firm1'] += .3
        stock['rides_Firm1'] += 1
        if fail:
            raise RuntimeError('branch failure')
        return None, metrics, metrics, 0, None, None
    core.simulate_batch = simulate
    for name in ('last_share', 'last_revpr', 'last_profitpr', 'last_firm2_profitpr',
                 'last_firm2_share', 'last_fulfillment', 'last_gap'):
        setattr(core, name, 0.)
    core.last_price_gap_metrics = {'mae': 0.}
    core._profit_margin = lambda metrics: 0.
    core._reward_diagnostics = lambda **kwargs: {'reward': stock['Firm1']}
    def run():
        return core._counterfactual_action_values(selected_action=0, policy_top_action=0,
            policy_second_action=1, action_mask=np.ones(3, bool), day_of_week=0, hour=0,
            weather='clear', customers_per_step=1, sampled_profiles=[profile])
    if fail:
        with pytest.raises(RuntimeError, match='branch failure'):
            run()
    else:
        values, mask = run()
        assert mask.all()
        np.testing.assert_array_equal(values, 0.)
        assert len(seen) == 3
    assert all(value == before for value in seen)
    assert stock == before


def test_reward_objective_is_not_reinterpreted():
    from platform_mdp import LongTermProfitReward, LongTermProfitRewardConfig
    reward = LongTermProfitReward(LongTermProfitRewardConfig(stage_specific_rewards=True,
        learning_corrections=True, opponent_profit_lead_weight=.6))
    for own, rival in [(4., 3.), (-2., 5.), (100., -20.)]:
        for stage in ('static', 'dynamic', None):
            assert reward.compute(own_profit_per_request=own, rival_profit_per_request=rival,
                training_regime=stage)['reward'] == pytest.approx(.475*own-.285*rival)
