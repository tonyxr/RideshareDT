"""Regressions for final-policy provenance and live benchmark telemetry."""
import pytest
import torch
from collections import Counter
from types import SimpleNamespace
from Core import Core
from platform_mdp import LongTermProfitRewardConfig, LongTermProfitReward, TrainingStageScheduler
from pricing_models import (
    FirmMetrics, FirmSurgeDriverIncentivePricer, FirmMPCGridPricer,
    FirmAggressiveAdaptiveBestResponsePricer,
)


@pytest.mark.parametrize("controller", [FirmSurgeDriverIncentivePricer, FirmMPCGridPricer])
def test_benchmark_supply_stress_refreshes_before_act_and_recovers(controller):
    p = controller(seed=123)
    metrics = FirmMetrics(revenue=100, total=100, chosen=90, completed=10, wins=10)
    shortage = dict(fulfillment_rate=.1, acceptance_rate=.1, avg_wait_minutes=30,
                    avg_pickup_minutes=30, driver_earnings_per_hour=1,
                    idle_driver_share=0, utilization=1)
    for _ in range(20):
        p.update(metrics, 0., supply_state=shortage)
    assert p.last_supply_stress > .9
    p.act(2.5, .25, 8, 'rain', city_pmile=1., city_booking=2., city_airport=4.)
    if isinstance(p, FirmSurgeDriverIncentivePricer):
        assert p.supply_incentive_multiplier == pytest.approx(1.025)
    recovery = dict(fulfillment_rate=1, acceptance_rate=1, avg_wait_minutes=0,
                    avg_pickup_minutes=0, driver_earnings_per_hour=30,
                    idle_driver_share=.6, utilization=.2)
    recovered = FirmMetrics(revenue=100, total=100, chosen=50, completed=50, wins=50)
    for _ in range(80):
        p.update(recovered, 0., supply_state=recovery)
    assert p.last_supply_stress < .001


@pytest.mark.parametrize("eligibility", [True, False, None])
def test_history_cannot_override_latest_even_with_higher_reward(eligibility):
    history = dict(day=1380, score=100, validation=dict(checkpoint_eligible=True))
    latest = dict(day=3000, score=-1, validation=dict(checkpoint_eligible=eligibility))
    selected, _ = Core._evaluation_checkpoint(history, latest, history)
    assert selected is latest


def test_missing_latest_never_substitutes_history():
    history = dict(day=1380, score=100)
    selected, status = Core._evaluation_checkpoint(history, None, history)
    assert selected is None
    assert status == 'final_actor'


def test_legacy_rival_keeps_act_time_diagnostic_cadence():
    p = FirmAggressiveAdaptiveBestResponsePricer(seed=1)
    p.last_supply_stress = .123
    p.last_demand_shortfall = -.321
    p.update(FirmMetrics(total=100, chosen=90, completed=10), 0.,
             supply_state=dict(fulfillment_rate=0., acceptance_rate=0., avg_wait_minutes=50.))
    assert p.last_supply_stress == .123
    assert p.last_demand_shortfall == -.321


@pytest.mark.parametrize('mode', [None, 'deployed', 'log_only', 'typo'])
def test_policy_projection_cannot_be_enabled_or_selected_by_default(mode):
    core = Core.__new__(Core)
    with pytest.raises(ValueError, match='guardrails are disabled'):
        core._apply_eval_guardrail(mode, metrics=None, price_gap_f2_minus_f1=0, base=None)


def test_final_snapshot_uses_live_weights_and_copies_mutable_state():
    net = torch.nn.Linear(2, 2)
    normalizer = {"count": [7]}
    optimizer = {"step": [9]}
    core = Core.__new__(Core)
    core.current_training_stage = SimpleNamespace(name="dynamic_opponent")
    core.firm1 = SimpleNamespace(agent=SimpleNamespace(
        net=net, normalizer_state_dict=lambda: normalizer,
        optimizer_state_dict=lambda: optimizer))
    earlier = dict(day=2980, score=100, validation={"checkpoint_eligible": True})
    final = core._capture_final_training_checkpoint(3000, earlier)
    assert final['day'] == 3000
    assert final['validation'] == {}
    assert torch.equal(final['agent_state']['weight'], net.weight)
    normalizer['count'][0] = 99
    optimizer['step'][0] = 99
    with torch.no_grad():
        net.weight.add_(1)
    assert final['normalizer_state']['count'] == [7]
    assert final['optimizer_state']['step'] == [9]
    assert not torch.equal(final['agent_state']['weight'], net.weight)


@pytest.mark.parametrize("stage,weights", [("market", (.95, 0., .05)),
                                         ("static", (.475, .475, .05)),
                                         ("dynamic", (.475, .475, .05)),
                                         (None, (.475, .475, .05))])
def test_stage_reward_exact_decomposition(stage, weights):
    model = LongTermProfitReward(LongTermProfitRewardConfig(stage_specific_rewards=True))
    result = model.compute(own_profit_per_request=4., rival_profit_per_request=3.,
                           own_completed_share=.55, rival_completed_share=.45,
                           training_regime=stage)
    assert result['reward'] == pytest.approx(
        weights[0] * 4 / 2 + weights[1] * (4 - 3) + weights[2]
    )
    assert result['reward_weight_own_profit'] == weights[0]
    assert result['reward_weight_profit_lead'] == weights[1]
    assert result['reward_weight_completed_share_lead'] == weights[2]
    assert result['reward_active_component_count'] == (2 if stage == 'market' else 3)


def test_stage_reward_is_unclipped_monotone_and_market_ignores_rival():
    model = LongTermProfitReward(LongTermProfitRewardConfig(stage_specific_rewards=True))
    def reward(p, rival, stage):
        return model.compute(own_profit_per_request=p, rival_profit_per_request=rival,
                             training_regime=stage)['reward']
    assert reward(10, 100, 'market') == reward(10, -100, 'market') == 4.75
    assert reward(10, 100, 'dynamic') < 0
    assert reward(20, 100, 'dynamic') > reward(10, 100, 'dynamic')
    assert reward(10, 1, 'dynamic') > reward(10, 2, 'dynamic')
    assert reward(100, 0, 'dynamic') > 1


def test_flags_off_reward_exact_identity_across_stages():
    model = LongTermProfitReward(LongTermProfitRewardConfig())
    kwargs = dict(own_profit_per_request=4., rival_profit_per_request=3.)
    assert all(model.compute(training_regime=s, **kwargs) == model.compute(**kwargs)
               for s in ('market', 'static', 'dynamic'))


def test_evaluation_reward_cannot_inherit_current_training_stage():
    from test_core_ppo_fixes import CorePPOFixRegressionTests
    core = CorePPOFixRegressionTests._diagnostic_core()
    core.long_term_profit_reward_config = LongTermProfitRewardConfig(stage_specific_rewards=True)
    core.long_term_profit_reward_model = LongTermProfitReward(core.long_term_profit_reward_config)
    core.current_training_stage = SimpleNamespace(opponent_regime='market')
    kwargs = dict(share=.5, baseline_share=.5, rev_per_request=8, mean_gap=0, prev_share=.5,
                  prev_rev_per_request=8, profit_per_request=4, baseline_profit_per_request=3)
    train = core._reward_diagnostics(**kwargs, training_regime='market')
    evaluation = core._reward_diagnostics(**kwargs)
    assert train['reward'] == pytest.approx(1.9)
    assert evaluation['reward'] == pytest.approx(1.425)


def test_10000_day_stage_allocation_has_complete_opponent_batches():
    scheduler = TrainingStageScheduler(stage_fractions=(.1, .108, .792), dynamic_episode_days=60)
    counts = Counter(scheduler.stage_at(d / 10000).opponent_regime for d in range(10000))
    assert counts == dict(market=1000, static=1080, dynamic=7920)
    assert counts['dynamic'] % 180 == 0
