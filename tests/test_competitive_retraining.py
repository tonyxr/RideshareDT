"""Reward arithmetic and retraining configuration; no performance thresholds."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from Core import Core
from platform_mdp import LongTermProfitRewardConfig, PolicyObjectiveReward
from scripts.run_v26r_staged import command


def test_competitive_weight_changes_only_opponent_reward():
    default = LongTermProfitRewardConfig(stage_specific_rewards=True, learning_corrections=True,
                                        own_profit_scale=2, profit_advantage_scale=2)
    competitive = replace(default, opponent_profit_lead_weight=.6)
    for own, rival in ((2, 1), (1, 2), (-1, -2), (-2, 1), (100, -100)):
        args = dict(own_profit_per_request=own, rival_profit_per_request=rival)
        reward = PolicyObjectiveReward(competitive)
        market = reward.compute(training_regime='market', **args)
        assert market == PolicyObjectiveReward(default).compute(training_regime='market', **args)
        for stage in ('static', 'dynamic', None):
            result = reward.compute(training_regime=stage, **args)
            assert result['reward_base'] == pytest.approx(
                .95 * (.4*own + .6*(own-rival)) / 2
            )
            assert reward.compute(training_regime=stage, own_profit_per_request=own+1,
                                  rival_profit_per_request=rival)['reward_base'] > result['reward_base']
            assert reward.compute(training_regime=stage, own_profit_per_request=own,
                                  rival_profit_per_request=rival+1)['reward_base'] < result['reward_base']
        core = SimpleNamespace(long_term_profit_reward_config=competitive)
        scores, _ = Core._objective_firm_scores(core, [own], [rival], [0], [0])
        np.testing.assert_allclose(scores, [result['reward_base']])


@pytest.mark.parametrize('weight', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_weight_is_configuration_error_not_silent_clipping(weight):
    with pytest.raises(ValueError):
        LongTermProfitRewardConfig(stage_specific_rewards=True, opponent_profit_lead_weight=weight)


@pytest.mark.parametrize('city,slug', [('Chicago','chicago'), ('Boston','boston'), ('New York City','nyc')])
def test_retraining_changes_only_requested_command_options(tmp_path, city, slug):
    before = command(tmp_path, days=20000, city=city, slug=slug)
    after = command(tmp_path, days=20000, city=city, slug=slug,
                    update_days=1, rollout_transitions=30, lead_weight=.6)
    expected = {'--ppo_update_interval_days':'1', '--ppo_min_rollout_transitions':'30',
                '--opponent_profit_lead_weight':'0.6'}
    for flag, value in expected.items():
        i = before.index(flag)
        assert after[i+1] == value
        before[i+1] = value
    assert after == before


def test_city_worker_uses_requested_settings_without_restarting_corners(tmp_path, monkeypatch):
    from scripts import run_v26r_multicity as runner
    calls = []
    def watch(cmd, out, label):
        calls.append((cmd, label))
        (out / 'policy.final_training.pt').write_bytes(b'test checkpoint')
    monkeypatch.setattr(runner, 'watch', watch)
    monkeypatch.setattr(runner, 'training_evidence', lambda *a: {'days': 20000})
    monkeypatch.setattr(runner, 'summarize', lambda *a, **kw: {})
    args = SimpleNamespace(city='boston', days=20000, smoke=False, update_days=1,
                           rollout_transitions=30, lead_weight=.6, skip_corners=True)
    runner.city_job(args, tmp_path)
    assert len(calls) == 1 and calls[0][1] == 'training'
    cmd = calls[0][0]
    for flag, value in {'--ppo_update_interval_days':'1', '--ppo_min_rollout_transitions':'30',
                        '--opponent_profit_lead_weight':'0.6', '--market':'Boston'}.items():
        assert cmd[cmd.index(flag)+1] == value
    assert '--learning_corrections' in cmd
    assert json.loads((tmp_path/'status.json').read_text())['corner_evaluation'] == 'not requested'
