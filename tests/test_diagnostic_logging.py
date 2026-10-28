import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from diagnostic_logging import DiagnosticJournal, atomic_json


def row(day=0, update=False):
    return dict(batch=day, training_stage="dynamic_opponent", training_opponent="memory",
                training_episode=1, avg_reward=.2, raw_policy_entropy=1.3,
                behavior_exploration_rate=.05, ppo_exploration_rate=.175,
                avg_profit_advantage_per_request=-.1, ppo_update_performed=update,
                ppo_approx_kl=.004 if update else float("nan"),
                ppo_explained_variance=.5 if update else float("nan"),
                ppo_value_loss=1.1 if update else float("nan"))


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_daily_and_update_records_durable_and_schema_can_grow(tmp_path):
    prefix = tmp_path/"run"
    journal = DiagnosticJournal(prefix, plots=False)
    journal.training_day(row())
    assert read(tmp_path/"run_training_live.jsonl")[0]["ppo_approx_kl"] is None
    second = row(1, True)
    second["validation_new_field"] = np.float32(.4)
    journal.training_day(second, episode_end=True)
    assert len(read(tmp_path/"run_training_live.jsonl")) == 2
    assert len(read(tmp_path/"run_ppo_updates.jsonl")) == 1
    episode = read(tmp_path/"run_episodes.jsonl")[0]
    assert episode["days"] == 2 and episode["ppo_updates"] == 1
    assert episode["entropy_mean"] == 1.3
    assert episode["critic_ev_mean"] == .5
    assert episode["exploration_mean"] == .05  # not the PPO batch-average .175
    journal.finish_episode()
    assert len(read(tmp_path/"run_episodes.jsonl")) == 1


def test_restart_cannot_mix_experiments(tmp_path):
    DiagnosticJournal(tmp_path/"run", plots=False)
    with pytest.raises(FileExistsError):
        DiagnosticJournal(tmp_path/"run", plots=False)


def test_graphs_do_not_change_rng_or_input(tmp_path):
    journal = DiagnosticJournal(tmp_path/"run")
    sample = row(0, True)
    before = dict(sample)
    state = np.random.get_state()
    py_state = random.getstate()
    journal.training_day(sample, episode_end=True)
    assert sample == before
    assert random.getstate() == py_state
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    assert (tmp_path/"run_episode_diagnostics.png").stat().st_size > 1000


def test_completed_records_survive_abrupt_process_exit(tmp_path):
    # Separate logging-only child, deliberately exits without cleanup.
    code = ("from diagnostic_logging import DiagnosticJournal; import os; "
            f"j=DiagnosticJournal({str(tmp_path/'run')!r}, plots=False); "
            "j.append('training_live', {'batch': 1, 'raw_policy_entropy': 1.2}); os._exit(7)")
    result = subprocess.run([sys.executable, "-c", code],
                            cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 7
    assert read(tmp_path/"run_training_live.jsonl")[0]["raw_policy_entropy"] == 1.2


def test_atomic_reports_replace_only_completed_json(tmp_path):
    path = tmp_path/"report.json"
    atomic_json(path, {"value": np.array([1, float("nan")])})
    assert json.loads(path.read_text()) == {"value": [1.0, None]}
    assert not path.with_name(path.name+".tmp").exists()


def test_recovery_command_cannot_retrain_or_overwrite_checkpoint(tmp_path):
    from scripts.resume_v26r_evaluation import evaluation_command
    from scripts.run_v26r_staged import command
    source = tmp_path/"source"
    source.mkdir()
    original = command(source)
    (source/"command.json").write_text(json.dumps(original))
    resumed = evaluation_command(source, tmp_path/"recovery")
    def value(cmd, key):
        return cmd[cmd.index(key)+1]
    assert "--eval_only" in resumed
    assert value(resumed, "--train_timesteps") == "0"
    assert "--trained_model_out" not in resumed
    assert value(resumed, "--model_registry_dir") == ""
    assert value(resumed, "--trained_model_in") == str(source/"policy.final_training.pt")
    for key in ["--eval_timesteps", "--eval_customers", "--checkpoint_validation_customers",
                "--seed", "--pool", "--threshold_cache_path", "--calibration_json",
                "--firm2_mode", "--train_steps_per_day", "--static_baseline_multipliers"]:
        assert value(resumed, key) == value(original, key)
