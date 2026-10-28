from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_revision15d_12h_full_counterfactual as dedicated  # noqa: E402
import run_revision15d_multicity_full as full  # noqa: E402
from platform_mdp import TrainingStageScheduler  # noqa: E402


def test_dedicated_12h_command_uses_complete_actor_credit(tmp_path):
    counts = dedicated.DEFAULT_STAGE_DAYS
    fractions = dedicated.stage_fractions(dedicated.DEFAULT_DAYS, counts)
    for slug in dedicated.CITIES:
        command = dedicated.build_training_command(
            slug,
            tmp_path / slug,
            dedicated.DEFAULT_DAYS,
            251026,
            fractions,
        )
        expected = {
            "--train_timesteps": str(dedicated.DEFAULT_DAYS),
            "--nmcc_branches": str(full.hybrid_action_count(slug)),
            "--nmcc_every_k_decisions": "1",
            "--nmcc_horizon_days": "12",
            "--nmcc_actor_weight": "1.0",
            "--nmcc_policy_coeff": "0.25",
            "--nmcc_stages": "market,static,catchup,profit",
            "--branch_imitation_coeff": "0.1",
            "--profit_planner_sample_cap": "8",
            "--ppo_rollout_episodes": "1",
            "--ppo_min_rollout_transitions": "14",
        }
        for flag, value in expected.items():
            assert full.option(command, flag) == value
        assert full.option(command, "--nmcc_dump") == str(
            dedicated.nmcc_dump_path(tmp_path / slug, slug).resolve()
        )


def test_dedicated_schedule_has_exact_trajectory_aligned_stages():
    counts = dedicated.DEFAULT_STAGE_DAYS
    assert counts == (14, 14, 56, 112)
    assert all(value % dedicated.TRAJECTORY_DAYS == 0 for value in counts)
    fractions = dedicated.stage_fractions(dedicated.DEFAULT_DAYS, counts)
    scheduler = TrainingStageScheduler(
        stage_fractions=tuple(fractions[key] for key in dedicated.STAGE_KEYS),
        dynamic_episode_days=dedicated.TRAJECTORY_DAYS,
    )
    observed = {stage.name: 0 for stage in scheduler.stages}
    for day in range(dedicated.DEFAULT_DAYS):
        observed[
            scheduler.stage_at(day / dedicated.DEFAULT_DAYS).name
        ] += 1
    assert observed == {
        "market": 14,
        "static_opponent": 14,
        "price_gap_catchup": 56,
        "profit_maximizers": 112,
    }


def test_dedicated_evaluation_is_broad_but_bounded(tmp_path):
    command = dedicated.bounded.evaluation_command(
        "nyc",
        tmp_path,
        tmp_path / "policy.pt",
        openings=dedicated.EVALUATION_OPENINGS,
    )
    assert full.option(command, "--openings") == "1"
    assert full.option(command, "--horizon-days") == "14"
    assert len(full.option(command, "--rivals").split(",")) == 23
    assert len(full.option(command, "--specs").split(";")) == 5
    assert len(command[5].split(",")) == 6


def test_full_credit_validator_rejects_missing_action_advantage(tmp_path):
    out = tmp_path / "nyc"
    out.mkdir()
    dump = dedicated.nmcc_dump_path(out, "nyc")
    row = {
        "day": 0,
        "stage": "market",
        "horizon_days": 12,
        "feasible_action_count": 3,
        "computed_action_count": 2,
        "advantage_coverage": 2 / 3,
        "selected": "hold",
        "values": {"hold": 0.0, "whole_tariff_step:+0.10": 0.2},
        "raw_values": {"hold": 1.0, "whole_tariff_step:+0.10": 1.2},
        "seconds": 1.0,
    }
    dump.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="incomplete feasible-action credit"):
        dedicated.validate_full_counterfactual_credit(
            "nyc", out, 1, (1, 0, 0, 0)
        )
