from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_revision15d_8h_multicity as fast  # noqa: E402
import run_revision15d_multicity_full as full  # noqa: E402
from platform_mdp import TrainingStageScheduler  # noqa: E402


def test_eight_hour_training_contract_preserves_credit_depth(tmp_path):
    fractions = {
        "market": 0.05,
        "static": 0.05,
        "catchup": 0.375,
        "profit": 0.525,
    }
    for slug in fast.CITIES:
        command = fast.build_training_command(
            slug, tmp_path / slug, 6000, 251025, fractions
        )
        assert full.option(command, "--train_timesteps") == "6000"
        assert full.option(command, "--nmcc_horizon_days") == "12"
        assert full.option(command, "--nmcc_branches") == str(
            full.hybrid_action_count(slug)
        )
        assert full.option(command, "--nmcc_every_k_decisions") == "211"
        assert full.option(command, "--training_dynamic_episode_days") == "14"
        assert full.option(command, "--ppo_rollout_episodes") == "2"
        assert full.option(command, "--eval_timesteps") == "14"
        assert full.option(command, "--comparison_limit") == "1000"
        assert full.option(
            command, "--checkpoint_validation_interval_days"
        ) == "6000"


def test_eight_hour_evaluation_is_stratified_and_parallel(tmp_path):
    policy = tmp_path / "policy.final_training.pt"
    command = fast.evaluation_command("nyc", tmp_path, policy)
    assert command[5] == ",".join(fast.EVALUATION_CASES)
    assert full.option(command, "--workers") == "3"
    assert full.option(command, "--openings") == "2"
    assert full.option(command, "--horizon-days") == "14"
    assert len(full.option(command, "--specs").split(";")) == 5
    assert len(full.option(command, "--rivals").split(",")) == len(full.RIVALS)


def test_eight_hour_case_subset_is_explicit():
    assert len(fast.EVALUATION_CASES) == 6
    assert "C0_calm" in fast.EVALUATION_CASES
    assert "A1_shortage_severe" in fast.EVALUATION_CASES
    assert "S6_shortage_during_promotion" in fast.EVALUATION_CASES


def test_full_counterfactual_today_contract_replays_every_feasible_action_daily(
    tmp_path,
):
    fractions = {
        "market": 2 / 48,
        "static": 2 / 48,
        "catchup": 18 / 48,
        "profit": 26 / 48,
    }
    for slug in fast.CITIES:
        command = fast.build_training_command(
            slug,
            tmp_path / slug,
            48,
            251025,
            fractions,
            nmcc_every_k_decisions=1,
            nmcc_horizon_days=7,
            nmcc_actor_weight=1.0,
            nmcc_stages="static,catchup,profit",
        )
        assert full.option(command, "--nmcc_every_k_decisions") == "1"
        assert full.option(command, "--nmcc_horizon_days") == "7"
        assert full.option(command, "--nmcc_actor_weight") == "1.0"
        assert full.option(command, "--nmcc_stages") == "static,catchup,profit"
        assert full.option(command, "--nmcc_branches") == str(
            full.hybrid_action_count(slug)
        )

    evaluation = fast.evaluation_command(
        "nyc", tmp_path, tmp_path / "policy.pt", openings=1
    )
    assert full.option(evaluation, "--openings") == "1"


def test_exact_short_stage_days_survive_scheduler_boundary_rounding():
    fractions = fast.resolve_stage_fractions(48, "2,2,18,26")
    scheduler = TrainingStageScheduler(
        stage_fractions=tuple(fractions[key] for key in fast.STAGE_KEYS),
        dynamic_episode_days=14,
    )
    counts = {stage.name: 0 for stage in scheduler.stages}
    for day in range(48):
        counts[scheduler.stage_at(day / 48).name] += 1
    assert counts == {
        "market": 2,
        "static_opponent": 2,
        "price_gap_catchup": 18,
        "profit_maximizers": 26,
    }
