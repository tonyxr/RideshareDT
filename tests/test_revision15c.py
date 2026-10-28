"""Revision 15c reward, action, curriculum, and launch contracts."""
import csv
import json
from pathlib import Path

import pytest

from Core import _parse_stage_market_share_weights
from calibration_artifact import load_city_artifact
from platform_mdp import (
    LongTermProfitRewardConfig,
    PolicyObjectiveReward,
    TrainingStageScheduler,
)
from pricing_models import FirmRLPricer
from scripts import run_revision15c_multicity_full as r15c
from scripts.summarize_revision15b_benchmarks import row_metrics


KEYS = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]


def reward_config() -> LongTermProfitRewardConfig:
    return LongTermProfitRewardConfig(
        objective_mode="profit_maximization",
        stage_specific_rewards=True,
        four_stage_curriculum_rewards=True,
        learning_corrections=True,
        temporal_competitive_reward=False,
        stage_profit_lead_weights=r15c.STAGE_PROFIT_LEAD_WEIGHTS,
        stage_market_share_competitiveness_weights=(
            r15c.STAGE_MARKET_SHARE_WEIGHTS
        ),
        market_share_competitiveness_weight=0.0,
    )


def test_each_stage_has_its_own_local_exploration_decay():
    scheduler = TrainingStageScheduler(
        stage_fractions=(0.05, 0.05, 0.375, 0.525),
        dynamic_episode_days=30,
    )
    starts = []
    for stage in scheduler.stages:
        opening = scheduler.controls_for_stage(stage, 0.0)
        closing = scheduler.controls_for_stage(stage, 1.0)
        starts.append(opening["exploration_rate"])
        assert opening["exploration_rate"] == pytest.approx(stage.exploration_rate)
        assert closing["exploration_rate"] == pytest.approx(stage.exploration_end_rate)
        assert opening["exploration_rate"] > closing["exploration_rate"]
        assert opening["entropy_scale"] > closing["entropy_scale"]
        assert opening["learning_rate_scale"] > closing["learning_rate_scale"]
    assert starts == pytest.approx([0.25, 0.16, 0.12, 0.08])


def test_dense_factorized_actor_separates_coefficient_and_magnitude():
    pricer = FirmRLPricer(
        seed=15,
        opt_keys=KEYS,
        tariff_targets=r15c.MAGNITUDE_TARGETS,
    )
    assert pricer.level_step_actions is False
    assert pricer.independent_coefficient_actions is True
    assert pricer.agent.action_dim == 1 + len(KEYS) * len(r15c.MAGNITUDE_TARGETS)
    assert pricer.agent.net.pi_target_head is not None
    assert pricer.agent.net.pi_magnitude_head is not None
    for action, mapping in pricer.action_to_steps.items():
        if action:
            # The 1.0x magnitude is an intentional coefficient-specific reset
            # to its calibrated anchor; every other magnitude changes exactly
            # one coefficient.
            expected = 0 if pricer.action_to_multiplier[action] == 1.0 else 1
            assert sum(value != 0 for value in mapping.values()) == expected
    for key in KEYS:
        magnitudes = {
            pricer.action_to_multiplier[action]
            for action, target in pricer.action_to_target_key.items()
            if target == key
        }
        assert magnitudes == set(r15c.MAGNITUDE_TARGETS)


def test_final_stage_reward_is_majority_competitive_and_preserves_share():
    config = reward_config()
    rows = {
        stage: config.economic_weights(stage)
        for stage in ("market", "static", "catchup", "profit")
    }
    assert rows["market"] == pytest.approx({
        "own_profit": 1.0,
        "profit_lead": 0.0,
        "own_completed_share": 0.0,
        "completed_share_lead": 0.0,
    })
    assert [rows[stage]["completed_share_lead"] for stage in rows] == pytest.approx(
        r15c.STAGE_MARKET_SHARE_WEIGHTS
    )
    assert rows["profit"] == pytest.approx({
        "own_profit": 0.2625,
        "profit_lead": 0.4875,
        "own_completed_share": 0.0,
        "completed_share_lead": 0.25,
    })
    assert (
        rows["profit"]["profit_lead"]
        + rows["profit"]["completed_share_lead"]
    ) == pytest.approx(0.7375)
    # The component critic width is invariant across stage boundaries even
    # when the Stage 1 scalarization assigns zero weight to both share terms.
    assert {
        len(config.critic_component_weights(stage)) for stage in rows
    } == {4}
    for row in rows.values():
        assert sum(row.values()) == pytest.approx(1.0)


def test_reward_mechanism_label_reports_stage_specific_share_term():
    model = PolicyObjectiveReward(reward_config())
    assert model.mechanism_name == "staged_profit_share_v30_corrected"


def test_stage_share_schedule_parser_and_validation_fail_closed():
    assert _parse_stage_market_share_weights("0,0.05,0.10,0.25") == (
        0.0, 0.05, 0.10, 0.25
    )
    with pytest.raises(ValueError, match="four-stage"):
        LongTermProfitRewardConfig(
            stage_market_share_competitiveness_weights=(0.0, 0.05, 0.1, 0.25)
        )
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        LongTermProfitRewardConfig(
            stage_specific_rewards=True,
            four_stage_curriculum_rewards=True,
            stage_market_share_competitiveness_weights=(0.0, 0.05, 0.1, 1.0),
        )


@pytest.mark.parametrize("city", tuple(r15c.CITIES))
def test_multicity_command_is_4000_day_revision15c_contract(city):
    out = Path("/tmp") / f"revision15c-test-{city}"
    command = r15c.build_training_command(city, out, 4000, 251015)
    assert "--level_step_actions" not in command
    assert "--tariff_structure" not in command
    assert "--stage_entropy_end_scale" not in command
    assert "--stage_lr_end_scale" not in command
    assert r15c.option(command, "--tariff_targets") == ",".join(
        str(value) for value in r15c.MAGNITUDE_TARGETS
    )
    assert r15c.option(command, "--stage_profit_lead_weights") == (
        "0.0,0.35,0.55,0.65"
    )
    assert r15c.option(
        command, "--stage_market_share_competitiveness_weights"
    ) == "0.0,0.05,0.1,0.25"
    assert r15c.option(command, "--training_stage_dynamic_fraction") == "0.375"
    assert r15c.option(command, "--training_stage_profit_fraction") == "0.525"
    expected_coefficients = 4 if city == "boston" else 5
    assert r15c.manipulated_coefficient_count(city) == expected_coefficients
    assert r15c.option(
        command, "--manipulated_price_coefficients"
    ) == str(expected_coefficients)

    calibration = load_city_artifact(
        r15c.CITY_CALIBRATION_ROOT / city / "city_calibration.json",
        city=r15c.CITIES[city]["market"],
        lead_weight=0.6,
        grid=list(r15c.MAGNITUDE_TARGETS),
    )
    assert calibration["grid"] == list(r15c.MAGNITUDE_TARGETS)


@pytest.mark.parametrize("city", tuple(r15c.CITIES))
def test_multicity_command_supports_exact_8000_day_stage_contract(city):
    out = Path("/tmp") / f"revision15c-8000-test-{city}"
    fractions = r15c.resolve_stage_fractions(8000, "400,400,2400,4800")
    assert fractions == {
        "market": 0.05,
        "static": 0.05,
        "catchup": 0.30,
        "profit": 0.60,
    }
    command = r15c.build_training_command(city, out, 8000, 251016, fractions)
    assert r15c.option(command, "--train_timesteps") == "8000"
    assert r15c.option(command, "--training_stage_market_fraction") == "0.05"
    assert r15c.option(command, "--training_stage_static_fraction") == "0.05"
    assert r15c.option(command, "--training_stage_dynamic_fraction") == "0.3"
    assert r15c.option(command, "--training_stage_profit_fraction") == "0.6"


def test_exact_stage_days_fail_closed():
    with pytest.raises(ValueError, match="sum"):
        r15c.resolve_stage_fractions(8000, "400,400,2400,4700")
    with pytest.raises(ValueError, match="four positive integers"):
        r15c.resolve_stage_fractions(8000, "400,400,7200")


def test_boston_prunes_unidentified_zero_airport_action_family():
    assert r15c.manipulated_coefficient_count("boston") == 4
    command = r15c.build_training_command(
        "boston", Path("/tmp/revision15c-boston-identifiability"), 4000, 251015
    )
    assert r15c.option(command, "--manipulated_price_coefficients") == "4"

    # Four identified coefficients times fifteen magnitudes, plus hold.
    pricer = FirmRLPricer(
        seed=15,
        opt_keys=KEYS[:4],
        tariff_targets=r15c.MAGNITUDE_TARGETS,
    )
    assert pricer.agent.action_dim == 61
    assert "airport_fee" not in pricer.action_keys


def test_root_manifest_records_city_specific_action_dimensions(tmp_path):
    root = tmp_path / "manifest"
    r15c.prepare_root(root, 4000, 251015)
    payload = json.loads((root / "manifest.json").read_text())
    contract = payload["action_contract"]
    assert contract["coefficient_targets_by_city"] == {
        "nyc": 5,
        "boston": 4,
        "chicago": 5,
    }
    assert contract["joint_actions_including_hold_by_city"] == {
        "nyc": 76,
        "boston": 61,
        "chicago": 76,
    }


def test_reality_gap_preserves_quote_semantics_and_legacy_alias(tmp_path):
    path = tmp_path / "observed.csv"
    rows = [{
        "observed_price": str(12.0 + (index % 2)),
        "actual_paid": "999.0",  # must not override the authoritative field
        "price_semantics": "posted_quote_excluding_tip",
        "rl_predicted_price": str(13.0 + (index % 2)),
        "anchor_tariff_price": str(10.0 + (index % 2)),
    } for index in range(5000)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = r15c.reality_gap_summary(path)
    assert summary["observed_price_column"] == "observed_price"
    assert summary["price_semantics"] == "posted_quote_excluding_tip"
    assert summary["observed_price_mean"] == pytest.approx(12.5)
    assert summary["actual_price_mean"] == pytest.approx(12.5)
    assert summary["mae"] == pytest.approx(1.0)


def test_reality_gap_constant_series_reports_undefined_correlation():
    assert r15c.correlation([12.0, 12.0], [13.0, 14.0]) is None
    assert r15c.correlation([12.0, 13.0], [14.0, 14.0]) is None


def test_benchmark_summary_separates_absolute_and_two_firm_share():
    metrics = row_metrics({"own_share": 0.30, "rival_share": 0.20})
    assert metrics["firm1_completed_request_share"] == pytest.approx(0.30)
    assert metrics["firm2_completed_request_share"] == pytest.approx(0.20)
    assert metrics["outside_or_unfulfilled_share"] == pytest.approx(0.50)
    assert metrics["firm1_market_share"] == pytest.approx(0.60)
    assert metrics["firm2_market_share"] == pytest.approx(0.40)
