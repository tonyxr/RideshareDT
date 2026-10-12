import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "v25_v29_ablation_ladder", ROOT / "scripts/v25_v29_ablation_ladder.py"
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _command(stage_id: str) -> list[str]:
    stage = next(stage for stage in module.STAGES if stage.stage_id == stage_id)
    return module.command_for(stage, module.BUDGETS["smoke"], 17101, ROOT / "tmp" / stage_id)


def test_ladder_is_cumulative_and_rooted_in_frozen_v25():
    assert [stage.stage_id for stage in module.STAGES] == [
        "s0_frozen_v25",
        "s1_active_v25_contract",
        "s2_v29_setup",
        "s3_v29_simulator",
        "s4_v29_mdp",
        "s5_v29_learner",
    ]
    assert str(module.BASELINE / "Core.py") in _command("s0_frozen_v25")
    assert str(module.ROOT / "Core.py") in _command("s1_active_v25_contract")


def test_active_v25_contract_does_not_enable_later_components():
    command = _command("s1_active_v25_contract")
    for flag in (
        "--v26", "--v27", "--v28", "--v29", "--rider_memory",
        "--independent_coefficient_actions", "--compact_foresight_observation",
    ):
        assert flag not in command
    assert "--credit_assignment_v25" in command


def test_setup_changes_only_data_contract_before_simulator():
    command = _command("s2_v29_setup")
    assert "native_uncapped_primary.jsonl" in " ".join(command)
    assert "--gpt_threshold_mode" in command
    assert "--v27" not in command
    assert "--independent_coefficient_actions" not in command
    assert "--credit_assignment_v25" in command


def test_simulator_precedes_mdp_and_learner():
    command = _command("s3_v29_simulator")
    for flag in ("--v27", "--shocks", "--rider_memory"):
        assert flag in command
    assert "--independent_coefficient_actions" not in command
    assert "--compact_foresight_observation" not in command
    assert "--credit_assignment_v25" in command


def test_mdp_precedes_learner_and_enforces_requested_floor():
    command = _command("s4_v29_mdp")
    for flag in (
        "--v28", "--independent_coefficient_actions",
        "--compact_foresight_observation", "--minimum_tariff_multiplier",
    ):
        assert flag in command
    floor_index = command.index("--minimum_tariff_multiplier")
    assert float(command[floor_index + 1]) == 1.0
    assert "--credit_assignment_v25" in command
    assert "--v29" not in command


def test_final_rung_changes_learner_contract_without_cpi_or_fallback():
    command = _command("s5_v29_learner")
    assert "--v29" in command
    assert "--credit_assignment_v25" not in command
    assert command[command.index("--policy_objective") + 1] == "profit_with_dominance"
    assert command[command.index("--training_curriculum") + 1] == "environment_then_opponent"
    joined = " ".join(command).lower()
    assert "conservative" not in joined
    assert "fallback" not in joined


def test_matched_budget_and_seed_are_identical_across_rungs():
    commands = [_command(stage.stage_id) for stage in module.STAGES]
    for flag in (
        "--seed", "--experiment_seed", "--train_timesteps", "--train_customers",
        "--train_steps_per_day", "--firm1_action_interval_steps",
        "--ppo_min_rollout_transitions", "--eval_timesteps", "--eval_customers",
    ):
        values = [command[command.index(flag) + 1] for command in commands]
        assert len(set(values)) == 1, (flag, values)
