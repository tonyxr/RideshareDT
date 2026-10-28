"""Revision 15: level-step actions on a fixed structure, branch imitation, runner contract."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pricing_models as pm  # noqa: E402
from market_dynamics import rescale_pricer_to_market  # noqa: E402
from platform_mdp import SEGMENTS, PlatformObservationModel  # noqa: E402
from PPOAgent import PPOAgent  # noqa: E402

KEYS = list(pm.TARIFF_LEVEL_KEYS)
CALIBRATION = ROOT.parent / "data/processed/nyc/calibration.json"


def nyc_anchors():
    tariff = json.loads(CALIBRATION.read_text())["calibration"]["tariff"]
    return {key: float(tariff[key]) for key in KEYS}


def structure_payload(anchors, reference=1.3):
    raw = {"base_fare": 2.0, "per_minute": 1.6, "per_mile": 1.3, "booking_fee": 1.2, "airport_fee": 1.0}
    level = pm.tariff_level_of(raw, anchors)
    return {
        "format": pm.TARIFF_STRUCTURE_FORMAT,
        "structure": {key: value / level for key, value in raw.items()},
        "anchors": dict(anchors),
        "reference_level": reference,
        "source": {"test": True},
    }


class Market:
    def __init__(self, anchors):
        self.curr_market = type("Anchors", (), dict(anchors))()


@pytest.fixture
def level_pricer():
    anchors = nyc_anchors()
    pricer = pm.FirmRLPricer(
        seed=3, opt_keys=KEYS, independent_coefficient_actions=False, level_step_actions=True,
        tariff_structure=structure_payload(anchors), tariff_level_bounds=(0.6, 2.0),
        entropy_coeff_start=0.01, learning_corrections=False,
    )
    rescale_pricer_to_market(pricer, anchors)
    return pricer, Market(anchors), anchors


# ---------------------------------------------------------------- action contract
def test_level_action_space_and_labels(level_pricer):
    pricer, _, _ = level_pricer
    assert pricer.agent.action_dim == 5
    assert [pricer.action_label(a) for a in range(5)] == [
        "hold", "level_step:-0.20", "level_step:-0.10", "level_step:+0.10", "level_step:+0.20"]
    assert pricer.action_to_multiplier == {} and pricer.action_to_target_key == {}
    assert pricer.agent.ent_coeff == pytest.approx(0.01)
    assert pricer.agent.max_ent_coeff == pytest.approx(0.01)


def test_mask_post_step_restore_and_bounds(level_pricer):
    pricer, market, anchors = level_pricer
    assert pricer.feasible_action_mask(market).tolist() == [True, False, False, False, False]
    values = pricer.post_tariff_level(1.3, market)
    structure = pricer.tariff_structure["structure"]
    for key in KEYS:
        lb, ub = pricer.config.bounds[key]
        assert values[key] == pytest.approx(float(np.clip(anchors[key] * structure[key] * 1.3, lb, ub)))
    snapshot = pricer.snapshot_execution_state()
    overrides = copy.deepcopy(pricer.overrides)
    pricer.apply_action(4, market)
    assert pricer.current_tariff_level() == pytest.approx(1.5)
    assert pricer.last_action_descriptor.target == "tariff_level"
    pricer.restore_execution_state(snapshot)
    pricer.overrides = overrides
    assert pricer.current_tariff_level() == pytest.approx(1.3)
    for _ in range(7):
        pricer.apply_action(3, market)
    assert pricer.current_tariff_level() == pytest.approx(2.0)
    assert pricer.feasible_action_mask(market).tolist() == [True, True, True, False, False]
    with pytest.raises(ValueError, match="masked level step"):
        pricer.apply_action(3, market)
    pricer.apply_action(0, market)
    assert pricer.current_tariff_level() == pytest.approx(2.0)
    pricer.release_tariff_level()
    assert pricer.feasible_action_mask(market).tolist() == [True, False, False, False, False]
    with pytest.raises(RuntimeError):
        pricer.apply_action(3, market)


def test_invalid_levels_structures_and_anchors(level_pricer):
    pricer, market, anchors = level_pricer
    for bad in (1.25, 2.1, 0.5):
        with pytest.raises(ValueError):
            pricer.post_tariff_level(bad, market)
    shifted = dict(anchors, base_fare=anchors["base_fare"] * 1.01)
    with pytest.raises(ValueError, match="differs"):
        pricer.post_tariff_level(1.0, Market(shifted))
    good = structure_payload(anchors)
    for mutate in (
        lambda q: q.update(format="other"),
        lambda q: q["structure"].pop("airport_fee"),
        lambda q: q["structure"].update(base_fare=q["structure"]["base_fare"] * 1.1),
        lambda q: q.update(reference_level=1.35),
        lambda q: q.update(reference_level=2.2),
    ):
        payload = copy.deepcopy(good)
        mutate(payload)
        with pytest.raises(ValueError):
            pm.validate_tariff_structure(payload, KEYS, (0.6, 2.0))


def test_zero_city_anchor_is_valid_when_total_exposure_is_positive():
    anchors = nyc_anchors()
    anchors["airport_fee"] = 0.0
    payload = structure_payload(anchors)
    validated = pm.validate_tariff_structure(payload, KEYS, (0.6, 2.0))
    assert validated["anchors"]["airport_fee"] == 0.0
    assert pm.tariff_level_of(validated["structure"], validated["anchors"]) == pytest.approx(1.0)

    negative = copy.deepcopy(payload)
    negative["anchors"]["airport_fee"] = -0.01
    with pytest.raises(ValueError, match="nonnegative"):
        pm.validate_tariff_structure(negative, KEYS, (0.6, 2.0))

    all_zero = copy.deepcopy(payload)
    all_zero["anchors"] = {key: 0.0 for key in KEYS}
    with pytest.raises(ValueError, match="positive exposure-weighted"):
        pm.validate_tariff_structure(all_zero, KEYS, (0.6, 2.0))


def test_contracts_are_exclusive_and_flags_off_unchanged():
    anchors = nyc_anchors()
    with pytest.raises(ValueError):
        pm.FirmRLPricer(seed=1, opt_keys=KEYS, independent_coefficient_actions=True, level_step_actions=True,
                        tariff_structure=structure_payload(anchors), tariff_level_bounds=(0.6, 2.0))
    with pytest.raises(ValueError):
        pm.FirmRLPricer(seed=1, opt_keys=KEYS, independent_coefficient_actions=False, level_step_actions=True,
                        tariff_structure=structure_payload(anchors), tariff_level_bounds=(0.6, 2.0),
                        tariff_targets=(1.0, 1.2))
    legacy = pm.FirmRLPricer(seed=1, opt_keys=KEYS, tariff_targets=(0.6, 0.8, 1.0, 1.2))
    assert legacy.level_step_actions is False and legacy.tariff_level is None
    assert legacy.agent.action_dim == 1 + 5 * 4
    assert legacy.feasible_action_mask(None).all()


def test_action_features_use_level_targets(level_pricer):
    pricer, market, anchors = level_pricer
    pricer.post_tariff_level(1.3, market)
    standalone = pricer.build_action_feature_matrix(market, {})
    assert standalone.shape == (5, 20) and np.isfinite(standalone).all()
    assert standalone[0, 0] == 1.0 and standalone[1, 1] == -1.0 and standalone[4, 1] == 1.0
    observer = PlatformObservationModel.__new__(PlatformObservationModel)
    observer.quote_probes = {segment: {"gap": 0.0, "uncertainty": 1.0} for segment in SEGMENTS}
    observer.config = type("Config", (), {"gap_scale_dollars": 2.0})()
    observer.action_feature_dim = PlatformObservationModel.action_feature_dim
    own = {key: float(getattr(pricer.overrides, key)) for key in KEYS}
    kwargs = dict(action_steps=pricer.action_to_steps, action_keys=pricer.action_keys, own_coefficients=own,
                  anchor_coefficients=anchors, coefficient_steps=dict(pricer.config.step),
                  coefficient_bounds=dict(pricer.config.bounds), step_scale=1.0, target_gap=0.0)
    matrix = observer.build_action_features(**kwargs, action_target_values=pricer.level_target_values(market))
    assert matrix.shape == (5, observer.action_feature_dim) and np.isfinite(matrix).all()
    assert matrix[0, 0] == 1.0 and matrix[3, 1] == 1.0 and matrix[2, 1] == -1.0
    with pytest.raises(ValueError, match="exclusive"):
        observer.build_action_features(**kwargs, action_target_multipliers={1: 1.0},
                                       action_target_values={1: own})


# ---------------------------------------------------------------- branch imitation
def test_branch_imitation_rows_temperature_and_loss():
    nan = float("nan")
    values = torch.tensor([
        [0.0, -0.1, 0.05, 0.10, 0.20],   # complete
        [0.0, nan, -0.05, 0.02, 0.04],   # step 1 masked and unobserved: complete
        [0.0, nan, nan, 0.10, nan],      # feasible actions missing: incomplete
    ])
    action_masks = torch.tensor([
        [True] * 5,
        [True, False, True, True, True],
        [True] * 5,
    ])
    observed, complete = PPOAgent._branch_imitation_rows(values, torch.isfinite(values), action_masks)
    assert complete.tolist() == [True, True, False]
    tau = PPOAgent._branch_imitation_temperature(values, observed, complete, 0.25)
    # torch.median returns the lower of the two middle values for an even count.
    assert tau == pytest.approx(0.25 * 0.09)
    target_logits = torch.where(observed, values / tau, torch.full_like(values, -1e9))
    loss_at_target = PPOAgent._branch_imitation_loss(target_logits, values, observed, complete, tau)
    assert float(loss_at_target) == pytest.approx(0.0, abs=1e-6)
    logits = torch.zeros(3, 5, requires_grad=True)
    masked = logits.masked_fill(~action_masks, -1e9)
    loss = PPOAgent._branch_imitation_loss(masked, values, observed, complete, tau)
    assert float(loss) > 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert int(torch.argmin(logits.grad[0])) == 4      # best branch logit is pushed up
    assert torch.all(logits.grad[2] == 0.0)            # incomplete row receives no gradient


# ---------------------------------------------------------------- runner contract
def test_revision15_command_contract(tmp_path):
    import run_revision15_verify as r15
    structure = r15.write_uniform_structure(tmp_path / "structure.json")
    command = r15.r15_train_command(tmp_path / "run", 4000, oracle=False, structure=structure, smoke=False)
    r15.assert_level_contract(command)
    for flag in ("--tariff_targets", "--whole_tariff_actions"):
        assert flag not in command
    expected = {
        "--rl_warm_start_target": "0", "--nmcc_branches": "5", "--nmcc_horizon_days": "3",
        "--nmcc_policy_coeff": "0.0", "--nmcc_actor_weight": "0.8", "--nmcc_stages": "static,catchup,profit",
        "--branch_imitation_coeff": "0.5", "--branch_imitation_temperature_scale": "0.25",
        "--ppo_entropy_start": "0.01", "--ppo_min_rollout_transitions": "30",
        "--stage_profit_lead_weights": "0.3,0.3,0.3,0.3", "--opponent_profit_lead_weight": "0.6",
        "--tariff_level_bounds": "0.6,2.0", "--train_timesteps": "4000",
    }
    for flag, value in expected.items():
        assert command[command.index(flag) + 1] == value, flag
    for flag in ("--level_step_actions", "--momentum_features", "--opponent_features"):
        assert flag in command
    with pytest.raises(ValueError):
        r15.r15_train_command(tmp_path / "bad", 40, oracle=False, structure=structure, smoke=True,
                              nmcc_branches=3, imitation=0.5)


def test_probe_report_gate(tmp_path):
    import r15_probe_report as report
    rng = np.random.default_rng(5)
    path = tmp_path / "probe.jsonl"
    with path.open("w") as handle:
        for index in range(80):
            shortage = index % 2 == 0
            lift = 0.10 if shortage else -0.02
            values = {"hold": 0.0,
                      "level_step:+0.10": lift + rng.normal(0, 0.02),
                      "level_step:+0.20": lift + rng.normal(0, 0.02),
                      "level_step:-0.10": -lift + rng.normal(0, 0.02)}
            handle.write(json.dumps({"regime": ["driver_shortage"] if shortage else [],
                                     "selected": "hold", "values": values, "firm1_tariff_level": 1.3}) + "\n")
    result = report.analyze(path)
    assert result["pass"] and result["gate"]["t"] > 3.0


# ---------------------------------------------------------------- Core integration (NYC CLI)
def _capture_core(tmp_path, monkeypatch, extra, *, level):
    import Core
    from scripts.run_v26r_staged import command

    def capacity(core, **kwargs):
        return dict(enabled=False, test_only=True, sample_size=kwargs["sample_size"])

    monkeypatch.setattr("city_supply_calibration.calibrate_capacity", capacity)
    argv = command(tmp_path, days=0, city="New York City", slug="nyc", integration=True)
    if level:
        index = argv.index("--tariff_targets")
        del argv[index:index + 2]
    argv[argv.index("--rl_warm_start_target") + 1] = "0"
    argv += ["--learning_corrections", "--learner_revision", "4", "--no-live_diagnostics",
             "--shock_scenario", str(ROOT / "scenarios/calm.json"), "--policy_sweep", "static:1", *extra]
    captured = []

    class Captured(Exception):
        pass

    def capture(core, **kwargs):
        captured.append(core)
        raise Captured()

    monkeypatch.setattr(Core.Core, "run_policy_sweep", capture)
    monkeypatch.setattr(sys, "argv", argv[2:])
    with pytest.raises(Captured):
        Core.main()
    return captured[0]


def test_core_level_contract_end_to_end(tmp_path, monkeypatch):
    anchors = nyc_anchors()
    structure_file = tmp_path / "structure.json"
    structure_file.write_text(json.dumps(structure_payload(anchors, reference=1.3)))
    core = _capture_core(tmp_path / "level", monkeypatch, [
        "--level_step_actions", "--tariff_structure", str(structure_file), "--tariff_level_bounds", "0.6,2.0",
        "--nmcc_branches", "5", "--nmcc_policy_coeff", "0.0", "--branch_imitation_coeff", "0.5",
        "--ppo_entropy_start", "0.01"], level=True)
    legacy = _capture_core(tmp_path / "legacy", monkeypatch, [], level=False)
    pricer = core.firm1
    assert pricer.level_step_actions and pricer.agent.action_dim == 5
    assert pricer.agent.branch_imitation_coeff == pytest.approx(0.5)
    assert core.counterfactual_day_credit == 5

    # Opening: Firm 1 on S * L0 inside the band; Firm 2 identical to the flags-off path.
    core._reset_adaptive_episode(tariff_reset_fraction=0.15, episode_seed=4242)
    legacy._reset_adaptive_episode(tariff_reset_fraction=0.15, episode_seed=4242)
    level = pricer.current_tariff_level()
    assert level in (1.2, 1.3, 1.4)
    expected = pricer.structure_values(level, core.market)
    for key in KEYS:
        assert float(getattr(pricer.overrides, key)) == pytest.approx(expected[key])
        assert float(getattr(core.firm2.overrides, key)) == float(getattr(legacy.firm2.overrides, key))

    # Branch snapshot restores the level together with the tariff.
    snapshot = core._snapshot_counterfactual_state()
    pricer.apply_action(3, core.market)
    assert pricer.current_tariff_level() == pytest.approx(level + 0.1)
    core._restore_counterfactual_state(snapshot)
    assert core.firm1.current_tariff_level() == pytest.approx(level)

    features = core._action_feature_matrix_for_pricer(core.firm1, {})
    assert features.shape[0] == 5 and np.isfinite(features).all()
    assert [core._economic_action_group(a) for a in range(5)] == [0, -1, -1, 1, 1]
    assert core._fare_equivalent_action_impact(4) == pytest.approx(2 * core._fare_equivalent_action_impact(3))
    with pytest.raises(ValueError, match="no warm start"):
        core.warm_start_actor(1.2)

    # Artifact round trip and contract enforcement.
    path = tmp_path / "policy.pt"
    core.save_trained_model(str(path))
    saved = torch.load(path, weights_only=False, map_location="cpu")
    assert saved["action_semantics"] == pm.LEVEL_STEP_ACTION_SEMANTICS
    assert saved["tariff_structure"] == core.firm1.tariff_structure
    core.load_trained_model(str(path))
    original = copy.deepcopy(core.firm1.tariff_structure)
    core.firm1.tariff_structure["reference_level"] = 1.4
    with pytest.raises(ValueError, match="tariff structure"):
        core.load_trained_model(str(path))
    core.firm1.tariff_structure = original

    # A comparator posting releases the level; only hold is then defined.
    core._post_firm1_target(1.0)
    assert core.firm1.current_tariff_level() is None
    assert core.firm1.feasible_action_mask(core.market).tolist() == [True, False, False, False, False]


def test_cli_rejects_inconsistent_level_flags(tmp_path, monkeypatch):
    anchors = nyc_anchors()
    structure_file = tmp_path / "structure.json"
    structure_file.write_text(json.dumps(structure_payload(anchors)))
    with pytest.raises(SystemExit):
        _capture_core(tmp_path / "a", monkeypatch, ["--level_step_actions"], level=True)
    with pytest.raises(SystemExit):
        _capture_core(tmp_path / "b", monkeypatch, ["--tariff_structure", str(structure_file)], level=True)
    with pytest.raises(ValueError, match="replay every action"):
        _capture_core(tmp_path / "c", monkeypatch, [
            "--level_step_actions", "--tariff_structure", str(structure_file), "--nmcc_branches", "3",
            "--branch_imitation_coeff", "0.5"], level=True)
