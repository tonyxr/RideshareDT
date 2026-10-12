from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from v28_evaluation_reporting import (
    BootstrapConfig,
    ClairvoyantEventPolicyFactory,
    EvaluationContext,
    EvaluationContractError,
    FinalTestAlreadyConsumed,
    FixedEvaluationPolicy,
    ReportIntegrityError,
    compute_adaptivity_index,
    evaluate_policy_suite,
    freeze_report,
    load_frozen_report,
    paired_policy_comparison,
    render_frozen_report,
    rollout_policy_with_metrics,
    shock_aligned_action_summary,
    summarize_seed_values,
)


BOOTSTRAP = BootstrapConfig(replicates=300, confidence=0.90, seed=19)


@dataclass
class _Firm:
    total: float
    reference_total: float
    completed: float
    dispatch_offers: float
    profit: float
    revenue: float
    driver_pay: float
    wait_minutes: float


@dataclass
class _DriverState:
    total_online_minutes: float
    total_driver_pay: float


class _Supply:
    def __init__(self) -> None:
        self.last_states = {}


class _Core:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.action = np.array([1.0, 0.0, 1.0])
        self.block = 0
        self.driver_supply = _Supply()
        self.last_welfare_stats = {}
        self.last_v27_step = {}

    def simulate_batch(self):
        base, _, incentive = self.action
        seed_noise = (self.seed % 7) * 0.10
        completed1 = 18.0 + 3.0 * base
        completed2 = 30.0
        fare1 = 8.0 + base
        fare2 = 9.0
        pay1 = completed1 * (4.0 * incentive)
        pay2 = completed2 * 4.0
        firm1 = _Firm(
            total=100.0,
            reference_total=100.0,
            completed=completed1,
            dispatch_offers=35.0,
            profit=100.0 * (base + seed_noise),
            revenue=completed1 * fare1,
            driver_pay=pay1,
            wait_minutes=completed1 * (7.0 - 0.2 * incentive),
        )
        firm2 = _Firm(
            total=100.0,
            reference_total=100.0,
            completed=completed2,
            dispatch_offers=40.0,
            profit=80.0,
            revenue=completed2 * fare2,
            driver_pay=pay2,
            wait_minutes=completed2 * 6.0,
        )
        low_served = 20.0 + base
        high_served = 30.0 + base
        self.last_welfare_stats = {
            "by_income": {
                "<50k": [50.0, low_served, low_served * 9.0, 70.0 + base, 10.0, low_served * 7.0],
                "200k+": [50.0, high_served, high_served * 10.0, 120.0 + base, 5.0, high_served * 6.0],
            },
            "firm1_profit": firm1.profit,
            "firm2_profit": firm2.profit,
            "driver_pay": pay1 + pay2,
        }
        self.driver_supply.last_states = {
            "Firm1": _DriverState(total_online_minutes=600.0, total_driver_pay=pay1)
        }
        self.last_v27_step = {
            "fleet": {
                "Firm1": {
                    "earnings_per_online_hour": pay1 / 10.0,
                    "online_drivers": 2.0,
                },
                "Firm2": {
                    "earnings_per_online_hour": pay2 / 10.0,
                    "online_drivers": 2.0,
                },
            }
        }
        self.block += 1
        return None, firm1, firm2, 0.0, 0.0, 0.0

    def _v27_block_minutes(self):
        return 300.0


class _FleetOnlyCore(_Core):
    def simulate_batch(self):
        result = super().simulate_batch()
        self.driver_supply.last_states = {}
        return result


class _MarketEnv:
    def __init__(self, split: str = "validation", days: int = 3) -> None:
        self.profile_split = split
        self.days = int(days)
        self.core = None

    def reset(self, seed: int):
        self.seed = int(seed)
        self.day = 0
        self.core = _Core(seed)
        return np.zeros(2, dtype=np.float32)

    def step(self, action):
        self.core.action = np.asarray(action, dtype=float)
        rewards = []
        for _ in range(2):
            _, firm1, _, _, _, _ = self.core.simulate_batch()
            rewards.append(firm1.profit / firm1.reference_total)
        reward = float(sum(rewards))
        active = ["driver_shortage"] if self.day == 1 else []
        info = {
            "day": self.day,
            "date": f"2026-01-{self.day + 1:02d}",
            "weather": "clear",
            "active_shocks": active,
            "shock_phases": ["active"] if active else [],
            "public_shock": {"storm_warning": 0.0},
            "block_tariff_multipliers": [float(action[0]), float(action[0])],
            "summary": {
                "profit_firm1": reward / 2.0,
                "completed_share_firm1": 0.20,
                "completed_share_firm2": 0.30,
                "fulfillment_firm1": 0.60,
                "wait_firm1": 7.0,
                "earnings_firm1": 20.0,
            },
        }
        self.day += 1
        return np.zeros(2, dtype=np.float32), reward, self.day >= self.days, info

    def clairvoyant_event_days(self, seed: int):
        del seed
        return (1,)


class _FleetOnlyEnv(_MarketEnv):
    def reset(self, seed: int):
        self.seed = int(seed)
        self.day = 0
        self.core = _FleetOnlyCore(seed)
        return np.zeros(2, dtype=np.float32)


class _SummaryOnlyEnv:
    profile_split = "validation"

    def reset(self, seed: int):
        self.day = 0
        return np.zeros(1, dtype=np.float32)

    def step(self, action):
        del action
        info = {
            "day": self.day,
            "active_shocks": [],
            "summary": {
                "profit_firm1": 1.5,
                "completed_share_firm1": 0.25,
                "completed_share_firm2": 0.35,
                "fulfillment_firm1": 0.75,
                "wait_firm1": 6.0,
                "earnings_firm1": 22.0,
            },
        }
        self.day += 1
        return np.zeros(1, dtype=np.float32), 1.5, self.day >= 2, info


class _Fixed:
    def __init__(self, base: float) -> None:
        self.base = float(base)

    def __call__(self, observation, day):
        del observation
        return np.array([self.base, 0.1 * day, 1.0 + 0.05 * day])


def _suite(split: str = "validation", ledger_path: Path | None = None):
    context = EvaluationContext(
        run_id="report-test",
        split=split,
        purpose="one_shot_final_audit" if split == "test" else "validation_diagnostic",
        policies_frozen=split == "test",
        final_test_authorized=split == "test",
        ledger_path=ledger_path,
    )
    factories = {
        "static": lambda: _Fixed(1.0),
        "learner": lambda: _Fixed(2.0),
        "oracle": lambda: _Fixed(3.0),
    }
    manifest = {
        "static": {"kind": "fixed", "action": [1.0, 0.0, 1.0]},
        "learner": {"checkpoint_sha256": "abc"},
        "oracle": {"kind": "clairvoyant", "fit_split": "validation"},
    }
    return evaluate_policy_suite(
        lambda: _MarketEnv(split=split),
        factories,
        [11, 12, 13, 14],
        context=context,
        policy_manifest=manifest,
        bootstrap=BOOTSTRAP,
        baseline_policy="static",
        learner_policy="learner",
        oracle_policy="oracle",
        metadata={"city": "synthetic"},
    )


def test_seed_bootstrap_and_paired_comparison_are_clustered_and_reproducible():
    first = summarize_seed_values([1.0, 2.0, 4.0, 8.0], bootstrap=BOOTSTRAP, label="x")
    second = summarize_seed_values([1.0, 2.0, 4.0, 8.0], bootstrap=BOOTSTRAP, label="x")
    assert first == second
    assert first["n_seeds"] == 4
    assert first["ci_lower"] < first["mean"] < first["ci_upper"]

    left = {"episodes": [{"seed": 1, "metrics": {"episode_return": 3.0}}, {"seed": 2, "metrics": {"episode_return": 7.0}}]}
    right = {"episodes": [{"seed": 1, "metrics": {"episode_return": 1.0}}, {"seed": 2, "metrics": {"episode_return": 3.0}}]}
    paired = paired_policy_comparison(left, right, bootstrap=BOOTSTRAP, label="l-r")
    assert paired["episode_return"]["mean"] == pytest.approx(3.0)
    assert paired["episode_return"]["paired_seeds"] == [1, 2]


def test_live_rollout_captures_exact_operational_welfare_and_equity_metrics():
    episode = rollout_policy_with_metrics(
        _MarketEnv(), _Fixed(2.0), seed=11, expected_split="validation"
    )
    metrics = episode["metrics"]
    assert episode["metric_source"] == "core_intraday_blocks"
    assert episode["days"] == 3
    assert metrics["mean_fare_paid_firm1"] == pytest.approx(10.0)
    assert metrics["driver_pay_per_completed_firm1"] == pytest.approx(4.2)
    assert metrics["fulfillment_rate_firm1"] == pytest.approx(24.0 / 35.0)
    assert metrics["driver_earnings_per_online_hour_firm1"] == pytest.approx(10.08)
    assert metrics["consumer_surplus_per_request_market"] is not None
    assert metrics["equity_served_rate__<50k"] == pytest.approx(22.0 / 50.0)
    assert metrics["equity_served_gap_high_minus_low"] == pytest.approx(0.20)
    assert episode["daily_trace"][1]["active_shocks"] == ["driver_shortage"]


def test_compact_historical_rollout_marks_unobserved_metrics_missing():
    episode = rollout_policy_with_metrics(
        _SummaryOnlyEnv(), _Fixed(1.0), seed=4, expected_split="validation"
    )
    assert episode["metric_source"] == "environment_daily_summary"
    assert episode["metrics"]["profit_per_request_firm1"] == pytest.approx(1.5)
    assert episode["metrics"]["mean_fare_paid_firm1"] is None
    assert "does not contain fare totals" in episode["metric_missing"]["mean_fare_paid_firm1"]
    assert episode["metrics"]["consumer_surplus_per_request_market"] is None


def test_driver_earnings_use_online_hour_weighted_fleet_telemetry_without_batch_state():
    episode = rollout_policy_with_metrics(
        _FleetOnlyEnv(), _Fixed(2.0), seed=11, expected_split="validation"
    )
    assert episode["metrics"]["driver_earnings_per_online_hour_firm1"] == pytest.approx(
        10.08
    )
    assert (
        "driver_earnings_per_online_hour_firm1" not in episode["metric_missing"]
    )


def test_suite_has_crn_pairing_adaptivity_and_shock_aligned_actions():
    report = _suite()
    assert report["evaluation_contract"]["common_random_numbers"] is True
    comparison = report["paired_comparisons"]["learner_minus_static"]["episode_return"]
    assert comparison["n_seeds"] == 4
    assert comparison["mean"] == pytest.approx(6.0)
    assert report["adaptivity_index"]["value"] == pytest.approx(0.5)
    assert report["adaptivity_index"]["oracle_is_fair_policy"] is False
    aligned = report["shock_aligned_actions"]["policies"]["learner"]["events"]
    assert "driver_shortage" in aligned
    onset = next(row for row in aligned["driver_shortage"] if row["relative_day"] == 0)
    assert onset["actions"]["base_tariff_multiplier"]["mean"] == pytest.approx(2.0)

    direct = shock_aligned_action_summary(
        report["policies"], bootstrap=BOOTSTRAP, days_before=1, days_after=1
    )
    assert direct["days_before"] == 1


def test_seed_aware_clairvoyant_factory_uses_only_declared_event_days():
    report = evaluate_policy_suite(
        lambda: _MarketEnv(),
        {
            "static": lambda: FixedEvaluationPolicy([1.0, 0.0, 1.0]),
            "oracle": ClairvoyantEventPolicyFactory(
                [1.0, 0.0, 1.0], [3.0, 0.0, 1.0]
            ),
        },
        [7, 8],
        context=EvaluationContext(run_id="oracle", split="validation"),
        policy_manifest={
            "static": {"action": [1.0, 0.0, 1.0]},
            "oracle": {"information": "latent event days"},
        },
        bootstrap=BOOTSTRAP,
        baseline_policy="static",
    )
    actions = report["policies"]["oracle"]["episodes"][0]["actions"]
    assert [row[0] for row in actions] == [1.0, 3.0, 1.0]


def test_adaptivity_is_withheld_when_oracle_has_no_positive_opportunity():
    episode = lambda seed, value: {"seed": seed, "metrics": {"episode_return": value}}
    learner = {"episodes": [episode(1, 2.0), episode(2, 2.0)]}
    static = {"episodes": [episode(1, 1.0), episode(2, 1.0)]}
    oracle = {"episodes": [episode(1, 0.5), episode(2, 0.5)]}
    result = compute_adaptivity_index(
        learner, static, oracle, bootstrap=BOOTSTRAP
    )
    assert result["available"] is False
    assert result["oracle_minus_static"] < 0.0


def test_final_test_is_explicit_one_shot_and_consumed_before_rollout(tmp_path):
    ledger = tmp_path / "final_test_ledger.json"
    report = _suite("test", ledger)
    assert report["evaluation_contract"]["final_test_one_shot"] is True
    assert ledger.exists()
    assert Path(str(ledger) + ".claimed").exists()
    with pytest.raises(FinalTestAlreadyConsumed):
        _suite("test", ledger)

    with pytest.raises(EvaluationContractError):
        EvaluationContext(
            run_id="bad",
            split="test",
            policies_frozen=True,
            selection_performed=True,
            final_test_authorized=True,
            ledger_path=tmp_path / "bad.json",
        )


def test_profile_split_mismatch_fails_before_reporting():
    with pytest.raises(EvaluationContractError, match="profile_split"):
        rollout_policy_with_metrics(
            _MarketEnv(split="train"),
            _Fixed(1.0),
            seed=1,
            expected_split="validation",
        )


def test_freeze_integrity_and_publication_bundle(tmp_path):
    report = _suite()
    frozen = freeze_report(
        report,
        tmp_path / "frozen.json",
        frozen_at_utc="2026-10-07T00:00:00+00:00",
    )
    loaded = load_frozen_report(frozen)
    assert loaded["freeze"]["content_sha256"]

    bundle = render_frozen_report(frozen, tmp_path / "rendered")
    assert Path(bundle["json"]).exists()
    assert Path(bundle["markdown"]).exists()
    figure_names = {Path(path).name for path in bundle["figures"]}
    assert "policy_performance.png" in figure_names
    assert "stakeholder_outcomes.png" in figure_names
    assert "income_equity.png" in figure_names
    assert "shock_adaptation.png" in figure_names
    markdown = Path(bundle["markdown"]).read_text(encoding="utf-8")
    assert "Paired return contrasts" in markdown
    assert "unfair information upper bound" in markdown

    tampered = json.loads(frozen.read_text(encoding="utf-8"))
    tampered["run_id"] = "changed-after-freeze"
    broken = tmp_path / "tampered.json"
    broken.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ReportIntegrityError):
        load_frozen_report(broken)
