#!/usr/bin/env python3
"""Train and evaluate the v28 recurrent continuous-control learner.

The ladder is deliberately staged:

* L0: static rival, no shocks -- optimizer sanity check;
* L1: a public winter storm with demand, duration, transit, and fleet effects;
* L2: an unannounced driver shortage -- partial-observability check;
* L3: aggressive adaptive rival, followed by retention checks on prior rungs.
* MIXED: post-gate production curriculum with the natural empirical calendar
  and the complete stochastic shock process.  It cannot start until L0-L3 pass.

Every benchmark candidate is evaluated on identical common-random-number
episode seeds.  Training, benchmark-selection, and final evaluation seeds are
disjoint.  Full-budget defaults are intentionally expensive and must be
selected explicitly with ``--budget full``.

Examples
--------
Smoke validation::

    python scripts/v28_learner_ladder.py --city chicago --budget smoke \
      --stages l0,l1,l2 --out artifacts/v28_rsac/chicago_smoke

Full staged run::

    python scripts/v28_learner_ladder.py --city chicago --budget full \
      --stages l0,l1,l2,l3 --out artifacts/v28_rsac/chicago_full

Post-gate mixed-shock production (only after the same run passes L0-L3)::

    python scripts/v28_learner_ladder.py --city chicago --budget full \
      --stages l0,l1,l2,l3,mixed --out artifacts/v28_rsac/chicago_full_mixed
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
try:
    from scipy.stats import t as student_t
except ImportError:  # pragma: no cover - exercised only in minimal deployments
    student_t = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from calibration_utils import load_calibration_json  # noqa: E402
from continuous_benchmarks import (  # noqa: E402
    CEMConfig,
    FixedPolicy,
    RecedingHorizonCEM,
    PublicSchedulePolicy,
    evaluate_clairvoyant_event_schedule,
    evaluate_policy,
    evaluate_recurrent_agent,
    evaluate_uncapped_price_tail,
    optimize_public_schedule,
    optimize_clairvoyant_event_action,
    optimize_static_policy,
    optimize_static_tariff,
    paired_difference,
)
from continuous_market_env import (  # noqa: E402
    ContinuousMarketEnv,
    DailyEnvironmentConfig,
    build_v28_core,
)
from market_shocks import ShockConfig, ShockEvent, ShockScenario  # noqa: E402
from recurrent_sac import (  # noqa: E402
    ActionTransform,
    Episode,
    EpisodeBuilder,
    NumericalPolicyActionError,
    RecurrentSACAgent,
    RecurrentSACConfig,
)


NUMERICAL_POLICY_PENALTY = -1.0e6


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_torch_payload(path: Path, device: Optional[str] = None) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location=device or "cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.load(path, map_location=device or "cpu")


def load_committed_economic_incumbent(
    directory: str | Path,
    *,
    generation: str = "current",
    device: Optional[str] = None,
) -> Tuple[RecurrentSACAgent, Dict[str, Any]]:
    """Resolve and verify only a manifest-committed incumbent generation."""
    root = Path(directory)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "ride_response_economic_incumbent_v1":
        raise ValueError("unsupported economic-incumbent manifest")
    if generation not in {"current", "previous"}:
        raise ValueError("generation must be 'current' or 'previous'")
    entry = manifest.get(generation)
    if not isinstance(entry, dict):
        raise FileNotFoundError(f"manifest has no {generation} incumbent")
    filename = str(entry.get("filename", ""))
    if not filename or Path(filename).name != filename:
        raise ValueError("incumbent manifest contains an unsafe filename")
    artifact = root / filename
    actual_sha = _sha256_file(artifact)
    if actual_sha != str(entry.get("full_file_sha256", "")):
        raise ValueError("economic-incumbent checkpoint SHA256 mismatch")
    payload = _load_torch_payload(artifact, device=device)
    embedded = dict(
        dict(payload.get("metadata", {}) or {}).get(
            "economic_incumbent_artifact", {}
        )
        or {}
    )
    payload_incumbent = dict(payload.get("economic_incumbent", {}) or {})
    expected_capture = int(entry["capture_count"])
    expected_label = str(entry["label"])
    expected_policy_hash = str(entry["policy_normalizer_sha256"])
    expected_identity_indices = tuple(
        int(index) for index in entry.get("observation_identity_indices", ())
    )
    if (
        int(embedded.get("capture_count", -1)) != expected_capture
        or str(embedded.get("label", "")) != expected_label
        or str(embedded.get("policy_normalizer_sha256", ""))
        != expected_policy_hash
        or int(payload_incumbent.get("capture_count", -1)) != expected_capture
        or str(payload_incumbent.get("label", "")) != expected_label
        or tuple(
            int(index)
            for index in embedded.get("observation_identity_indices", ())
        )
        != expected_identity_indices
        or tuple(
            int(index)
            for index in dict(payload.get("config", {}) or {}).get(
                "observation_identity_indices", ()
            )
        )
        != expected_identity_indices
    ):
        raise ValueError("economic-incumbent embedded contract mismatch")
    agent = RecurrentSACAgent.load(artifact, device=device)
    if (
        not agent.config.observation_identity_indices
        or tuple(agent.config.observation_identity_indices)
        != expected_identity_indices
    ):
        raise ValueError(
            "v28 incumbent checkpoint has no categorical observation identity mask"
        )
    contract = agent.economic_incumbent_contract()
    if (
        int(contract["capture_count"]) != expected_capture
        or str(contract["label"]) != expected_label
    ):
        raise ValueError("loaded economic incumbent does not match manifest")
    verified = copy.deepcopy(entry)
    verified.update(
        {
            "artifact": str(artifact),
            "generation": generation,
            "manifest": str(manifest_path),
            "verified": True,
        }
    )
    return agent, verified


class GeometricActionCommitment:
    """Repeat a sampled behavior action for a geometric number of days."""

    def __init__(self, mean_days: float, rng: np.random.Generator) -> None:
        if not np.isfinite(mean_days) or mean_days < 1.0:
            raise ValueError("commitment mean must be finite and at least one day")
        self.mean_days = float(mean_days)
        self.rng = rng
        self.action: Optional[np.ndarray] = None
        self.coordinates: Optional[np.ndarray] = None
        self.duration = 0
        self.remaining = 0

    @property
    def needs_action(self) -> bool:
        return self.remaining <= 0

    def begin(
        self,
        action: np.ndarray | Sequence[float],
        coordinates: np.ndarray | Sequence[float],
    ) -> int:
        probability = 1.0 / self.mean_days
        self.duration = (
            1 if probability >= 1.0 else int(self.rng.geometric(probability))
        )
        self.remaining = self.duration
        self.action = np.asarray(action, dtype=np.float64).copy()
        self.coordinates = np.asarray(coordinates, dtype=np.float64).copy()
        return self.duration

    def take(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        if self.needs_action or self.action is None or self.coordinates is None:
            raise RuntimeError("commitment has no active action")
        repeated = self.remaining < self.duration
        self.remaining -= 1
        return (
            self.action.copy(),
            self.coordinates.copy(),
            {
                "duration": int(self.duration),
                "repeated": bool(repeated),
                "remaining_after_step": int(self.remaining),
            },
        )


STAGES = ("l0", "l1", "l2", "l3", "mixed")
STAGE_OFFSETS = {
    "l0": 0,
    "l1": 100_000,
    "l2": 200_000,
    "l3": 300_000,
    "mixed": 400_000,
}


@dataclass(frozen=True)
class Budget:
    train_episodes: int
    episode_days: int
    warmup_episodes: int
    evaluation_seeds: int
    benchmark_train_seeds: int
    customers_per_block: int
    profile_pool_size: int
    batch_size: int
    burn_in: int
    learn_length: int
    updates_per_day: float
    cem_population: int
    cem_iterations: int


BUDGETS = {
    "smoke": Budget(
        train_episodes=3,
        episode_days=12,
        warmup_episodes=1,
        evaluation_seeds=2,
        benchmark_train_seeds=2,
        customers_per_block=30,
        profile_pool_size=512,
        batch_size=4,
        burn_in=2,
        learn_length=4,
        updates_per_day=0.25,
        cem_population=6,
        cem_iterations=2,
    ),
    "pilot": Budget(
        train_episodes=24,
        episode_days=30,
        warmup_episodes=3,
        evaluation_seeds=8,
        benchmark_train_seeds=3,
        customers_per_block=80,
        profile_pool_size=5_000,
        batch_size=32,
        burn_in=6,
        learn_length=16,
        updates_per_day=0.50,
        cem_population=20,
        cem_iterations=3,
    ),
    "full": Budget(
        train_episodes=160,
        episode_days=60,
        warmup_episodes=8,
        evaluation_seeds=30,
        benchmark_train_seeds=6,
        customers_per_block=100,
        profile_pool_size=20_000,
        batch_size=64,
        burn_in=8,
        learn_length=24,
        updates_per_day=1.0,
        cem_population=32,
        cem_iterations=5,
    ),
}

ACTOR_ACCEPTANCE_SEEDS_PER_EPISODE = {
    "smoke": 4,
    "pilot": 8,
    "full": 12,
}


def _student_t_critical(probability: float, degrees_of_freedom: int) -> float:
    """Student-t quantile with a fail-safe mpmath fallback."""
    probability = float(probability)
    degrees = int(degrees_of_freedom)
    if not 0.5 < probability < 1.0 or degrees <= 0:
        raise ValueError("Student-t quantile inputs are invalid")
    if student_t is not None:
        return float(student_t.ppf(probability, df=degrees))
    import mpmath

    def cdf(value: float) -> float:
        x = float(value)
        beta_argument = degrees / (degrees + x * x)
        tail = 0.5 * mpmath.betainc(
            degrees / 2.0,
            0.5,
            0.0,
            beta_argument,
            regularized=True,
        )
        return float(1.0 - tail)

    low, high = 0.0, 1.0
    while cdf(high) < probability:
        high *= 2.0
        if high > 1.0e6:
            raise RuntimeError("Student-t quantile failed to bracket")
    for _ in range(100):
        middle = 0.5 * (low + high)
        if cdf(middle) < probability:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _scenario_factory(stage: str, episode_days: int):
    public_onset = max(3, int(episode_days) // 3)
    start = dt.date(2023, 1, 9)
    if stage == "l1":
        def factory(seed: int) -> ShockScenario:
            del seed
            return ShockScenario(
                start_date=start,
                events=[
                    ShockEvent(
                        "winter_storm",
                        public_onset,
                        min(public_onset + 1, episode_days - 1),
                        intensity=1.0,
                        detail={"transit_suspended": True},
                    )
                ],
                forecasts={
                    public_onset: True,
                    min(public_onset + 1, episode_days - 1): True,
                },
            )
        return factory
    if stage == "l2":
        def factory(seed: int) -> ShockScenario:
            # Timing, duration, and severity vary independently by episode.
            # None is exposed to the actor, so time-to-go cannot serve as a
            # countdown to a fixed latent shortage.
            rng = np.random.default_rng([int(seed), 28_002])
            intensity = float(rng.uniform(0.50, 0.78))
            duration = int(rng.integers(2, 5))
            low = min(max(2, int(episode_days) // 6), max(0, episode_days - 1))
            high = max(low, int(episode_days) - duration - 2)
            onset = int(rng.integers(low, high + 1))
            return ShockScenario(
                start_date=start,
                events=[
                    ShockEvent(
                        "driver_shortage",
                        onset,
                        min(onset + duration - 1, episode_days - 1),
                        intensity=intensity,
                        announced=False,
                    )
                ],
            )
        return factory
    return None


def _shock_config(stage: str, calibration: Mapping[str, Any]) -> ShockConfig:
    common = {
        "emergency_surge_cap": 0.0,
        "conditional_sampling": True,
    }
    if stage == "mixed":
        # Production uses every configured stochastic event and the natural
        # empirical calendar.  Shock-proximate openings remain enabled so rare
        # regimes are represented during training, never during held-out test.
        return ShockConfig.from_calibration(
            calibration,
            enabled=True,
            **common,
        )
    isolated = {
        **common,
        "training_shock_opening_share": 0.0,
    }
    if stage in {"l0", "l3"}:
        return ShockConfig.from_calibration(
            calibration, enabled=False, types=(), **isolated
        )
    if stage == "l1":
        return ShockConfig.from_calibration(
            calibration,
            enabled=True,
            types=("winter_storm",),
            storm_daily_hazard={},
            **isolated,
        )
    if stage == "l2":
        return ShockConfig.from_calibration(
            calibration,
            enabled=True,
            types=("driver_shortage",),
            shortage_rate_per_day=0.0,
            **isolated,
        )
    raise ValueError(f"unsupported isolated learner rung {stage!r}")


def _dataset_paths(city: str) -> Tuple[Path, Path, Path]:
    calibration = PROJECT_ROOT / "data" / "processed" / city / "calibration.json"
    trip_bank = PROJECT_ROOT / "data" / "processed" / city / "trip_bank.csv"
    profiles = (
        PROJECT_ROOT
        / "artifacts"
        / "v28_thresholds"
        / city
        / "native_uncapped_primary.jsonl"
    )
    return calibration, trip_bank, profiles


def _episode_seeds(seed: int, stage: str, count: int, split: str) -> List[int]:
    split_offset = {
        "train": 0,
        "validation_fit": 10_000_000,
        "validation_gate": 15_000_000,
        "test": 20_000_000,
        "tail": 30_000_000,
        "train_warm_fit": 35_000_000,
        "warm": 40_000_000,
        "train_critic_gate": 45_000_000,
        "train_selection": 50_000_000,
        "train_actor_accept": 55_000_000,
        "train_absolute_fit": 60_000_000,
        "train_critic_rank": 65_000_000,
        "train_critic_absolute_gate": 70_000_000,
    }[split]
    base = int(seed) + STAGE_OFFSETS[stage] + split_offset
    return [base + 104_729 * index for index in range(int(count))]


def _observation_identity_indices(
    observation_names: Sequence[str],
) -> Tuple[int, ...]:
    """Return schema-derived indicators that must remain in natural units.

    Weather is one-hot and every public shock feature is bounded in [0, 1]
    by the environment contract (including proximity and transit
    availability).  Bypassing RMS for the complete named families also makes
    newly added public binary flags fail closed into the same treatment.
    """
    names = tuple(str(name) for name in observation_names)
    indices = tuple(
        index
        for index, name in enumerate(names)
        if name.startswith("weather_") or name.startswith("public_")
    )
    if not indices:
        raise ValueError("v28 observation schema has no weather/public indicators")
    return indices


def _critic_probe_days(episode_days: int) -> Tuple[int, ...]:
    """Fixed preregistered critic probe strata, scaled only for smoke horizons."""
    horizon = int(episode_days)
    if horizon <= 0:
        raise ValueError("critic probe horizon must be positive")
    nominal = np.asarray((0, 10, 19, 29), dtype=np.float64)
    if horizon >= 30:
        return tuple(int(day) for day in nominal)
    scaled = np.rint(nominal * float(horizon - 1) / 29.0).astype(int)
    return tuple(dict.fromkeys(int(day) for day in scaled))


class EpisodeActorAcceptance:
    """TRAIN-only paired-CRN safe policy improvement at episode boundaries."""

    def __init__(
        self,
        agent: RecurrentSACAgent,
        environment_factory,
        seed_blocks: Sequence[Sequence[int]],
        critic_gate_runtime: Dict[str, Any],
        *,
        stage: str = "l0",
        forbidden_seeds: Sequence[int] = (),
        familywise_alpha: float = 0.05,
        incumbent_checkpoint_directory: Optional[Path] = None,
    ) -> None:
        blocks = tuple(tuple(int(seed) for seed in block) for block in seed_blocks)
        if not blocks or any(len(block) < 2 for block in blocks):
            raise ValueError("actor acceptance needs at least two seeds per episode")
        if len({len(block) for block in blocks}) != 1:
            raise ValueError("actor acceptance seed blocks must have equal size")
        flattened = [seed for block in blocks for seed in block]
        if len(flattened) != len(set(flattened)):
            raise ValueError("actor acceptance seeds may never be reused")
        if not set(flattened).isdisjoint({int(seed) for seed in forbidden_seeds}):
            raise ValueError("actor acceptance seeds must be disjoint from all other splits")
        alpha = float(familywise_alpha)
        if not np.isfinite(alpha) or not 0.0 < alpha < 0.5:
            raise ValueError("familywise alpha must lie in (0, 0.5)")
        self.agent = agent
        self.stage = str(stage)
        if self.stage not in STAGES:
            raise ValueError(f"unknown actor-acceptance stage {self.stage!r}")
        self.environment_factory = environment_factory
        self.seed_blocks = blocks
        self.critic_gate_runtime = critic_gate_runtime
        self.familywise_alpha = alpha
        self.per_look_alpha = alpha / len(blocks)
        self.degrees_of_freedom = len(blocks[0]) - 1
        self.critical_value = _student_t_critical(
            1.0 - self.per_look_alpha,
            self.degrees_of_freedom,
        )
        if not np.isfinite(self.critical_value):
            raise ValueError("actor acceptance Student-t critical value is non-finite")
        self.events: List[Dict[str, Any]] = []
        self.consumed_seeds: set[int] = set()
        self.incumbent_pass_streak = int(critic_gate_runtime["pass_streak"])
        self.proposed_actor_updates = 0
        self.incumbent_checkpoint_directory = (
            None
            if incumbent_checkpoint_directory is None
            else Path(incumbent_checkpoint_directory)
        )
        self.incumbent_checkpoints: List[str] = []
        self.incumbent_checkpoint_history: List[Dict[str, Any]] = []
        incumbent_label = (
            "l0_warm_economic_incumbent"
            if self.stage == "l0"
            else f"{self.stage}_entering_economic_incumbent"
        )
        if self.agent.economic_incumbent_contract()["available"]:
            self.agent.begin_economic_stage(incumbent_label)
        else:
            self.agent.initialize_economic_incumbent(incumbent_label)
        warm_authorization = critic_gate_runtime.get("warm_gate_authorization")
        if not isinstance(warm_authorization, dict):
            # Lightweight tests may construct the runtime directly.  The
            # production warm-start always provides the full warm gate row.
            warm_authorization = {
                "update": 0,
                "passed": False,
                "gate_passed_current_version": bool(self.incumbent_pass_streak),
                "consecutive_passes": int(self.incumbent_pass_streak),
                "actor_temperature_authorized_this_update": False,
                "source": "synthetic_warm_gate_contract",
            }
        self.incumbent_gate_authorization = copy.deepcopy(warm_authorization)
        self.incumbent_gate_restored_after_rejection = False
        self._persist_incumbent("warm_reference")

    def _latest_passing_gate_authorization(self) -> Dict[str, Any]:
        if not self.critic_gate_runtime["events"]:
            return copy.deepcopy(self.incumbent_gate_authorization)
        latest = self.critic_gate_runtime["events"][-1]
        if not (
            bool(latest.get("passed", False))
            and bool(latest.get("gate_passed_current_version", False))
        ):
            raise RuntimeError(
                "an economic incumbent may advance only from the exact latest "
                "passing same-version gate"
            )
        return copy.deepcopy(latest)

    def _advance_incumbent_gate_authorization(self) -> None:
        self.incumbent_gate_authorization = (
            self._latest_passing_gate_authorization()
        )
        self.incumbent_gate_restored_after_rejection = False

    def current_incumbent_gate(self) -> Dict[str, Any]:
        current = copy.deepcopy(self.incumbent_gate_authorization)
        contract = self.agent.economic_incumbent_contract()
        current.update(
            {
                "incumbent_label": contract["label"],
                "incumbent_capture_count": int(contract["capture_count"]),
                "accepted_actor_blocks": int(contract["accepted_actor_blocks"]),
                "restored_after_rejection": bool(
                    self.incumbent_gate_restored_after_rejection
                ),
                "is_current_economic_incumbent_gate": True,
            }
        )
        return current

    def _persist_incumbent(self, reason: str) -> Optional[str]:
        """Durably save each economically accepted actor transaction."""
        if self.incumbent_checkpoint_directory is None:
            return None
        root = self.incumbent_checkpoint_directory
        root.mkdir(parents=True, exist_ok=True)
        contract = self.agent.economic_incumbent_contract()
        sequence = int(contract["capture_count"])
        transaction = self.agent.economic_incumbent_transaction_state()
        policy_digest = hashlib.sha256()
        for name, tensor in sorted(transaction["learner"]["actor"].items()):
            policy_digest.update(name.encode("utf-8"))
            policy_digest.update(tensor.detach().cpu().numpy().tobytes())
        normalizer = transaction["learner"]["observation_normalizer"]
        identity_mask = np.asarray(
            normalizer.get("identity_mask", ()), dtype=np.uint8
        )
        if identity_mask.shape != np.asarray(normalizer["mean"]).shape:
            raise RuntimeError("incumbent normalizer identity-mask schema mismatch")
        identity_indices = tuple(
            int(index) for index in self.agent.config.observation_identity_indices
        )
        if not identity_indices or not np.array_equal(
            np.flatnonzero(identity_mask), np.asarray(identity_indices)
        ):
            raise RuntimeError("incumbent normalizer identity mask is not v28 schema-bound")
        policy_digest.update(b"ride_response_observation_normalizer_v2_identity")
        policy_digest.update(np.asarray(normalizer["mean"]).tobytes())
        policy_digest.update(np.asarray(normalizer["var"]).tobytes())
        policy_digest.update(
            np.asarray([normalizer["count"]], dtype=np.float64).tobytes()
        )
        policy_digest.update(identity_mask.tobytes())
        policy_digest.update(
            np.asarray(identity_indices, dtype=np.int64).tobytes()
        )
        policy_hash = policy_digest.hexdigest()
        filename = f"incumbent_{sequence:04d}_{policy_hash[:16]}.pt"
        destination = root / filename
        temporary = root / f"{filename}.tmp"
        if destination.exists():
            raise FileExistsError(
                f"immutable incumbent checkpoint already exists: {destination}"
            )
        embedded = {
            "capture_count": sequence,
            "label": str(contract["label"]),
            "accepted_actor_blocks": int(contract["accepted_actor_blocks"]),
            "policy_normalizer_sha256": policy_hash,
            "observation_identity_indices": list(identity_indices),
        }
        self.agent.save(
            temporary,
            metadata={
                "artifact_role": "economically_verified_incumbent",
                "reason": str(reason),
                "incumbent": contract,
                "economic_incumbent_artifact": embedded,
                "validation_used": False,
                "test_used": False,
            },
        )
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(root)
        saved_payload = _load_torch_payload(destination)
        saved_embedded = dict(
            dict(saved_payload.get("metadata", {}) or {}).get(
                "economic_incumbent_artifact", {}
            )
            or {}
        )
        saved_incumbent = dict(saved_payload.get("economic_incumbent", {}) or {})
        if (
            saved_embedded != embedded
            or int(saved_incumbent.get("capture_count", -1)) != sequence
            or str(saved_incumbent.get("label", "")) != str(contract["label"])
        ):
            raise RuntimeError("saved incumbent failed embedded contract verification")
        full_file_sha = _sha256_file(destination)
        manifest = root / "manifest.json"
        old_manifest: Dict[str, Any] = {}
        if manifest.exists():
            old_manifest = json.loads(manifest.read_text())
            if old_manifest.get("schema") != "ride_response_economic_incumbent_v1":
                raise ValueError("unsupported existing incumbent manifest")
        entry = {
            "filename": filename,
            "capture_count": sequence,
            "label": str(contract["label"]),
            "accepted_actor_blocks": int(contract["accepted_actor_blocks"]),
            "reason": str(reason),
            "policy_normalizer_sha256": policy_hash,
            "observation_identity_indices": list(identity_indices),
            "full_file_sha256": full_file_sha,
        }
        old_history = list(old_manifest.get("history", []))
        new_history = [*old_history, copy.deepcopy(entry)]
        new_manifest = {
            "schema": "ride_response_economic_incumbent_v1",
            "current": copy.deepcopy(entry),
            "previous": copy.deepcopy(old_manifest.get("current")),
            "history": new_history,
        }
        manifest_temporary = root / "manifest.json.tmp"
        with manifest_temporary.open("w") as handle:
            json.dump(new_manifest, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(manifest_temporary, manifest)
        _fsync_directory(root)

        retained_filenames = {
            str(row["filename"])
            for row in (new_manifest["current"], new_manifest["previous"])
            if isinstance(row, dict)
        }
        for candidate in root.glob("incumbent_*.pt"):
            if candidate.name not in retained_filenames:
                candidate.unlink()
        _fsync_directory(root)
        self.incumbent_checkpoint_history = new_history
        self.incumbent_checkpoints = [
            str(root / name) for name in sorted(retained_filenames)
        ]
        return str(destination)

    @staticmethod
    def _policy_changed(
        candidate: Mapping[str, Any], incumbent: Mapping[str, Any]
    ) -> bool:
        candidate_learner = candidate["learner"]
        incumbent_learner = incumbent["learner"]
        if not torch_equal_state(
            candidate_learner["actor"], incumbent_learner["actor"]
        ):
            return True
        candidate_normalizer = candidate_learner["observation_normalizer"]
        incumbent_normalizer = incumbent_learner["observation_normalizer"]
        return not (
            np.array_equal(
                np.asarray(candidate_normalizer["mean"]),
                np.asarray(incumbent_normalizer["mean"]),
            )
            and np.array_equal(
                np.asarray(candidate_normalizer["var"]),
                np.asarray(incumbent_normalizer["var"]),
            )
            and float(candidate_normalizer["count"])
            == float(incumbent_normalizer["count"])
        )

    def record_critic_proposal_rejection(
        self,
        episode_index: int,
        *,
        rollback: Mapping[str, Any],
        rejected_update: int,
    ) -> Dict[str, Any]:
        """Audit a gate-rejected episode without spending CPI seeds."""
        index = int(episode_index)
        if index != len(self.events) or index >= len(self.seed_blocks):
            raise RuntimeError("critic rejection episodes must be recorded in order")
        self.critic_gate_runtime["pass_streak"] = self.incumbent_pass_streak
        self.incumbent_gate_restored_after_rejection = True
        if self.critic_gate_runtime["events"]:
            latest = self.critic_gate_runtime["events"][-1]
            latest["terminal_on_failure"] = False
            latest["failure_disposition"] = (
                "reject_episode_restore_economic_incumbent"
            )
        row = {
            "episode": index,
            "decision": "critic_gate_rejected_restore_incumbent",
            "accepted": False,
            "actor_proposal": False,
            "critic_proposal_rejection": True,
            "rejected_update": int(rejected_update),
            "acceptance_seeds": [],
            "reserved_acceptance_seed_block_not_consumed": list(
                self.seed_blocks[index]
            ),
            "seeds_consumed": False,
            "split": "heldout_train_critic_gate",
            "learned_from": False,
            "added_to_replay": False,
            "paired_lcb": None,
            "accepted_actor_blocks": int(self.agent.accepted_actor_blocks),
            "rollback": copy.deepcopy(dict(rollback)),
            "critic_gate_pass_streak_restored": int(
                self.critic_gate_runtime["pass_streak"]
            ),
            "incumbent": self.agent.economic_incumbent_contract(),
            "current_incumbent_gate": self.current_incumbent_gate(),
        }
        self.events.append(row)
        return copy.deepcopy(row)

    def _evaluate_transaction(
        self, transaction: Mapping[str, Any], seeds: Sequence[int]
    ) -> Dict[str, Any]:
        self.agent.load_training_transaction_state(transaction)
        try:
            result = evaluate_recurrent_agent(
                self.environment_factory,
                self.agent,
                seeds,
                deterministic=True,
            )
        finally:
            # Evaluation-side action diagnostics must never enter training.
            self.agent.load_training_transaction_state(transaction)
        return result

    def evaluate_episode(
        self,
        episode_index: int,
        *,
        proposal_numerically_valid: bool,
    ) -> Dict[str, Any]:
        index = int(episode_index)
        if index != len(self.events) or index >= len(self.seed_blocks):
            raise RuntimeError("actor acceptance episodes must be evaluated exactly once in order")
        block = self.seed_blocks[index]
        candidate = self.agent.training_transaction_state()
        incumbent = self.agent.economic_incumbent_transaction_state()
        candidate_pass_streak = int(self.critic_gate_runtime["pass_streak"])
        policy_changed = self._policy_changed(candidate, incumbent)
        candidate_actor_updates = int(
            candidate["learner"]["actor_updates_applied"]
        )
        incumbent_actor_updates = int(
            incumbent["learner"]["actor_updates_applied"]
        )
        proposal_update_count = max(
            0, candidate_actor_updates - incumbent_actor_updates
        )
        self.proposed_actor_updates += proposal_update_count

        if not policy_changed:
            self.agent.load_training_transaction_state(candidate)
            incumbent_contract = self.agent.advance_economic_incumbent_without_actor_change(
                f"episode_{index}_critic_only_advance"
            )
            self.incumbent_pass_streak = candidate_pass_streak
            self._advance_incumbent_gate_authorization()
            row = {
                "episode": index,
                "decision": "critic_only_advance",
                "accepted": True,
                "actor_proposal": False,
                "acceptance_seeds": list(block),
                "seeds_consumed": False,
                "paired_lcb": None,
                "accepted_actor_blocks": int(self.agent.accepted_actor_blocks),
                "incumbent": incumbent_contract,
                "current_incumbent_gate": self.current_incumbent_gate(),
            }
            self.events.append(row)
            return copy.deepcopy(row)

        numerical_reject = not bool(proposal_numerically_valid)
        candidate_evaluation: Optional[Dict[str, Any]] = None
        incumbent_evaluation: Optional[Dict[str, Any]] = None
        numerical_reason: Optional[str] = (
            "training episode contained a numerical-invalid action"
            if numerical_reject
            else None
        )
        if not numerical_reject:
            try:
                candidate_evaluation = self._evaluate_transaction(candidate, block)
            except (NumericalPolicyActionError, FloatingPointError, OverflowError) as exc:
                numerical_reject = True
                numerical_reason = f"{type(exc).__name__}: {exc}"
        if not numerical_reject:
            incumbent_evaluation = self._evaluate_transaction(incumbent, block)
            candidate_values = np.asarray(
                candidate_evaluation["values"], dtype=np.float64
            )
            incumbent_values = np.asarray(
                incumbent_evaluation["values"], dtype=np.float64
            )
            finite = bool(
                np.all(np.isfinite(candidate_values))
                and np.all(np.isfinite(incumbent_values))
            )
            if finite:
                difference = paired_difference(
                    candidate_evaluation, incumbent_evaluation
                )
                mean = float(difference["mean"])
                standard_error = float(difference["standard_error"])
                lcb = (
                    mean - self.critical_value * standard_error
                    if np.isfinite(standard_error)
                    else float("-inf")
                )
            else:
                numerical_reject = True
                numerical_reason = "non-finite paired acceptance return"
                difference = None
                lcb = float("-inf")
        else:
            difference = None
            lcb = float("-inf")

        self.consumed_seeds.update(block)
        accepted = bool(not numerical_reject and np.isfinite(lcb) and lcb > 0.0)
        if accepted:
            self.agent.load_training_transaction_state(candidate)
            self.critic_gate_runtime["pass_streak"] = candidate_pass_streak
            incumbent_contract = self.agent.accept_economic_incumbent(
                f"episode_{index}_positive_lcb"
            )
            self._advance_incumbent_gate_authorization()
            incumbent_checkpoint = self._persist_incumbent(
                f"episode_{index}_positive_lcb"
            )
            self.incumbent_pass_streak = candidate_pass_streak
            rollback = None
            decision = "accepted_positive_multiplicity_corrected_lcb"
        else:
            rollback = self.agent.restore_economic_incumbent(
                numerical_reason or "paired actor gain LCB was not strictly positive"
            )
            self.critic_gate_runtime["pass_streak"] = self.incumbent_pass_streak
            incumbent_contract = self.agent.economic_incumbent_contract()
            decision = "rejected_restore_incumbent"
            incumbent_checkpoint = None
        row = {
            "episode": index,
            "decision": decision,
            "accepted": accepted,
            "actor_proposal": True,
            "proposal_actor_updates": int(proposal_update_count),
            "acceptance_seeds": list(block),
            "seeds_consumed": True,
            "split": "train_actor_accept",
            "learned_from": False,
            "added_to_replay": False,
            "familywise_alpha": float(self.familywise_alpha),
            "number_of_preregistered_looks": int(len(self.seed_blocks)),
            "per_look_bonferroni_alpha": float(self.per_look_alpha),
            "one_sided_student_t_critical_value": float(self.critical_value),
            "student_t_degrees_of_freedom": int(self.degrees_of_freedom),
            "paired_candidate_minus_incumbent": difference,
            "paired_lcb": float(lcb),
            "numerical_reject": bool(numerical_reject),
            "numerical_reject_reason": numerical_reason,
            "accepted_actor_blocks": int(self.agent.accepted_actor_blocks),
            "anchor_decay_counter": "accepted_actor_blocks",
            "rollback": rollback,
            "critic_gate_pass_streak_restored": int(
                self.critic_gate_runtime["pass_streak"]
            ),
            "incumbent": incumbent_contract,
            "current_incumbent_gate": self.current_incumbent_gate(),
            "incumbent_checkpoint": incumbent_checkpoint,
        }
        self.events.append(row)
        return copy.deepcopy(row)

    def summary(self) -> Dict[str, Any]:
        actor_events = [row for row in self.events if row["actor_proposal"]]
        accepted = [row for row in actor_events if row["accepted"]]
        rejected = [row for row in actor_events if not row["accepted"]]
        return {
            "stage": self.stage,
            "split": "train_actor_accept",
            "train_only": True,
            "preregistered_seed_blocks": [list(block) for block in self.seed_blocks],
            "seed_reuse": False,
            "familywise_alpha": float(self.familywise_alpha),
            "per_look_bonferroni_alpha": float(self.per_look_alpha),
            "one_sided_student_t_critical_value": float(self.critical_value),
            "student_t_degrees_of_freedom": int(self.degrees_of_freedom),
            "acceptance_blocks_consumed": int(
                sum(bool(row.get("seeds_consumed")) for row in self.events)
            ),
            "acceptance_seed_count_consumed": int(len(self.consumed_seeds)),
            "actor_proposals": int(len(actor_events)),
            "accepted_actor_proposals": int(len(accepted)),
            "rejected_actor_proposals": int(len(rejected)),
            "proposed_actor_updates": int(self.proposed_actor_updates),
            "accepted_actor_blocks": int(self.agent.accepted_actor_blocks),
            "positive_actor_gain_reported": bool(accepted),
            "events": copy.deepcopy(self.events),
            "incumbent": self.agent.economic_incumbent_contract(),
            "current_incumbent_gate": self.current_incumbent_gate(),
            "incumbent_checkpoints": list(self.incumbent_checkpoints),
            "incumbent_checkpoint_history": copy.deepcopy(
                self.incumbent_checkpoint_history
            ),
            "validation_used": False,
            "test_used": False,
            "acceptance_episodes_added_to_replay": False,
        }


def torch_equal_state(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Exact equality for module state dictionaries without importing test helpers."""
    if set(left) != set(right):
        return False
    return all(
        bool(
            np.array_equal(
                left[key].detach().cpu().numpy(),
                right[key].detach().cpu().numpy(),
            )
        )
        for key in left
    )


def _l0_learner_readiness(
    current_incumbent_gate: Mapping[str, Any],
    actor_gain_report: Mapping[str, Any],
) -> bool:
    nonnumerical_actor_proposal = any(
        row.get("actor_proposal", False)
        and not row.get("numerical_reject", False)
        for row in actor_gain_report["events"]
    )
    return bool(
        current_incumbent_gate.get("passed", False)
        and current_incumbent_gate.get(
            "actor_temperature_authorized_this_update", False
        )
        and actor_gain_report["proposed_actor_updates"] > 0
        and nonnumerical_actor_proposal
    )


def _adaptive_stage_learner_readiness(
    current_incumbent_gate: Mapping[str, Any],
    actor_gain_report: Mapping[str, Any],
) -> bool:
    """Later rungs require an economically accepted adaptive actor update."""
    return bool(
        _l0_learner_readiness(current_incumbent_gate, actor_gain_report)
        and actor_gain_report.get("positive_actor_gain_reported", False)
        and int(actor_gain_report.get("accepted_actor_proposals", 0)) > 0
    )


def train_stage(
    agent: RecurrentSACAgent,
    environment_factory,
    seeds: Sequence[int],
    *,
    warmup_episodes: int,
    updates_per_day: float,
    seed: int,
    exploration_commitment_mean_days: float = 3.0,
    warmup_center_coordinates: Optional[np.ndarray | Sequence[float]] = None,
    warmup_coordinate_std: Optional[np.ndarray | Sequence[float]] = None,
    actor_gate_recheck=None,
    actor_acceptance: Optional[EpisodeActorAcceptance] = None,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    logs: List[Dict[str, Any]] = []
    update_credit = 0.0
    for episode_index, episode_seed in enumerate(seeds):
        training_rng_before_episode = copy.deepcopy(rng.bit_generator.state)
        update_credit_before_episode = float(update_credit)
        environment = environment_factory()
        observation = environment.reset(int(episode_seed))
        builder = EpisodeBuilder(observation)
        hidden = None
        previous_action = None
        previous_reward = 0.0
        episode_return = 0.0
        extreme_days = 0
        shock_days = 0
        numerical_invalid_actions = 0
        commitment_durations: List[int] = []
        commitment_repeat_days = 0
        commitment = GeometricActionCommitment(
            exploration_commitment_mean_days, rng
        )
        done = False
        days = 0
        started = time.time()
        while not done:
            if commitment.needs_action:
                try:
                    if episode_index < int(warmup_episodes):
                        center = np.zeros(agent.action_dim, dtype=np.float64)
                        if warmup_center_coordinates is not None:
                            center = np.asarray(
                                warmup_center_coordinates, dtype=np.float64
                            ).reshape(agent.action_dim)
                        spread = np.full(agent.action_dim, 0.45, dtype=np.float64)
                        if warmup_coordinate_std is not None:
                            spread = np.asarray(
                                warmup_coordinate_std, dtype=np.float64
                            ).reshape(agent.action_dim)
                        if (
                            not np.all(np.isfinite(center))
                            or not np.all(np.isfinite(spread))
                            or np.any(spread <= 0.0)
                        ):
                            raise ValueError("warmup exploration center/std is invalid")
                        coordinates = rng.normal(center, spread)
                        # Native-coordinate Gaussian exploration is uncapped;
                        # invalid exponentiation is rejected, never clipped.
                        action = agent.action_transform.coordinates_to_physical(
                            coordinates
                        )
                        action_info = {
                            "extreme_action": bool(
                                np.any(np.abs(coordinates) > 4.0)
                            )
                        }
                    else:
                        action, hidden, action_info = agent.act(
                            observation,
                            previous_action=previous_action,
                            previous_reward=previous_reward,
                            hidden=hidden,
                            deterministic=False,
                        )
                        coordinates = np.asarray(
                            action_info["policy_coordinates"], dtype=np.float64
                        )
                except NumericalPolicyActionError as exc:
                    numerical_invalid_actions += 1
                    builder.append(
                        exc.coordinates,
                        NUMERICAL_POLICY_PENALTY,
                        observation,
                        True,
                    )
                    episode_return += NUMERICAL_POLICY_PENALTY
                    days += 1
                    done = True
                    break
                commitment_durations.append(
                    commitment.begin(action, coordinates)
                )
            else:
                if episode_index >= int(warmup_episodes):
                    hidden = agent.advance_actor_hidden(
                        observation,
                        previous_action=previous_action,
                        previous_reward=previous_reward,
                        hidden=hidden,
                    )
                action_info = {"extreme_action": False}
            action, coordinates, commitment_info = commitment.take()
            commitment_repeat_days += int(commitment_info["repeated"])
            try:
                next_observation, reward, done, info = environment.step(action)
            except (FloatingPointError, OverflowError) as exc:
                # The invalid action is recorded in the policy's native
                # coordinates and receives a terminal penalty.  It is neither
                # clipped nor replaced by a feasible action.
                agent.nonfinite_action_rejections += 1
                numerical_invalid_actions += 1
                next_observation = observation.copy()
                reward = NUMERICAL_POLICY_PENALTY
                done = True
                info = {
                    "active_shocks": [],
                    "numerical_invalid_action": True,
                    "numerical_error": f"{type(exc).__name__}: {exc}",
                }
            builder.append(coordinates, reward, next_observation, done)
            observation = next_observation
            previous_action = np.asarray(action, dtype=np.float64)
            previous_reward = float(reward)
            episode_return += float(reward)
            extreme_days += int(bool(action_info.get("extreme_action", False)))
            shock_days += int(bool(info.get("active_shocks")))
            days += 1
        agent.add_episode(builder.finish(), actions_are_physical=False)
        update_credit += float(days) * float(updates_per_day)
        update_rows: List[Dict[str, Any]] = []
        terminal_training_failure = False
        critic_proposal_rejection = False
        while update_credit >= 1.0 and agent.can_update():
            update_result = agent.update(
                critic_gate=actor_gate_recheck,
                restore_economic_incumbent_on_gate_failure=(
                    actor_acceptance is not None
                ),
            )
            update_rows.append(update_result)
            update_credit -= 1.0
            if bool(update_result.get("terminal_training_failure", 0.0)):
                terminal_training_failure = True
                update_credit = 0.0
                break
            if bool(update_result.get("critic_proposal_rejection", 0.0)):
                critic_proposal_rejection = True
                update_credit = update_credit_before_episode
                rng.bit_generator.state = copy.deepcopy(
                    training_rng_before_episode
                )
                gate_row = update_result.get("safe_critic_gate")
                if isinstance(gate_row, dict):
                    gate_row["terminal_on_failure"] = False
                    gate_row["failure_disposition"] = (
                        "reject_episode_restore_economic_incumbent"
                    )
                break
        numeric_keys = (
            [
                key
                for key, value in update_rows[0].items()
                if isinstance(value, (int, float, np.integer, np.floating))
            ]
            if update_rows
            else []
        )
        update_summary = {
            key: float(np.mean([row[key] for row in update_rows]))
            for key in numeric_keys
        } if update_rows else {}
        row = {
            "episode": int(episode_index),
            "seed": int(episode_seed),
            "return": float(episode_return),
            "days": int(days),
            "shock_days": int(shock_days),
            "extreme_action_days": int(extreme_days),
            "numerical_invalid_actions": int(numerical_invalid_actions),
            "commitment_count": int(len(commitment_durations)),
            "commitment_repeat_days": int(commitment_repeat_days),
            "commitment_mean_sampled_days": (
                float(np.mean(commitment_durations))
                if commitment_durations
                else 0.0
            ),
            "replay_transitions": int(len(agent.replay)),
            "updates_this_episode": int(len(update_rows)),
            "elapsed_seconds": float(time.time() - started),
            **update_summary,
            "terminal_training_failure": bool(terminal_training_failure),
            "critic_proposal_rejection": bool(critic_proposal_rejection),
        }
        safe_gate_updates = [
            copy.deepcopy(update["safe_critic_gate"])
            for update in update_rows
            if "safe_critic_gate" in update
        ]
        if safe_gate_updates:
            row["safe_critic_gate_updates"] = safe_gate_updates
            row["safe_critic_gate_recheck"] = safe_gate_updates[-1]
        rollback_rows = [
            copy.deepcopy(update["rollback"])
            for update in update_rows
            if "rollback" in update
        ]
        if rollback_rows:
            row["rollback"] = rollback_rows[-1]
        if actor_acceptance is not None and critic_proposal_rejection:
            rejected_update = int(
                update_rows[-1].get("rejected_update", agent.updates + 1)
            )
            rejection = actor_acceptance.record_critic_proposal_rejection(
                episode_index,
                rollback=update_rows[-1]["rollback"],
                rejected_update=rejected_update,
            )
            row["actor_acceptance"] = rejection
            row["nonterminal_economic_rollback"] = True
            row["training_rng_restored"] = True
            row["update_credit_restored"] = True
            row["acceptance_block_consumed"] = False
            row["replay_transitions"] = int(len(agent.replay))
        elif actor_acceptance is not None and not terminal_training_failure:
            acceptance = actor_acceptance.evaluate_episode(
                episode_index,
                proposal_numerically_valid=(numerical_invalid_actions == 0),
            )
            row["actor_acceptance"] = acceptance
            if acceptance["actor_proposal"] and not acceptance["accepted"]:
                rng.bit_generator.state = copy.deepcopy(
                    training_rng_before_episode
                )
                update_credit = update_credit_before_episode
                row["nonterminal_economic_rollback"] = True
                row["training_rng_restored"] = True
                row["update_credit_restored"] = True
            else:
                row["nonterminal_economic_rollback"] = False
                row["training_rng_restored"] = False
                row["update_credit_restored"] = False
            row["replay_transitions"] = int(len(agent.replay))
        logs.append(row)
        print(
            f"  episode {episode_index + 1:4d}/{len(seeds)} "
            f"return={episode_return:+.3f} shocks={shock_days:2d} "
            f"updates={len(update_rows):3d} temp={row.get('temperature', float('nan')):.3f} "
            f"[{row['elapsed_seconds']:.1f}s]",
            flush=True,
        )
        if terminal_training_failure:
            print(
                "  TERMINAL: same-version critic gate failed; restored the "
                "last complete passing learner state",
                flush=True,
            )
            break
        if critic_proposal_rejection:
            print(
                "  REJECTED: same-version critic proposal missed its TRAIN "
                "gate; restored the economically accepted incumbent and "
                "continued",
                flush=True,
            )
    return logs


def _local_action_support(
    transform: ActionTransform,
    target_coordinates: np.ndarray,
    radius: float | Sequence[float] | np.ndarray,
) -> Dict[str, Any]:
    """Symmetric, full-rank local support without clipping any control."""
    radii = np.broadcast_to(
        np.asarray(radius, dtype=np.float64), (transform.dim,)
    ).copy()
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("local support radii must be finite and positive")
    center = np.asarray(target_coordinates, dtype=np.float64).reshape(transform.dim)
    offsets: List[Tuple[str, np.ndarray]] = [
        ("center", np.zeros(transform.dim, dtype=np.float64))
    ]
    for axis, name in enumerate(transform.names):
        direction = np.zeros(transform.dim, dtype=np.float64)
        direction[axis] = radii[axis]
        offsets.append((f"axis:{name}:+", direction.copy()))
        offsets.append((f"axis:{name}:-", -direction.copy()))
    for mask in range(2 ** transform.dim):
        signs = np.asarray(
            [1.0 if mask & (1 << axis) else -1.0 for axis in range(transform.dim)],
            dtype=np.float64,
        )
        offsets.append(
            (
                "joint:" + ",".join("+" if value > 0 else "-" for value in signs),
                radii * signs,
            )
        )
    rows = []
    for index, (label, offset) in enumerate(offsets):
        coordinates = center + offset
        physical = transform.coordinates_to_physical(coordinates)
        rows.append(
            {
                "index": int(index),
                "label": label,
                "offset": offset.tolist(),
                "coordinates": coordinates.tolist(),
                "action": physical.tolist(),
            }
        )
    matrix = np.asarray([row["offset"] for row in rows[1:]], dtype=np.float64)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    offset_keys = {
        tuple(np.round(np.asarray(row["offset"], dtype=float), 12))
        for row in rows
    }
    symmetric = all(
        tuple(np.round(-np.asarray(row["offset"], dtype=float), 12)) in offset_keys
        for row in rows
    )
    return {
        "radius_coordinates": radii.tolist(),
        "actions": rows,
        "support_rank": int(np.linalg.matrix_rank(matrix)),
        "singular_values": singular_values.tolist(),
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "symmetric": bool(symmetric),
        "unbounded_transform_no_clipping": True,
    }


def _local_deviation_episode(
    environment_factory,
    warm_action: np.ndarray,
    deviation_action: np.ndarray,
    seed: int,
    deviation_day: int,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[Episode, float, np.ndarray]:
    """One local action deviation followed by the fixed warm rule.

    The candidate is used for exactly one specified day and the warm rule is
    used before and after it, so the realized return-to-go at the branch is a
    matched Monte-Carlo target for Q^warm(s_t, a).  Holding every candidate
    for the whole episode would compare different continuation policies.
    """
    environment = environment_factory()
    observation = environment.reset(int(seed))
    reset_observation = np.asarray(observation, dtype=np.float32).copy()
    builder = EpisodeBuilder(observation)
    total = 0.0
    done = False
    day = 0
    while not done:
        action = deviation_action if day == int(deviation_day) else warm_action
        next_observation, reward, done, _ = environment.step(action)
        builder.append(action, reward, next_observation, done)
        observation = next_observation
        total += float(reward)
        day += 1
    episode = builder.finish()
    if deviation_day < 0 or deviation_day >= episode.length:
        raise ValueError("deviation_day lies outside episode")
    episode.metadata = {
        **dict(metadata or {}),
        "seed": int(seed),
        "deviation_day": int(deviation_day),
    }
    return episode, total, reset_observation


def _recurrent_policy_deviation_episode(
    environment_factory,
    agent: RecurrentSACAgent,
    seed: int,
    deviation_day: int,
    offset_coordinates: np.ndarray | Sequence[float],
    metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[Episode, float, np.ndarray]:
    """Run one coordinate perturbation with the entering recurrent policy.

    The actor and its observation normalizer are held fixed while collecting a
    stage's panels.  Candidate and center rows therefore have bit-identical
    public/recurrent prefixes, and after the branch both use the same entering
    policy as their continuation.  Only diagnostic action counters are
    restored; no replay, optimizer, normalizer, or RNG state is learned from
    these rollouts here.
    """
    offset = np.asarray(offset_coordinates, dtype=np.float64).reshape(
        agent.action_dim
    )
    if not np.all(np.isfinite(offset)):
        raise ValueError("stage-local coordinate offset must be finite")
    diagnostic_state = {
        "extreme_action_count": int(agent.extreme_action_count),
        "nonfinite_action_rejections": int(agent.nonfinite_action_rejections),
        "max_abs_coordinate": agent.max_abs_coordinate.copy(),
        "max_physical_action": agent.max_physical_action.copy(),
        "min_physical_action": agent.min_physical_action.copy(),
    }
    try:
        environment = environment_factory()
        observation = environment.reset(int(seed))
        reset_observation = np.asarray(observation, dtype=np.float32).copy()
        builder = EpisodeBuilder(observation)
        hidden = None
        previous_action = None
        previous_reward = 0.0
        previous_done = False
        total = 0.0
        done = False
        day = 0
        branch_center_coordinates = None
        branch_action_coordinates = None
        while not done:
            center_action, next_hidden, action_info = agent.act(
                observation,
                hidden=hidden,
                previous_action=previous_action,
                previous_reward=previous_reward,
                previous_done=previous_done,
                deterministic=True,
            )
            center_coordinates = np.asarray(
                action_info["policy_coordinates"], dtype=np.float64
            )
            action = center_action
            if day == int(deviation_day):
                branch_center_coordinates = center_coordinates.copy()
                branch_action_coordinates = center_coordinates + offset
                action = agent.action_transform.coordinates_to_physical(
                    branch_action_coordinates
                )
            next_observation, reward, done, _ = environment.step(action)
            builder.append(action, reward, next_observation, done)
            observation = next_observation
            hidden = next_hidden
            previous_action = np.asarray(action, dtype=np.float64)
            previous_reward = float(reward)
            previous_done = bool(done)
            total += float(reward)
            day += 1
        episode = builder.finish()
        if deviation_day < 0 or deviation_day >= episode.length:
            raise ValueError("deviation_day lies outside episode")
        if branch_center_coordinates is None or branch_action_coordinates is None:
            raise RuntimeError("recurrent branch action was not recorded")
        episode.metadata = {
            **dict(metadata or {}),
            "seed": int(seed),
            "deviation_day": int(deviation_day),
            "offset_coordinates": offset.tolist(),
            "branch_center_coordinates": branch_center_coordinates.tolist(),
            "branch_action_coordinates": branch_action_coordinates.tolist(),
            "continuation": "frozen_entering_recurrent_policy",
        }
        return episode, total, reset_observation
    finally:
        agent.extreme_action_count = diagnostic_state["extreme_action_count"]
        agent.nonfinite_action_rejections = diagnostic_state[
            "nonfinite_action_rejections"
        ]
        agent.max_abs_coordinate = diagnostic_state["max_abs_coordinate"]
        agent.max_physical_action = diagnostic_state["max_physical_action"]
        agent.min_physical_action = diagnostic_state["min_physical_action"]


def _verify_recurrent_policy_panel(
    episodes: Sequence[Episode],
    *,
    support: Mapping[str, Any],
    seeds: Sequence[int],
    deviation_days: Sequence[int],
    transform: ActionTransform,
    deviation_day_by_seed: Optional[Mapping[int, int]] = None,
) -> Dict[str, Any]:
    """Verify a complete matched-prefix recurrent seed/day/support panel."""
    if deviation_day_by_seed is None:
        seed_day_pairs = [
            (int(seed), int(day)) for seed in seeds for day in deviation_days
        ]
    else:
        normalized_mapping = {
            int(seed): int(day) for seed, day in deviation_day_by_seed.items()
        }
        if set(normalized_mapping) != {int(seed) for seed in seeds}:
            raise ValueError("recurrent fit panel deviation-day map is incomplete")
        seed_day_pairs = [
            (int(seed), normalized_mapping[int(seed)]) for seed in seeds
        ]
    expected = {
        (seed, day, int(row["index"]))
        for seed, day in seed_day_pairs
        for row in support["actions"]
    }
    observed: Dict[Tuple[int, int, int], Episode] = {}
    failures: List[str] = []
    offsets = {
        int(row["index"]): np.asarray(row["offset"], dtype=np.float64)
        for row in support["actions"]
    }
    for episode in episodes:
        metadata = episode.metadata
        key = (
            int(metadata["seed"]),
            int(metadata["deviation_day"]),
            int(metadata["support_index"]),
        )
        if key in observed:
            failures.append(f"duplicate seed/day/support row {key}")
        observed[key] = episode
    if set(observed) != expected:
        failures.append(
            f"panel keys differ: missing={len(expected - set(observed))}, "
            f"extra={len(set(observed) - expected)}"
        )
    for seed, day in seed_day_pairs:
        center = observed.get((int(seed), int(day), 0))
        if center is None:
            continue
        center_coordinates = transform.physical_to_coordinates(center.actions)
        for support_index, offset in offsets.items():
            candidate = observed.get((int(seed), int(day), support_index))
            if candidate is None:
                continue
            candidate_coordinates = transform.physical_to_coordinates(
                candidate.actions
            )
            if not np.array_equal(
                candidate.observations[: int(day) + 1],
                center.observations[: int(day) + 1],
            ):
                failures.append(
                    f"seed {seed}, day {day}, support {support_index}: "
                    "observation prefix mismatch"
                )
            if not np.array_equal(
                candidate.rewards[: int(day)], center.rewards[: int(day)]
            ):
                failures.append(
                    f"seed {seed}, day {day}, support {support_index}: "
                    "reward prefix mismatch"
                )
            if not np.allclose(
                candidate_coordinates[: int(day)],
                center_coordinates[: int(day)],
                atol=1.0e-7,
            ):
                failures.append(
                    f"seed {seed}, day {day}, support {support_index}: "
                    "action prefix mismatch"
                )
            expected_branch = center_coordinates[int(day)] + offset
            if not np.allclose(
                candidate_coordinates[int(day)], expected_branch, atol=1.0e-6
            ):
                failures.append(
                    f"seed {seed}, day {day}, support {support_index}: "
                    "branch offset mismatch"
                )
    return {
        "passed": bool(
            not failures
            and set(observed) == expected
            and int(support["support_rank"]) == transform.dim
            and bool(support["symmetric"])
        ),
        "criterion": (
            "complete TRAIN seed/day/support panel with exact recurrent/public "
            "prefix matching and a one-day coordinate perturbation"
        ),
        "episode_count": int(len(episodes)),
        "expected_episode_count": int(len(expected)),
        "seed_count": int(len(set(int(seed) for seed in seeds))),
        "seeds": [int(seed) for seed in seeds],
        "deviation_days": sorted({int(day) for _, day in seed_day_pairs}),
        "crossed_seed_day_panel": bool(deviation_day_by_seed is None),
        "support_count": int(len(support["actions"])),
        "matched_prefix_verified": not any("prefix mismatch" in row for row in failures),
        "failures": failures,
    }


def _rank_values(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    cursor = 0
    while cursor < array.size:
        stop = cursor + 1
        while stop < array.size and array[order[stop]] == array[order[cursor]]:
            stop += 1
        ranks[order[cursor:stop]] = 0.5 * (cursor + stop - 1)
        cursor = stop
    return ranks


def _stratified_deviation_days(
    seeds: Sequence[int], episode_days: int
) -> Dict[int, int]:
    if episode_days <= 0 or not seeds:
        raise ValueError("stratified deviations need seeds and a positive horizon")
    locations = np.rint(
        np.linspace(0, int(episode_days) - 1, num=len(seeds))
    ).astype(int)
    return {int(seed): int(day) for seed, day in zip(seeds, locations)}


def _weather_coverage(
    episodes: Sequence[Episode], observation_names: Sequence[str]
) -> Dict[str, Any]:
    """Audit independent absolute-fit trajectories for categorical support."""
    names = tuple(str(name) for name in observation_names)
    weather = {
        name: index for index, name in enumerate(names) if name.startswith("weather_")
    }
    public = {
        name: index for index, name in enumerate(names) if name.startswith("public_")
    }
    path_counts = {name: 0 for name in weather}
    day_counts = {name: 0 for name in weather}
    bounded_failures: List[str] = []
    unique_seeds = set()
    for episode in episodes:
        unique_seeds.add(int(episode.metadata["seed"]))
        observations = np.asarray(episode.observations[:-1], dtype=np.float64)
        for name, index in weather.items():
            active = observations[:, index] > 0.5
            path_counts[name] += int(bool(np.any(active)))
            day_counts[name] += int(np.sum(active))
        for name, index in {**weather, **public}.items():
            values = observations[:, index]
            if np.any(~np.isfinite(values)) or np.any(values < -1.0e-7) or np.any(
                values > 1.0 + 1.0e-7
            ):
                bounded_failures.append(name)
    required = tuple(
        name for name in ("weather_clear", "weather_rain") if name in weather
    )
    missing = [name for name in required if path_counts[name] <= 0]
    return {
        "passed": bool(
            len(episodes) == 64
            and len(unique_seeds) == len(episodes)
            and not missing
            and not bounded_failures
        ),
        "criterion": (
            "exactly 64 independent center-policy TRAIN paths; clear and rain observed; "
            "all weather/public indicators finite in [0,1]"
        ),
        "episode_count": int(len(episodes)),
        "unique_seed_count": int(len(unique_seeds)),
        "required_categories": list(required),
        "missing_required_categories": missing,
        "weather_path_counts": path_counts,
        "weather_day_counts": day_counts,
        "bounded_indicator_failures": sorted(set(bounded_failures)),
    }


def _verify_crossed_critic_panel(
    episodes: Sequence[Episode],
    *,
    support: Mapping[str, Any],
    seeds: Sequence[int],
    deviation_days: Sequence[int],
    center_coordinates: np.ndarray,
) -> Dict[str, Any]:
    """Verify the held-out TRAIN rank panel is the full seed/day/support cross."""
    expected_keys = {
        (int(seed), int(day), int(row["index"]))
        for seed in seeds
        for day in deviation_days
        for row in support["actions"]
    }
    observed: Dict[Tuple[int, int, int], Episode] = {}
    failures: List[str] = []
    for episode in episodes:
        metadata = episode.metadata
        key = (
            int(metadata["seed"]),
            int(metadata["deviation_day"]),
            int(metadata["support_index"]),
        )
        if key in observed:
            failures.append(f"duplicate seed/day/support row {key}")
        observed[key] = episode
    missing = sorted(expected_keys.difference(observed))
    extra = sorted(set(observed).difference(expected_keys))
    if missing:
        failures.append(f"missing {len(missing)} crossed rows")
    if extra:
        failures.append(f"unexpected {len(extra)} crossed rows")
    support_by_index = {
        int(row["index"]): np.asarray(row["coordinates"], dtype=np.float64)
        for row in support["actions"]
    }
    center = np.asarray(center_coordinates, dtype=np.float64)
    for seed in seeds:
        for day in deviation_days:
            center_episode = observed.get((int(seed), int(day), 0))
            if center_episode is None:
                continue
            for support_index, candidate in support_by_index.items():
                episode = observed.get((int(seed), int(day), support_index))
                if episode is None:
                    continue
                expected = np.tile(center, (episode.length, 1))
                expected[int(day)] = candidate
                if not np.allclose(episode.actions, expected, atol=1.0e-6):
                    failures.append(
                        f"seed {seed}, day {day}, support {support_index}: schedule mismatch"
                    )
                if not np.array_equal(
                    episode.observations[: int(day) + 1],
                    center_episode.observations[: int(day) + 1],
                ):
                    failures.append(
                        f"seed {seed}, day {day}, support {support_index}: prefix mismatch"
                    )
    return {
        "passed": bool(not failures and set(observed) == expected_keys),
        "criterion": "complete 8-seed x 4-day x 15-support held-out TRAIN cross",
        "episode_count": int(len(episodes)),
        "expected_episode_count": int(len(expected_keys)),
        "seed_count": int(len(set(int(seed) for seed in seeds))),
        "seeds": [int(seed) for seed in seeds],
        "deviation_days": [int(day) for day in deviation_days],
        "support_count": int(len(support["actions"])),
        "matched_prefix_verified": not any("prefix mismatch" in row for row in failures),
        "failures": failures,
    }


def _expected_value_calibration(
    diagnostics: Mapping[str, Any],
    *,
    deviation_days: Sequence[int],
    seeds_per_stratum: int,
    familywise_alpha: float = 0.05,
    threshold: float = 0.35,
) -> Dict[str, Any]:
    """Noise-aware expected-Q calibration on independent center trajectories.

    Each day stratum receives an independent seed cluster.  The simultaneous
    two-sided Student-t bounds are Bonferroni corrected across the fixed day
    strata; suffix length and replicated action supports never affect weight.
    """
    days = tuple(int(day) for day in deviation_days)
    if not days or len(days) != len(set(days)):
        raise ValueError("absolute calibration days must be unique")
    alpha = float(familywise_alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("familywise_alpha must lie in (0,1)")
    expected_per_day = int(seeds_per_stratum)
    if expected_per_day < 2:
        raise ValueError("absolute calibration needs at least two seeds per day")
    grouped: Dict[int, List[Mapping[str, Any]]] = {day: [] for day in days}
    observed_seeds: List[int] = []
    for row in diagnostics["episodes"]:
        metadata = dict(row.get("metadata", {}) or {})
        day = int(metadata.get("calibration_day", row["probe_transition"]))
        if day not in grouped:
            raise ValueError("absolute calibration row has an unregistered day")
        if int(metadata.get("support_index", 0)) != 0:
            raise ValueError("absolute calibration must use center-policy rows only")
        if metadata.get("heldout_for_gate") is not True:
            raise ValueError("absolute calibration row is not held out")
        seed = int(metadata["seed"])
        observed_seeds.append(seed)
        grouped[day].append(row)
    duplicate_seeds = len(observed_seeds) != len(set(observed_seeds))
    per_stratum_alpha = alpha / float(len(days))
    per_stratum_tail_alpha = alpha / float(2 * len(days))
    strata: Dict[str, Any] = {}
    passed = bool(not duplicate_seeds)
    ratios: List[float] = []
    for day in days:
        rows = grouped[day]
        residuals = np.asarray(
            [float(row["q_probe"]) - float(row["return_probe"]) for row in rows],
            dtype=np.float64,
        )
        returns = np.asarray(
            [float(row["return_probe"]) for row in rows], dtype=np.float64
        )
        count = int(residuals.size)
        finite = bool(np.all(np.isfinite(residuals)) and np.all(np.isfinite(returns)))
        if count >= 2 and finite:
            mean_residual = float(np.mean(residuals))
            standard_error = float(np.std(residuals, ddof=1) / math.sqrt(count))
            critical = _student_t_critical(
                1.0 - per_stratum_tail_alpha, count - 1
            )
            return_scale = max(float(np.mean(np.abs(returns))), 1.0e-8)
            numerator = abs(mean_residual) + critical * standard_error
            ratio = float(numerator / return_scale)
        else:
            mean_residual = float("nan")
            standard_error = float("nan")
            critical = float("nan")
            return_scale = float("nan")
            numerator = float("nan")
            ratio = float("inf")
        row_passed = bool(
            finite and count == expected_per_day and np.isfinite(ratio) and ratio <= threshold
        )
        passed = bool(passed and row_passed)
        ratios.append(ratio)
        strata[str(day)] = {
            "passed": row_passed,
            "count": count,
            "mean_residual_q_minus_return": mean_residual,
            "standard_error_clustered_by_seed": standard_error,
            "two_sided_student_t_critical_value": critical,
            "mean_absolute_return_scale": return_scale,
            "bias_plus_uncertainty": numerator,
            "normalized_bias_ucb": ratio,
            "raw_rmse_descriptive": (
                float(np.sqrt(np.mean(np.square(residuals)))) if finite and count else float("nan")
            ),
        }
    return {
        "passed": passed,
        "criterion": (
            "for every equally weighted fixed day stratum, "
            "(|mean(Q-G)| + Bonferroni two-sided Student-t * SE_seed) / "
            f"mean|G| <= {threshold:.2f}"
        ),
        "threshold": float(threshold),
        "familywise_alpha": alpha,
        "per_stratum_alpha": per_stratum_alpha,
        "per_stratum_tail_alpha": per_stratum_tail_alpha,
        "seeds_per_stratum": expected_per_day,
        "unique_seed_count": int(len(set(observed_seeds))),
        "duplicate_seed_clusters": bool(duplicate_seeds),
        "equal_stratum_mean_normalized_bias_ucb": float(np.mean(ratios)),
        "maximum_stratum_normalized_bias_ucb": float(np.max(ratios)),
        "raw_pooled_monte_carlo_rmse_descriptive": float(
            math.sqrt(float(diagnostics["monte_carlo_mse"]))
        ),
        "strata": strata,
    }


def _verify_local_replay_coverage(
    replay,
    *,
    group_id: str,
    support: Mapping[str, Any],
    seeds: Sequence[int],
    center_coordinates: np.ndarray,
) -> Dict[str, Any]:
    episodes = [
        episode
        for episode in replay.episodes
        if episode.metadata.get("group_id") == group_id
    ]
    expected_count = len(support["actions"]) * len(seeds)
    failures: List[str] = []
    by_seed: Dict[int, Dict[int, Episode]] = {}
    for episode in episodes:
        metadata = episode.metadata
        by_seed.setdefault(int(metadata["seed"]), {})[
            int(metadata["support_index"])
        ] = episode
    for seed in seeds:
        rows = by_seed.get(int(seed), {})
        if len(rows) != len(support["actions"]):
            failures.append(f"seed {seed}: incomplete support")
            continue
        center_episode = rows[0]
        deviation_day = int(center_episode.metadata["deviation_day"])
        for support_row in support["actions"]:
            index = int(support_row["index"])
            episode = rows[index]
            if int(episode.metadata["deviation_day"]) != deviation_day:
                failures.append(f"seed {seed}: inconsistent deviation day")
                continue
            candidate = np.asarray(support_row["coordinates"], dtype=np.float64)
            expected = np.tile(
                np.asarray(center_coordinates, dtype=np.float64),
                (episode.length, 1),
            )
            expected[deviation_day] = candidate
            if not np.allclose(episode.actions, expected, atol=1.0e-6):
                failures.append(f"seed {seed}, support {index}: action schedule mismatch")
            if not np.array_equal(
                episode.observations[: deviation_day + 1],
                center_episode.observations[: deviation_day + 1],
            ):
                failures.append(f"seed {seed}, support {index}: observation prefix mismatch")
            if not np.array_equal(
                episode.rewards[:deviation_day],
                center_episode.rewards[:deviation_day],
            ):
                failures.append(f"seed {seed}, support {index}: reward prefix mismatch")
    return {
        "passed": bool(
            len(episodes) == expected_count
            and not failures
            and int(support["support_rank"]) == len(center_coordinates)
            and bool(support["symmetric"])
        ),
        "episode_count": int(len(episodes)),
        "expected_episode_count": int(expected_count),
        "support_rank": int(support["support_rank"]),
        "symmetric": bool(support["symmetric"]),
        "seeds": [int(seed) for seed in seeds],
        "deviation_days": {
            str(seed): int(by_seed[int(seed)][0].metadata["deviation_day"])
            for seed in seeds
            if int(seed) in by_seed and 0 in by_seed[int(seed)]
        },
        "matched_prefix_verified": not any("prefix mismatch" in row for row in failures),
        "failures": failures,
    }


def _critic_ranking_gate(
    support: Mapping[str, Any],
    replay_coverage: Mapping[str, Any],
    diagnostics_before: Mapping[str, Any],
    diagnostics_after: Mapping[str, Any],
    episode_support_indices: Sequence[int],
    episode_seeds: Sequence[int],
    episode_probe_days: Optional[Sequence[int]] = None,
    *,
    gate_panel_coverage: Optional[Mapping[str, Any]] = None,
    absolute_fit_coverage: Optional[Mapping[str, Any]] = None,
    expected_value_calibration_before: Optional[Mapping[str, Any]] = None,
    expected_value_calibration_after: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    action_count = len(support["actions"])
    actual = np.zeros(action_count, dtype=np.float64)
    predicted = np.zeros(action_count, dtype=np.float64)
    counts = np.zeros(action_count, dtype=np.float64)
    for support_index, row in zip(
        episode_support_indices, diagnostics_after["episodes"]
    ):
        actual[int(support_index)] += float(
            row.get("return_probe", row["return0"])
        )
        predicted[int(support_index)] += float(
            row.get("q_probe", row["q0"])
        )
        counts[int(support_index)] += 1.0
    if np.any(counts <= 0.0):
        raise RuntimeError("critic ranking gate is missing a support action")
    actual /= counts
    predicted /= counts
    # Rank paired candidate-minus-center differences within each CRN seed.
    # This removes arbitrary seed-level value offsets from the gate.
    paired_actual: List[float] = []
    paired_predicted: List[float] = []
    paired_predicted_q1: List[float] = []
    paired_predicted_q2: List[float] = []
    if episode_probe_days is None:
        episode_probe_days = [0] * len(episode_seeds)
    if not (
        len(episode_support_indices)
        == len(episode_seeds)
        == len(episode_probe_days)
        == len(diagnostics_after["episodes"])
    ):
        raise ValueError("critic ranking row labels do not match diagnostics")
    rows_by_group: Dict[Tuple[int, int], Dict[int, Mapping[str, Any]]] = {}
    for support_index, seed, day, row in zip(
        episode_support_indices,
        episode_seeds,
        episode_probe_days,
        diagnostics_after["episodes"],
    ):
        rows_by_group.setdefault((int(seed), int(day)), {})[
            int(support_index)
        ] = row
    for seed_rows in rows_by_group.values():
        center = seed_rows.get(0)
        if center is None or len(seed_rows) != action_count:
            raise RuntimeError("critic gate needs every support action for each seed")
        for support_index in range(1, action_count):
            row = seed_rows[support_index]
            paired_actual.append(
                float(row.get("return_probe", row["return0"]))
                - float(center.get("return_probe", center["return0"]))
            )
            paired_predicted.append(
                float(row.get("q_probe", row["q0"]))
                - float(center.get("q_probe", center["q0"]))
            )
            paired_predicted_q1.append(
                float(row.get("q1_probe", row["q0"]))
                - float(center.get("q1_probe", center["q0"]))
            )
            paired_predicted_q2.append(
                float(row.get("q2_probe", row["q0"]))
                - float(center.get("q2_probe", center["q0"]))
            )
    before_rows_by_group: Dict[Tuple[int, int], Dict[int, Mapping[str, Any]]] = {}
    for support_index, seed, day, row in zip(
        episode_support_indices,
        episode_seeds,
        episode_probe_days,
        diagnostics_before["episodes"],
    ):
        before_rows_by_group.setdefault((int(seed), int(day)), {})[
            int(support_index)
        ] = row
    prior_predicted: Dict[str, List[float]] = {
        "qmin": [],
        "q1": [],
        "q2": [],
    }
    for group in rows_by_group:
        seed_rows = before_rows_by_group.get(group, {})
        center = seed_rows.get(0)
        if center is None or len(seed_rows) != action_count:
            raise RuntimeError("structured prior is missing a heldout support action")
        for support_index in range(1, action_count):
            row = seed_rows[support_index]
            prior_predicted["qmin"].append(
                float(row.get("q_probe", row["q0"]))
                - float(center.get("q_probe", center["q0"]))
            )
            prior_predicted["q1"].append(
                float(row.get("q1_probe", row["q0"]))
                - float(center.get("q1_probe", center["q0"]))
            )
            prior_predicted["q2"].append(
                float(row.get("q2_probe", row["q0"]))
                - float(center.get("q2_probe", center["q0"]))
            )
    actual_ranks = _rank_values(paired_actual)
    predicted_ranks = _rank_values(paired_predicted)
    if np.std(actual_ranks) <= 0.0 or np.std(predicted_ranks) <= 0.0:
        spearman = 0.0
    else:
        spearman = float(np.corrcoef(actual_ranks, predicted_ranks)[0, 1])
    def directional_accuracy(predicted_values: Sequence[float]) -> Tuple[float, int]:
        comparable = 0
        correct = 0
        for actual_difference, predicted_difference in zip(
            paired_actual, predicted_values
        ):
            if abs(actual_difference) <= 1.0e-12:
                continue
            comparable += 1
            correct += int(actual_difference * predicted_difference > 0.0)
        return (float(correct / comparable) if comparable else 0.0, comparable)

    pairwise_accuracy, comparable = directional_accuracy(paired_predicted)
    q1_accuracy, _ = directional_accuracy(paired_predicted_q1)
    q2_accuracy, _ = directional_accuracy(paired_predicted_q2)
    actual_delta_array = np.asarray(paired_actual, dtype=np.float64)
    trained_predictions = {
        "qmin": np.asarray(paired_predicted, dtype=np.float64),
        "q1": np.asarray(paired_predicted_q1, dtype=np.float64),
        "q2": np.asarray(paired_predicted_q2, dtype=np.float64),
    }
    zero_delta_rmse = float(np.sqrt(np.mean(np.square(actual_delta_array))))
    anti_prior_rows: Dict[str, Any] = {}
    anti_prior_passed = True
    for name, trained in trained_predictions.items():
        prior = np.asarray(prior_predicted[name], dtype=np.float64)
        trained_rmse = float(
            np.sqrt(np.mean(np.square(trained - actual_delta_array)))
        )
        prior_rmse = float(
            np.sqrt(np.mean(np.square(prior - actual_delta_array)))
        )
        passed_row = bool(
            np.isfinite(trained_rmse)
            and trained_rmse < prior_rmse
            and trained_rmse < zero_delta_rmse
        )
        anti_prior_passed = bool(anti_prior_passed and passed_row)
        anti_prior_rows[name] = {
            "passed": passed_row,
            "trained_paired_rmse": trained_rmse,
            "frozen_structured_prior_rmse": prior_rmse,
            "zero_delta_baseline_rmse": zero_delta_rmse,
            "trained_to_prior_ratio": float(
                trained_rmse / max(prior_rmse, 1.0e-12)
            ),
            "trained_to_zero_delta_ratio": float(
                trained_rmse / max(zero_delta_rmse, 1.0e-12)
            ),
        }
    before_td = float(diagnostics_before["td_mse"])
    after_td = float(diagnostics_after["td_mse"])
    before_mc = float(diagnostics_before["monte_carlo_mse"])
    after_mc = float(diagnostics_after["monte_carlo_mse"])
    normalized_mc = float(diagnostics_after["monte_carlo_normalized_rmse"])
    relative_q0_error = float(
        diagnostics_after.get(
            "probe_relative_absolute_error",
            diagnostics_after["q0_relative_absolute_error"],
        )
    )
    q_entropy_ratio = float(
        diagnostics_after.get(
            "probe_q_to_entropy_scale_ratio",
            diagnostics_after["q0_to_entropy_scale_ratio"],
        )
    )
    if expected_value_calibration_after is None:
        # Backward-compatible helper/test path.  The production L0 runner
        # always supplies the disjoint center-policy expected-value panel.
        calibration_passed = bool(
            np.isfinite(after_mc)
            and after_mc < before_mc
            and normalized_mc <= 0.35
            and relative_q0_error <= 0.35
            and q_entropy_ratio >= 5.0
        )
        calibration_contract = {
            "passed": calibration_passed,
            "legacy_pooled_fallback": True,
        }
    else:
        calibration_contract = dict(expected_value_calibration_after)
        calibration_passed = bool(
            calibration_contract["passed"]
            and np.isfinite(after_mc)
            and q_entropy_ratio >= 5.0
        )
    ranking_passed = bool(
        comparable >= 6
        and spearman >= 0.25
        and pairwise_accuracy >= (2.0 / 3.0)
        and q1_accuracy >= 0.55
        and q2_accuracy >= 0.55
    )
    coverage_passed = bool(
        int(support["support_rank"]) == 3
        and bool(support["symmetric"])
        and bool(replay_coverage["passed"])
        and bool(
            True if gate_panel_coverage is None else gate_panel_coverage["passed"]
        )
        and bool(
            True if absolute_fit_coverage is None else absolute_fit_coverage["passed"]
        )
    )
    passed = bool(
        coverage_passed
        and bool(diagnostics_after["finite"])
        and calibration_passed
        and ranking_passed
        and anti_prior_passed
    )
    return {
        "passed": passed,
        "criterion": (
            "full-rank symmetric support; finite held-out TRAIN diagnostics; "
            "independent center-policy TRAIN expected-Q bias plus Bonferroni "
            "Student-t uncertainty <=0.35 in every equally weighted day stratum; "
            "raw pooled MC RMSE descriptive only; Q/entropy scale >=5; "
            "paired CRN Spearman >=0.25 and "
            "Qmin directional accuracy >=2/3 and each twin >=0.55"
            "; trained paired RMSE beats frozen structured-prior and zero-delta baselines"
        ),
        "coverage_passed": coverage_passed,
        "replay_coverage": dict(replay_coverage),
        "gate_panel_coverage": (
            None if gate_panel_coverage is None else dict(gate_panel_coverage)
        ),
        "absolute_fit_weather_coverage": (
            None if absolute_fit_coverage is None else dict(absolute_fit_coverage)
        ),
        "heldout_monte_carlo_calibration_passed": calibration_passed,
        "expected_value_calibration": calibration_contract,
        "expected_value_calibration_before": (
            None
            if expected_value_calibration_before is None
            else dict(expected_value_calibration_before)
        ),
        "action_ranking_passed": ranking_passed,
        "anti_prior_baseline_passed": anti_prior_passed,
        "anti_prior_baseline_comparison": anti_prior_rows,
        "spearman": spearman,
        "pairwise_accuracy": pairwise_accuracy,
        "q1_directional_accuracy": q1_accuracy,
        "q2_directional_accuracy": q2_accuracy,
        "informative_pair_count": int(comparable),
        "actual_return_probe_by_support_action": actual.tolist(),
        "predicted_q_probe_by_support_action": predicted.tolist(),
        "actual_return_by_support_action": actual.tolist(),
        "td_mse_before": before_td,
        "td_mse_after": after_td,
        "monte_carlo_mse_before": before_mc,
        "monte_carlo_mse_after": after_mc,
        "monte_carlo_normalized_rmse_after": normalized_mc,
        "monte_carlo_metrics_are_descriptive_only": bool(
            expected_value_calibration_after is not None
        ),
        "probe_relative_absolute_error_after": relative_q0_error,
        "probe_q_to_entropy_scale_ratio_after": q_entropy_ratio,
        # Deprecated aliases retained for readers of pre-redesign reports.
        "q0_relative_absolute_error_after": relative_q0_error,
        "q0_to_entropy_scale_ratio_after": q_entropy_ratio,
        "td_relative_change": float(
            (after_td - before_td) / max(abs(before_td), 1.0e-12)
        ),
    }


def warm_start_l0(
    agent: RecurrentSACAgent,
    training_environment_factory,
    training_fit_seeds: Sequence[int],
    rollout_seeds: Sequence[int],
    critic_gate_seeds: Sequence[int],
    training_episode_seeds: Sequence[int],
    *,
    absolute_fit_seeds: Optional[Sequence[int]] = None,
    absolute_gate_seeds: Optional[Sequence[int]] = None,
    critic_probe_days: Optional[Sequence[int]] = None,
    cem_config: CEMConfig,
    clone_steps: int,
    critic_steps: int,
    paired_advantage_steps: int,
    support_radius: float,
    paired_advantage_weight: float,
    anchor_coefficient: float,
    anchor_decay_updates: float,
) -> Dict[str, Any]:
    """P6: initialize L0 from a rule selected exclusively on training data."""
    static_fit = optimize_static_tariff(
        training_environment_factory,
        agent.action_transform,
        training_fit_seeds,
        config=copy.copy(cem_config),
    )
    rule_fit = optimize_static_policy(
        training_environment_factory,
        agent.action_transform,
        training_fit_seeds,
        config=copy.copy(cem_config),
        initial_action=static_fit["action"],
    )
    action = np.asarray(rule_fit["action"], dtype=np.float64)
    target_coordinates = agent.action_transform.physical_to_coordinates(action)
    support = _local_action_support(
        agent.action_transform, target_coordinates, support_radius
    )
    agent.configure_critic_action_reference(
        target_coordinates,
        np.asarray(support["radius_coordinates"], dtype=np.float64),
    )
    horizon_probe = training_environment_factory()
    episode_days = int(
        getattr(getattr(horizon_probe, "config", None), "episode_days", 1)
    )
    fit_deviation_days = _stratified_deviation_days(rollout_seeds, episode_days)
    gate_days = tuple(
        _critic_probe_days(episode_days)
        if critic_probe_days is None
        else (int(day) for day in critic_probe_days)
    )
    if any(day < 0 or day >= episode_days for day in gate_days):
        raise ValueError("critic probe day lies outside the training horizon")
    if len(gate_days) != len(set(gate_days)):
        raise ValueError("critic probe days must be unique")
    all_input_seeds = [
        *training_fit_seeds,
        *rollout_seeds,
        *critic_gate_seeds,
        *training_episode_seeds,
    ]
    fallback_base = max((int(seed) for seed in all_input_seeds), default=0)
    if absolute_fit_seeds is None:
        absolute_fit_seeds = tuple(
            fallback_base + 1_000_003 + 104_729 * index for index in range(64)
        )
    else:
        absolute_fit_seeds = tuple(int(seed) for seed in absolute_fit_seeds)
    if len(absolute_fit_seeds) != 64 or len(set(absolute_fit_seeds)) != len(
        absolute_fit_seeds
    ):
        raise ValueError("absolute critic fit needs exactly 64 independent seeds")
    if absolute_gate_seeds is None:
        absolute_gate_seeds = tuple(
            fallback_base + 3_000_007 + 104_729 * index
            for index in range(2 * len(gate_days))
        )
    else:
        absolute_gate_seeds = tuple(int(seed) for seed in absolute_gate_seeds)
    if len(absolute_gate_seeds) % len(gate_days) != 0:
        raise ValueError("absolute gate seeds must divide equally across day strata")
    absolute_gate_seeds_per_stratum = len(absolute_gate_seeds) // len(gate_days)
    if absolute_gate_seeds_per_stratum < 2 or len(set(absolute_gate_seeds)) != len(
        absolute_gate_seeds
    ):
        raise ValueError("absolute gate needs independent seed clusters per day")
    absolute_fit_other_seeds = set(
        int(seed)
        for stream in (
            training_fit_seeds,
            rollout_seeds,
            critic_gate_seeds,
            absolute_gate_seeds,
            training_episode_seeds,
        )
        for seed in stream
    )
    if not set(absolute_fit_seeds).isdisjoint(absolute_fit_other_seeds):
        raise ValueError("absolute-fit TRAIN seeds overlap another data stream")
    rollout_returns: List[float] = []
    reset_observations: List[np.ndarray] = []
    support_fit_rows = []
    warm_action = action.copy()
    replay_group_id = "l0_train_local_support"

    # Absolute value calibration uses genuinely independent center-policy
    # paths.  The 15-way CRN panel below is reserved for local action effects
    # and is not pseudoreplicated as absolute context coverage.
    absolute_fit_episodes: List[Episode] = []
    absolute_fit_rows: List[Dict[str, Any]] = []
    for seed in absolute_fit_seeds:
        episode, total, _ = _local_deviation_episode(
            training_environment_factory,
            warm_action,
            warm_action,
            int(seed),
            0,
            metadata={
                "group_id": "l0_train_absolute_fit",
                "profile_split": "train_absolute_fit",
                "heldout_for_gate": False,
                "support_index": 0,
                "support_label": "center_policy",
                "independent_absolute_path": True,
            },
        )
        absolute_fit_episodes.append(episode)
        absolute_fit_rows.append(
            {"seed": int(seed), "return": float(total), "days": int(episode.length)}
        )
        agent.add_episode(episode, actions_are_physical=True)
    schema_names = getattr(horizon_probe, "observation_names", None)
    if schema_names is None:
        schema_names = tuple(
            f"feature_{index}"
            for index in range(absolute_fit_episodes[0].observations.shape[1])
        )
    observation_names = tuple(schema_names)
    configured_identity = tuple(
        getattr(
            getattr(agent, "config", None),
            "observation_identity_indices",
            (),
        )
    )
    if hasattr(horizon_probe, "observation_names"):
        expected_identity = _observation_identity_indices(observation_names)
        if configured_identity != expected_identity:
            raise RuntimeError(
                "learner observation identity mask does not match environment schema"
            )
    absolute_fit_weather_coverage = _weather_coverage(
        absolute_fit_episodes, observation_names
    )

    for support_row in support["actions"]:
        support_action = np.asarray(support_row["action"], dtype=np.float64)
        for seed in rollout_seeds:
            episode, total, reset_observation = _local_deviation_episode(
                training_environment_factory,
                warm_action,
                support_action,
                int(seed),
                fit_deviation_days[int(seed)],
                metadata={
                    "group_id": replay_group_id,
                    "profile_split": "train_fit",
                    "heldout_for_gate": False,
                    "support_index": int(support_row["index"]),
                    "support_label": str(support_row["label"]),
                    "offset_coordinates": list(support_row["offset"]),
                },
            )
            reset_observations.append(reset_observation)
            agent.add_episode(
                episode,
                actions_are_physical=True,
                update_normalizer=False,
            )
            rollout_returns.append(total)
            support_fit_rows.append(
                {
                    "support_index": int(support_row["index"]),
                    "label": str(support_row["label"]),
                    "seed": int(seed),
                    "return": float(total),
                    "deviation_day": int(fit_deviation_days[int(seed)]),
                    "continuation": "fixed_training_warm_rule",
                }
            )

    replay_coverage = _verify_local_replay_coverage(
        agent.replay,
        group_id=replay_group_id,
        support=support,
        seeds=rollout_seeds,
        center_coordinates=target_coordinates,
    )
    paired_cache_contract = agent.configure_paired_crn_cache(
        [
            episode
            for episode in agent.replay.episodes
            if str(episode.metadata.get("group_id", "")) == replay_group_id
        ],
        source_group_id=replay_group_id,
        paired_advantage_weight=float(paired_advantage_weight),
    )

    critic_gate_episodes: List[Episode] = []
    critic_gate_support_indices: List[int] = []
    critic_gate_episode_seeds: List[int] = []
    critic_gate_probe_days: List[int] = []
    critic_gate_rows = []
    for support_row in support["actions"]:
        support_action = np.asarray(support_row["action"], dtype=np.float64)
        for seed in critic_gate_seeds:
            for deviation_day in gate_days:
                episode, total, _ = _local_deviation_episode(
                    training_environment_factory,
                    warm_action,
                    support_action,
                    int(seed),
                    int(deviation_day),
                    metadata={
                        "group_id": "l0_heldout_train_critic_rank",
                        "profile_split": "heldout_train_critic_rank",
                        "heldout_for_gate": True,
                        "support_index": int(support_row["index"]),
                        "support_label": str(support_row["label"]),
                        "offset_coordinates": list(support_row["offset"]),
                        "crossed_seed_day_panel": True,
                    },
                )
                coordinates = agent.action_transform.physical_to_coordinates(
                    episode.actions
                ).astype(np.float32)
                critic_gate_episodes.append(
                    Episode(
                        episode.observations,
                        coordinates,
                        episode.rewards,
                        episode.dones,
                        metadata=episode.metadata,
                    )
                )
                critic_gate_support_indices.append(int(support_row["index"]))
                critic_gate_episode_seeds.append(int(seed))
                critic_gate_probe_days.append(int(deviation_day))
                critic_gate_rows.append(
                    {
                        "support_index": int(support_row["index"]),
                        "label": str(support_row["label"]),
                        "seed": int(seed),
                        "return": float(total),
                        "deviation_day": int(deviation_day),
                    }
                )
    critic_gate_panel_coverage = _verify_crossed_critic_panel(
        critic_gate_episodes,
        support=support,
        seeds=critic_gate_seeds,
        deviation_days=gate_days,
        center_coordinates=target_coordinates,
    )

    absolute_gate_episodes: List[Episode] = []
    absolute_gate_rows: List[Dict[str, Any]] = []
    for stratum_index, calibration_day in enumerate(gate_days):
        start = stratum_index * absolute_gate_seeds_per_stratum
        stop = start + absolute_gate_seeds_per_stratum
        for seed in absolute_gate_seeds[start:stop]:
            episode, total, _ = _local_deviation_episode(
                training_environment_factory,
                warm_action,
                warm_action,
                int(seed),
                int(calibration_day),
                metadata={
                    "group_id": "l0_heldout_train_critic_absolute",
                    "profile_split": "heldout_train_critic_absolute",
                    "heldout_for_gate": True,
                    "support_index": 0,
                    "support_label": "center_policy",
                    "calibration_day": int(calibration_day),
                    "independent_seed_cluster": True,
                },
            )
            coordinates = agent.action_transform.physical_to_coordinates(
                episode.actions
            ).astype(np.float32)
            absolute_gate_episodes.append(
                Episode(
                    episode.observations,
                    coordinates,
                    episode.rewards,
                    episode.dones,
                    metadata=episode.metadata,
                )
            )
            absolute_gate_rows.append(
                {
                    "seed": int(seed),
                    "calibration_day": int(calibration_day),
                    "return": float(total),
                }
            )
    clone = agent.behavior_clone(
        target_coordinates,
        steps=int(clone_steps),
        log_std_coordinates=np.full(
            agent.action_transform.dim,
            np.log(max(float(support_radius) / 2.5, 1.0e-4)),
            dtype=np.float64,
        ),
    )
    reset_parity_rows = []
    for rollout_row, reset_observation in zip(support_fit_rows, reset_observations):
        deterministic_action, _, action_info = agent.act(
            reset_observation,
            hidden=None,
            previous_action=None,
            previous_reward=0.0,
            previous_done=False,
            deterministic=True,
        )
        relative_error = (
            (deterministic_action - action)
            / np.maximum(np.abs(action), 1.0e-12)
        )
        reset_parity_rows.append(
            {
                "seed": int(rollout_row["seed"]),
                "support_index": int(rollout_row["support_index"]),
                "support_label": str(rollout_row["label"]),
                "action": deterministic_action.tolist(),
                "policy_coordinates": np.asarray(
                    action_info["policy_coordinates"], dtype=np.float64
                ).tolist(),
                "target_action": action.tolist(),
                "max_abs_relative_error": float(
                    np.max(np.abs(relative_error))
                ),
            }
        )
    reset_parity_tolerance = 1.0e-5
    reset_parity_passed = bool(
        clone.get("reset_action_parity_passed", False)
        and all(
            row["max_abs_relative_error"] <= reset_parity_tolerance
            for row in reset_parity_rows
        )
    )
    if not reset_parity_passed:
        raise RuntimeError(
            "L0 warm-start behavior clone failed deterministic reset-action parity"
        )
    critic_before = agent.critic_validation(critic_gate_episodes)
    absolute_critic_before = agent.critic_validation(absolute_gate_episodes)
    expected_value_before = _expected_value_calibration(
        absolute_critic_before,
        deviation_days=gate_days,
        seeds_per_stratum=absolute_gate_seeds_per_stratum,
    )
    critic = agent.fit_critics_monte_carlo(
        steps=int(critic_steps),
        paired_advantage_steps=int(paired_advantage_steps),
        paired_advantage_weight=float(paired_advantage_weight),
    )
    critic_after = agent.critic_validation(critic_gate_episodes)
    absolute_critic_after = agent.critic_validation(absolute_gate_episodes)
    expected_value_after = _expected_value_calibration(
        absolute_critic_after,
        deviation_days=gate_days,
        seeds_per_stratum=absolute_gate_seeds_per_stratum,
    )
    critic_gate = _critic_ranking_gate(
        support,
        replay_coverage,
        critic_before,
        critic_after,
        critic_gate_support_indices,
        critic_gate_episode_seeds,
        critic_gate_probe_days,
        gate_panel_coverage=critic_gate_panel_coverage,
        absolute_fit_coverage=absolute_fit_weather_coverage,
        expected_value_calibration_before=expected_value_before,
        expected_value_calibration_after=expected_value_after,
    )
    agent.configure_safe_policy_improvement(
        # Require a second, post-training-episode pass before any actor step.
        actor_updates_enabled=False,
        warm_anchor_coordinates=target_coordinates,
        anchor_coefficient=float(anchor_coefficient),
        anchor_decay_updates=float(anchor_decay_updates),
        anchor_scale=np.full(agent.action_transform.dim, float(support_radius)),
        actor_update_interval=2,
    )
    initial_full_checkpoint = None
    if critic_gate["passed"]:
        initial_full_checkpoint = agent.capture_last_passing_state(
            "l0_warm_prefit_gate_pass"
        )
    return {
        "training_static_fit": static_fit,
        "training_full_rule_fit": rule_fit,
        "target_action": action.tolist(),
        "target_coordinates": target_coordinates.tolist(),
        "training_fit_seeds": [int(seed) for seed in training_fit_seeds],
        "rollout_seeds": [int(seed) for seed in rollout_seeds],
        "critic_gate_seeds": [int(seed) for seed in critic_gate_seeds],
        "critic_gate_probe_days": [int(day) for day in gate_days],
        "absolute_fit_seeds": [int(seed) for seed in absolute_fit_seeds],
        "absolute_gate_seeds": [int(seed) for seed in absolute_gate_seeds],
        "rollout_count": int(len(rollout_returns)),
        "rollout_return_mean": float(np.mean(rollout_returns)),
        "rollout_returns": rollout_returns,
        "local_action_support": support,
        "critic_action_reference_contract": agent.critic_action_reference_contract(),
        "replay_support_coverage": replay_coverage,
        "paired_crn_online_refresh_cache": paired_cache_contract,
        "support_fit_rollouts": support_fit_rows,
        "absolute_fit_rollouts": absolute_fit_rows,
        "absolute_fit_weather_coverage": absolute_fit_weather_coverage,
        "observation_normalization_contract": {
            "schema_derived": True,
            "identity_feature_names": [
                observation_names[index]
                for index in getattr(
                    getattr(agent, "config", None),
                    "observation_identity_indices",
                    (),
                )
            ],
            "identity_feature_indices": [
                int(index)
                for index in getattr(
                    getattr(agent, "config", None),
                    "observation_identity_indices",
                    (),
                )
            ],
            "categorical_unseen_values_remain_natural_units": True,
            "paired_support_panel_updates_normalizer": False,
            "normalizer_fit_source": "independent_center_policy_train_paths",
        },
        "critic_gate_rollouts": critic_gate_rows,
        "critic_gate_crossed_panel_coverage": critic_gate_panel_coverage,
        "absolute_gate_rollouts": absolute_gate_rows,
        "replay_transitions_after_prefill": int(len(agent.replay)),
        "behavior_clone": clone,
        "deterministic_reset_action_parity": {
            "passed": reset_parity_passed,
            "relative_error_tolerance": reset_parity_tolerance,
            "rows": reset_parity_rows,
        },
        "critic_prefit": critic,
        "critic_validation_before": critic_before,
        "critic_validation_after": critic_after,
        "absolute_critic_validation_before": absolute_critic_before,
        "absolute_critic_validation_after": absolute_critic_after,
        "expected_value_calibration_before": expected_value_before,
        "expected_value_calibration_after": expected_value_after,
        "critic_coverage_and_ranking_gate": critic_gate,
        "safe_policy_improvement": {
            "actor_updates_enabled_initially": False,
            "initial_critic_gate_passed": bool(critic_gate["passed"]),
            "consecutive_passes_required": 2,
            "warm_anchor_coefficient": float(anchor_coefficient),
            "warm_anchor_decay_updates": float(anchor_decay_updates),
            "warm_anchor_scale": [float(support_radius)] * agent.action_transform.dim,
            "actor_update_interval": 2,
            "same_version_gate_before_every_actor_update": True,
            "last_passing_full_checkpoint": initial_full_checkpoint,
        },
        "profile_split_contract": {
            "tariff_static_selection": "train_warm_fit",
            "full_3d_rule_selection": "train_warm_fit",
            "rollout_prefill": "train",
            "critic_gate": "heldout_train",
            "validation_profiles_used": False,
            "fit_rollout_seeds_disjoint": bool(
                set(training_fit_seeds).isdisjoint(rollout_seeds)
            ),
            "fit_learning_seeds_disjoint": bool(
                set(training_fit_seeds).isdisjoint(training_episode_seeds)
            ),
            "prefill_learning_seeds_disjoint": bool(
                set(rollout_seeds).isdisjoint(training_episode_seeds)
            ),
            "critic_gate_disjoint_from_fit": bool(
                set(critic_gate_seeds).isdisjoint(training_fit_seeds)
                and set(critic_gate_seeds).isdisjoint(rollout_seeds)
                and set(critic_gate_seeds).isdisjoint(absolute_fit_seeds)
            ),
            "critic_gate_learning_seeds_disjoint": bool(
                set(critic_gate_seeds).isdisjoint(training_episode_seeds)
            ),
            "absolute_fit_all_training_only": True,
            "absolute_fit_disjoint_from_paired_panel": bool(
                set(absolute_fit_seeds).isdisjoint(rollout_seeds)
            ),
            "absolute_fit_disjoint_from_all_other_streams": bool(
                set(absolute_fit_seeds).isdisjoint(training_fit_seeds)
                and set(absolute_fit_seeds).isdisjoint(rollout_seeds)
                and set(absolute_fit_seeds).isdisjoint(critic_gate_seeds)
                and set(absolute_fit_seeds).isdisjoint(absolute_gate_seeds)
                and set(absolute_fit_seeds).isdisjoint(training_episode_seeds)
            ),
            "absolute_gate_disjoint_from_all_learning": bool(
                set(absolute_gate_seeds).isdisjoint(training_fit_seeds)
                and set(absolute_gate_seeds).isdisjoint(rollout_seeds)
                and set(absolute_gate_seeds).isdisjoint(absolute_fit_seeds)
                and set(absolute_gate_seeds).isdisjoint(training_episode_seeds)
            ),
            "rank_gate_disjoint_from_absolute_gate": bool(
                set(critic_gate_seeds).isdisjoint(absolute_gate_seeds)
            ),
            "final_test_used": False,
        },
        "_critic_gate_runtime": {
            "episodes": critic_gate_episodes,
            "support_indices": critic_gate_support_indices,
            "episode_seeds": critic_gate_episode_seeds,
            "probe_days": critic_gate_probe_days,
            "gate_panel_coverage": critic_gate_panel_coverage,
            "absolute_fit_coverage": absolute_fit_weather_coverage,
            "absolute_episodes": absolute_gate_episodes,
            "absolute_deviation_days": list(gate_days),
            "absolute_seeds_per_stratum": int(absolute_gate_seeds_per_stratum),
            "absolute_diagnostics_before": absolute_critic_before,
            "diagnostics_before": critic_before,
            "support": support,
            "replay_coverage": replay_coverage,
            "pass_streak": int(bool(critic_gate["passed"])),
            "warm_gate_authorization": {
                "update": 0,
                "passed": False,
                "gate_passed_current_version": bool(critic_gate["passed"]),
                "consecutive_passes": int(bool(critic_gate["passed"])),
                "actor_temperature_authorized_this_update": False,
                "actor_updates_enabled_for_next_episode": False,
                "gate": copy.deepcopy(critic_gate),
                "diagnostics": copy.deepcopy(critic_after),
                "absolute_expected_value_diagnostics": copy.deepcopy(
                    expected_value_after
                ),
                "split": "heldout_train_critic_gate",
                "source": "l0_warm_prefit",
                "same_critic_version_as_actor_update": True,
                "validation_used": False,
                "test_used": False,
            },
            "events": [],
        },
    }


def make_train_critic_gate_rechecker(runtime: Dict[str, Any]):
    """Gate each exact post-TD/post-refresh critic version before actor use.

    The initial prefit pass counts as the first pass.  Every online update is
    evaluated again on the dedicated held-out TRAIN stream, and only that
    current result can authorize actor/temperature learning.  A structural
    failure remains terminal; under the L0 CPI controller, an ordinary miss
    rejects the episode transaction back to the accepted incumbent.
    """

    def recheck(agent: RecurrentSACAgent, update_index: int) -> Dict[str, Any]:
        diagnostics = agent.critic_validation(runtime["episodes"])
        absolute_diagnostics = agent.critic_validation(
            runtime["absolute_episodes"]
        )
        expected_value = _expected_value_calibration(
            absolute_diagnostics,
            deviation_days=runtime["absolute_deviation_days"],
            seeds_per_stratum=int(runtime["absolute_seeds_per_stratum"]),
        )
        gate = _critic_ranking_gate(
            runtime["support"],
            runtime["replay_coverage"],
            runtime["diagnostics_before"],
            diagnostics,
            runtime["support_indices"],
            runtime["episode_seeds"],
            runtime["probe_days"],
            gate_panel_coverage=runtime["gate_panel_coverage"],
            absolute_fit_coverage=runtime["absolute_fit_coverage"],
            expected_value_calibration_before=_expected_value_calibration(
                runtime["absolute_diagnostics_before"],
                deviation_days=runtime["absolute_deviation_days"],
                seeds_per_stratum=int(runtime["absolute_seeds_per_stratum"]),
            ),
            expected_value_calibration_after=expected_value,
        )
        current_gate_passed = bool(gate["passed"])
        if current_gate_passed:
            runtime["pass_streak"] = int(runtime["pass_streak"]) + 1
        else:
            runtime["pass_streak"] = 0
        permission = bool(current_gate_passed and runtime["pass_streak"] >= 2)
        agent.actor_updates_enabled = permission
        row = {
            "update": int(update_index),
            "passed": permission,
            "gate_passed_current_version": current_gate_passed,
            "consecutive_passes": int(runtime["pass_streak"]),
            "actor_temperature_authorized_this_update": permission,
            "actor_updates_enabled_for_next_episode": False,
            "gate": gate,
            "diagnostics": diagnostics,
            "absolute_critic_diagnostics": absolute_diagnostics,
            "absolute_expected_value_diagnostics": expected_value,
            "split": "heldout_train_critic_gate",
            "evaluated_after_td_and_train_fit_paired_refresh": True,
            "same_critic_version_as_actor_update": True,
            "terminal_on_failure": True,
            "validation_used": False,
            "test_used": False,
        }
        runtime["events"].append(row)
        return row

    return recheck


def select_train_safe_policy(
    agent: RecurrentSACAgent,
    reference_policy_state: Mapping[str, Any],
    training_environment_factory,
    selection_seeds: Sequence[int],
) -> Dict[str, Any]:
    """Retain a candidate only if it safely beats the frozen train baseline.

    Selection uses a disjoint TRAIN seed stream with paired CRN.  Validation
    and test factories are intentionally absent from this API.  The policy
    snapshot excludes the observation normalizer: keeping the current
    normalizer preserves the coordinate system used by the trained critics,
    while the fixed warm actor mean is observation independent.
    """
    candidate_state = agent.policy_training_state()
    candidate = evaluate_recurrent_agent(
        training_environment_factory, agent, selection_seeds, deterministic=True
    )
    agent.load_policy_training_state(reference_policy_state)
    reference = evaluate_recurrent_agent(
        training_environment_factory, agent, selection_seeds, deterministic=True
    )
    difference = paired_difference(candidate, reference)
    standard_error = float(difference["standard_error"])
    lcb = float(difference["mean"])
    if np.isfinite(standard_error):
        lcb -= 2.0 * standard_error
    finite = bool(
        np.all(np.isfinite(candidate["values"]))
        and np.all(np.isfinite(reference["values"]))
        and np.isfinite(lcb)
    )
    # Equality provides no evidence of improvement and commonly means the
    # actor never received permission to update.  In that case deploy the
    # named warm reference rather than relabeling it as a learned candidate.
    retain_candidate = bool(finite and lcb > 1.0e-12)
    if retain_candidate:
        agent.load_policy_training_state(candidate_state)
        selected = "candidate"
    else:
        # The reference policy is already loaded.  Critic, replay, target,
        # and normalizer learning are deliberately retained.
        selected = "warm_reference"
    selected_evaluation = evaluate_recurrent_agent(
        training_environment_factory, agent, selection_seeds, deterministic=True
    )
    return {
        "selected": selected,
        "retained_candidate": retain_candidate,
        "criterion": (
            "paired TRAIN LCB = mean(candidate-reference) - 2*SE must be "
            "strictly positive; ties retain the warm reference"
        ),
        "paired_candidate_minus_reference": difference,
        "paired_lower_confidence_bound": lcb,
        "candidate": _compact_evaluation(candidate),
        "reference": _compact_evaluation(reference),
        "selected_evaluation": _compact_evaluation(selected_evaluation),
        "selection_seeds": [int(seed) for seed in selection_seeds],
        "split": "train_selection",
        "validation_used": False,
        "test_used": False,
        "normalizer_restored": False,
        "critics_replay_targets_restored": False,
    }


def _compact_evaluation(evaluation: Mapping[str, Any]) -> Dict[str, Any]:
    episodes = []
    episode_welfare_rows: List[Dict[str, float]] = []
    for episode in evaluation.get("episodes", []):
        actions = np.asarray(episode.get("actions", []), dtype=np.float64)
        infos = list(episode.get("infos", []))
        sampling_rows = [
            row
            for info in infos
            for row in info.get("conditional_sampling", [])
            if isinstance(row, Mapping)
        ]
        welfare_path = [
            {
                str(key): float(value)
                for key, value in dict(info.get("welfare_summary", {}) or {}).items()
                if np.isfinite(float(value))
            }
            for info in infos
        ]
        welfare_keys = sorted({key for row in welfare_path for key in row})
        welfare_mean = {
            key: float(np.mean([row[key] for row in welfare_path if key in row]))
            for key in welfare_keys
        }
        episode_welfare_rows.append(welfare_mean)
        episodes.append(
            {
                "seed": int(episode["seed"]),
                "return": float(episode["return"]),
                "days": int(episode["days"]),
                "oracle_event_days": [
                    int(day) for day in episode.get("oracle_event_days", [])
                ],
                "action_mean": actions.mean(axis=0).tolist() if actions.size else [],
                "action_std": actions.std(axis=0).tolist() if actions.size else [],
                "action_min": actions.min(axis=0).tolist() if actions.size else [],
                "action_max": actions.max(axis=0).tolist() if actions.size else [],
                "daily_actions": actions.tolist() if actions.size else [],
                "active_shock_path": [
                    bool(info.get("active_shocks")) for info in infos
                ],
                "dates": [info.get("date") for info in infos],
                "shock_days": int(sum(bool(info.get("active_shocks")) for info in infos)),
                "conditional_sampling_blocks": int(len(sampling_rows)),
                "conditional_sampling_changed_blocks": int(
                    sum(bool(row.get("changed", False)) for row in sampling_rows)
                ),
                "synthetic_airport_candidates": int(
                    sum(int(row.get("synthetic_airport_candidates", 0)) for row in sampling_rows)
                ),
                "welfare_mean": welfare_mean,
                "welfare_path": welfare_path,
            }
        )
    welfare_keys = sorted(
        {key for row in episode_welfare_rows for key in row}
    )
    welfare = {}
    for key in welfare_keys:
        values = np.asarray(
            [row[key] for row in episode_welfare_rows if key in row], dtype=float
        )
        welfare[key] = {
            "mean": float(values.mean()),
            "standard_error": (
                float(values.std(ddof=1) / np.sqrt(values.size))
                if values.size > 1
                else float("nan")
            ),
        }
    return {
        "mean": float(evaluation["mean"]),
        "standard_error": float(evaluation["standard_error"]),
        "median": float(evaluation["median"]),
        "values": [float(value) for value in evaluation["values"]],
        "episodes": episodes,
        "welfare_and_equity": welfare,
    }


def evaluate_stage(
    stage: str,
    agent: RecurrentSACAgent,
    validation_environment_factory,
    validation_fit_seeds: Sequence[int],
    validation_gate_seeds: Sequence[int],
    *,
    cem_config: CEMConfig,
    run_mpc: bool,
    mpc_seeds: int,
) -> Dict[str, Any]:
    transform = agent.action_transform
    probe = validation_environment_factory()
    public_indicator = probe.public_event_indicator

    anchor_action = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    anchor = evaluate_policy(
        validation_environment_factory,
        lambda: FixedPolicy(anchor_action),
        validation_gate_seeds,
    )
    static_fit = optimize_static_tariff(
        validation_environment_factory,
        transform,
        validation_fit_seeds,
        config=cem_config,
    )
    static_action = np.asarray(static_fit["action"], dtype=np.float64)
    static = evaluate_policy(
        validation_environment_factory,
        lambda: FixedPolicy(static_action),
        validation_gate_seeds,
    )
    rule_fit = optimize_static_policy(
        validation_environment_factory,
        transform,
        validation_fit_seeds,
        config=copy.copy(cem_config),
        initial_action=static_action,
    )
    rule_action = np.asarray(rule_fit["action"], dtype=np.float64)
    rule = evaluate_policy(
        validation_environment_factory,
        lambda: FixedPolicy(rule_action),
        validation_gate_seeds,
    )
    schedule_fit = optimize_public_schedule(
        validation_environment_factory,
        transform,
        validation_fit_seeds,
        public_indicator,
        rule_action,
        config=copy.copy(cem_config),
    )
    normal_action = np.asarray(schedule_fit["normal_action"], dtype=np.float64)
    event_action = np.asarray(schedule_fit["event_action"], dtype=np.float64)
    schedule = evaluate_policy(
        validation_environment_factory,
        lambda: PublicSchedulePolicy(
            normal_action, event_action, public_indicator
        ),
        validation_gate_seeds,
    )
    learned = evaluate_recurrent_agent(
        validation_environment_factory,
        agent,
        validation_gate_seeds,
        deterministic=True,
    )
    result: Dict[str, Any] = {
        "scenario_sampling": "natural_calendar",
        "validation_fit_seeds": [int(seed) for seed in validation_fit_seeds],
        "validation_gate_seeds": [int(seed) for seed in validation_gate_seeds],
        "profile_split_contract": {
            "cem_selection": "validation_fit",
            "rung_gate": "validation_gate",
            "fit_gate_seed_disjoint": bool(
                set(validation_fit_seeds).isdisjoint(validation_gate_seeds)
            ),
            "held_out_test_used": False,
        },
        "fits": {
            "best_static_tariff": static_fit,
            "two_timescale_rule": rule_fit,
            "public_schedule": schedule_fit,
        },
        "policies": {
            "anchor": _compact_evaluation(anchor),
            "best_static_tariff": _compact_evaluation(static),
            "two_timescale_rule": _compact_evaluation(rule),
            "public_schedule": _compact_evaluation(schedule),
            "recurrent_sac": _compact_evaluation(learned),
        },
        "paired": {
            "rl_minus_static": paired_difference(learned, static),
            "rl_minus_rule": paired_difference(learned, rule),
            "rl_minus_schedule": paired_difference(learned, schedule),
            "schedule_minus_static": paired_difference(schedule, static),
            "schedule_minus_rule": paired_difference(schedule, rule),
        },
    }
    if stage == "l2":
        event_days_factory = probe.clairvoyant_event_days
        oracle_fit = optimize_clairvoyant_event_action(
            validation_environment_factory,
            transform,
            validation_fit_seeds,
            event_days_factory,
            rule_action,
            config=copy.copy(cem_config),
        )
        oracle = evaluate_clairvoyant_event_schedule(
            validation_environment_factory,
            oracle_fit["normal_action"],
            oracle_fit["event_action"],
            event_days_factory,
            validation_gate_seeds,
        )
        result["fits"]["clairvoyant_event_window_oracle"] = oracle_fit
        result["policies"]["clairvoyant_event_window_oracle"] = _compact_evaluation(
            oracle
        )
        result["policies"]["clairvoyant_event_window_oracle"].update(
            {
                "fair_comparison": False,
                "information": "latent shortage event/recovery window known by seed",
            }
        )
        result["paired"]["oracle_minus_static"] = paired_difference(oracle, static)
        result["paired"]["oracle_minus_rule"] = paired_difference(oracle, rule)
        result["paired"]["rl_minus_oracle"] = paired_difference(learned, oracle)
        opportunity = result["paired"]["oracle_minus_rule"]
    else:
        opportunity = result["paired"]["schedule_minus_rule"]
    denominator = float(opportunity["mean"])
    identified = bool(
        denominator > 0.0
        and np.isfinite(float(opportunity["standard_error"]))
        and denominator > 3.0 * float(opportunity["standard_error"])
    )
    result["adaptive_opportunity_identified"] = identified
    result["adaptivity_index"] = (
        float(result["paired"]["rl_minus_rule"]["mean"]) / denominator
        if identified
        else None
    )
    result["adaptivity_baseline"] = "best_fixed_full_3d_rule"

    if run_mpc:
        selected_seeds = list(validation_gate_seeds)[: int(max(1, mpc_seeds))]
        returns = []
        episode_rows = []
        for episode_seed in selected_seeds:
            environment = validation_environment_factory()
            observation = environment.reset(int(episode_seed))
            planner = RecedingHorizonCEM(
                transform,
                horizon=7,
                particles=4,
                config=CEMConfig(
                    population=max(8, cem_config.population // 2),
                    iterations=max(2, cem_config.iterations - 1),
                    seed=int(episode_seed),
                ),
                discount=1.0,
                reveal_latent=True,
            )
            total = 0.0
            done = False
            calls = 0
            while not done:
                del observation
                plan = planner.plan(environment, seed=int(episode_seed) + calls)
                observation, reward, done, _ = environment.step(plan["action"])
                total += float(reward)
                calls += 1
            returns.append(total)
            episode_rows.append({"seed": int(episode_seed), "return": total, "planning_calls": calls})
        result["perfect_state_cem_mpc_upper_bound"] = {
            "mean": float(np.mean(returns)),
            "standard_error": (
                float(np.std(returns, ddof=1) / np.sqrt(len(returns)))
                if len(returns) > 1 else float("nan")
            ),
            "episodes": episode_rows,
            "fair_comparison": False,
            "reason": "planner branches from exact latent simulator state",
        }
    return result


def prepare_stage_critic_safety(
    agent: RecurrentSACAgent,
    training_environment_factory,
    *,
    stage: str,
    rollout_seeds: Sequence[int],
    critic_gate_seeds: Sequence[int],
    absolute_fit_seeds: Sequence[int],
    absolute_gate_seeds: Sequence[int],
    training_episode_seeds: Sequence[int],
    critic_probe_days: Sequence[int],
    critic_steps: int,
    paired_advantage_steps: int,
    support_radius: float,
    paired_advantage_weight: float,
) -> Dict[str, Any]:
    """Install stage-specific TRAIN critic evidence around the entering policy.

    L1 and later stages change the state/action distribution.  Reusing L0's
    fixed-rule CRN cache or held-out gate would authorize actor gradients under
    stale evidence.  This routine freezes the entering recurrent policy,
    builds independent absolute-fit, local-support, rank, and calibration
    panels in the current stage environment, refits only the critics, and
    returns a fresh same-version gate runtime.  Validation and test factories
    are intentionally not accepted by this API.
    """
    stage_name = str(stage)
    if stage_name not in STAGES or stage_name == "l0":
        raise ValueError("stage critic refresh is for L1 and later stages")
    rollout_seeds = tuple(int(seed) for seed in rollout_seeds)
    critic_gate_seeds = tuple(int(seed) for seed in critic_gate_seeds)
    absolute_fit_seeds = tuple(int(seed) for seed in absolute_fit_seeds)
    absolute_gate_seeds = tuple(int(seed) for seed in absolute_gate_seeds)
    training_episode_seeds = tuple(int(seed) for seed in training_episode_seeds)
    gate_days = tuple(int(day) for day in critic_probe_days)
    if len(absolute_fit_seeds) != 64 or len(set(absolute_fit_seeds)) != 64:
        raise ValueError("stage absolute critic fit needs exactly 64 unique seeds")
    if len(critic_gate_seeds) != 8 or len(set(critic_gate_seeds)) != 8:
        raise ValueError("stage critic rank gate needs exactly 8 unique seeds")
    if not gate_days or len(gate_days) != len(set(gate_days)):
        raise ValueError("stage critic probe days must be non-empty and unique")
    if len(absolute_gate_seeds) != 8 * len(gate_days):
        raise ValueError("stage absolute gate needs 8 independent seeds per day")
    streams = {
        "local_support": set(rollout_seeds),
        "critic_rank": set(critic_gate_seeds),
        "absolute_fit": set(absolute_fit_seeds),
        "absolute_gate": set(absolute_gate_seeds),
        "online_train": set(training_episode_seeds),
    }
    if any(len(values) != len(tuple_values) for values, tuple_values in (
        (streams["local_support"], rollout_seeds),
        (streams["critic_rank"], critic_gate_seeds),
        (streams["absolute_gate"], absolute_gate_seeds),
        (streams["online_train"], training_episode_seeds),
    )):
        raise ValueError("stage TRAIN seed stream contains duplicates")
    stream_names = tuple(streams)
    overlaps = {
        f"{left}:{right}": sorted(streams[left].intersection(streams[right]))
        for index, left in enumerate(stream_names)
        for right in stream_names[index + 1 :]
        if streams[left].intersection(streams[right])
    }
    if overlaps:
        raise ValueError(f"stage TRAIN evidence streams overlap: {overlaps}")

    horizon_probe = training_environment_factory()
    episode_days = int(horizon_probe.config.episode_days)
    if any(day < 0 or day >= episode_days for day in gate_days):
        raise ValueError("stage critic probe day lies outside the horizon")
    observation_names = tuple(horizon_probe.observation_names)
    expected_identity = _observation_identity_indices(observation_names)
    if tuple(agent.config.observation_identity_indices) != expected_identity:
        raise RuntimeError("stage learner identity mask differs from environment schema")

    # The support is expressed as relative native-coordinate offsets.  Its
    # displayed physical actions are merely origin examples and are never
    # substituted for the entering policy's state-dependent center action.
    support = _local_action_support(
        agent.action_transform,
        np.zeros(agent.action_dim, dtype=np.float64),
        support_radius,
    )
    support["coordinate_semantics"] = "offset_from_entering_recurrent_policy"
    support["displayed_actions_are_origin_examples_only"] = True
    fit_days = _stratified_deviation_days(rollout_seeds, episode_days)
    absolute_group = f"{stage_name}_train_absolute_fit"
    support_group = f"{stage_name}_train_local_support"
    rank_group = f"{stage_name}_heldout_train_critic_rank"
    absolute_gate_group = f"{stage_name}_heldout_train_critic_absolute"

    absolute_fit_episodes: List[Episode] = []
    absolute_fit_rows: List[Dict[str, Any]] = []
    zero_offset = np.zeros(agent.action_dim, dtype=np.float64)
    for seed in absolute_fit_seeds:
        episode, total, _ = _recurrent_policy_deviation_episode(
            training_environment_factory,
            agent,
            int(seed),
            0,
            zero_offset,
            metadata={
                "group_id": absolute_group,
                "stage": stage_name,
                "profile_split": "train_absolute_fit",
                "heldout_for_gate": False,
                "support_index": 0,
                "support_label": "entering_policy_center",
                "independent_absolute_path": True,
            },
        )
        absolute_fit_episodes.append(episode)
        absolute_fit_rows.append(
            {"seed": int(seed), "return": float(total), "days": episode.length}
        )

    support_fit_episodes: List[Episode] = []
    support_fit_rows: List[Dict[str, Any]] = []
    for seed in rollout_seeds:
        deviation_day = fit_days[int(seed)]
        for support_row in support["actions"]:
            episode, total, _ = _recurrent_policy_deviation_episode(
                training_environment_factory,
                agent,
                int(seed),
                int(deviation_day),
                np.asarray(support_row["offset"], dtype=np.float64),
                metadata={
                    "group_id": support_group,
                    "stage": stage_name,
                    "profile_split": "train_fit",
                    "heldout_for_gate": False,
                    "support_index": int(support_row["index"]),
                    "support_label": str(support_row["label"]),
                },
            )
            support_fit_episodes.append(episode)
            support_fit_rows.append(
                {
                    "seed": int(seed),
                    "support_index": int(support_row["index"]),
                    "label": str(support_row["label"]),
                    "return": float(total),
                    "deviation_day": int(deviation_day),
                }
            )
    replay_coverage = _verify_recurrent_policy_panel(
        support_fit_episodes,
        support=support,
        seeds=rollout_seeds,
        deviation_days=tuple(sorted(set(fit_days.values()))),
        deviation_day_by_seed=fit_days,
        transform=agent.action_transform,
    )
    for episode in absolute_fit_episodes:
        agent.add_episode(
            episode, actions_are_physical=True, update_normalizer=False
        )
    for episode in support_fit_episodes:
        agent.add_episode(
            episode, actions_are_physical=True, update_normalizer=False
        )
    paired_cache_contract = agent.configure_paired_crn_cache(
        [
            episode
            for episode in agent.replay.episodes
            if str(episode.metadata.get("group_id", "")) == support_group
        ],
        source_group_id=support_group,
        paired_advantage_weight=float(paired_advantage_weight),
        replace_for_new_stage=True,
    )
    absolute_fit_weather_coverage = _weather_coverage(
        absolute_fit_episodes, observation_names
    )

    critic_gate_episodes: List[Episode] = []
    critic_gate_physical_episodes: List[Episode] = []
    critic_gate_support_indices: List[int] = []
    critic_gate_episode_seeds: List[int] = []
    critic_gate_probe_days: List[int] = []
    critic_gate_rows: List[Dict[str, Any]] = []
    for seed in critic_gate_seeds:
        for deviation_day in gate_days:
            for support_row in support["actions"]:
                episode, total, _ = _recurrent_policy_deviation_episode(
                    training_environment_factory,
                    agent,
                    int(seed),
                    int(deviation_day),
                    np.asarray(support_row["offset"], dtype=np.float64),
                    metadata={
                        "group_id": rank_group,
                        "stage": stage_name,
                        "profile_split": "heldout_train_critic_rank",
                        "heldout_for_gate": True,
                        "support_index": int(support_row["index"]),
                        "support_label": str(support_row["label"]),
                        "crossed_seed_day_panel": True,
                    },
                )
                coordinates = agent.action_transform.physical_to_coordinates(
                    episode.actions
                ).astype(np.float32)
                critic_gate_physical_episodes.append(episode)
                critic_gate_episodes.append(
                    Episode(
                        episode.observations,
                        coordinates,
                        episode.rewards,
                        episode.dones,
                        metadata=episode.metadata,
                    )
                )
                critic_gate_support_indices.append(int(support_row["index"]))
                critic_gate_episode_seeds.append(int(seed))
                critic_gate_probe_days.append(int(deviation_day))
                critic_gate_rows.append(
                    {
                        "seed": int(seed),
                        "support_index": int(support_row["index"]),
                        "label": str(support_row["label"]),
                        "return": float(total),
                        "deviation_day": int(deviation_day),
                    }
                )
    gate_panel_coverage = _verify_recurrent_policy_panel(
        critic_gate_physical_episodes,
        support=support,
        seeds=critic_gate_seeds,
        deviation_days=gate_days,
        transform=agent.action_transform,
    )

    absolute_gate_seeds_per_stratum = len(absolute_gate_seeds) // len(gate_days)
    absolute_gate_episodes: List[Episode] = []
    absolute_gate_rows: List[Dict[str, Any]] = []
    for stratum_index, calibration_day in enumerate(gate_days):
        start = stratum_index * absolute_gate_seeds_per_stratum
        stop = start + absolute_gate_seeds_per_stratum
        for seed in absolute_gate_seeds[start:stop]:
            episode, total, _ = _recurrent_policy_deviation_episode(
                training_environment_factory,
                agent,
                int(seed),
                int(calibration_day),
                zero_offset,
                metadata={
                    "group_id": absolute_gate_group,
                    "stage": stage_name,
                    "profile_split": "heldout_train_critic_absolute",
                    "heldout_for_gate": True,
                    "support_index": 0,
                    "support_label": "entering_policy_center",
                    "calibration_day": int(calibration_day),
                    "independent_seed_cluster": True,
                },
            )
            coordinates = agent.action_transform.physical_to_coordinates(
                episode.actions
            ).astype(np.float32)
            absolute_gate_episodes.append(
                Episode(
                    episode.observations,
                    coordinates,
                    episode.rewards,
                    episode.dones,
                    metadata=episode.metadata,
                )
            )
            absolute_gate_rows.append(
                {
                    "seed": int(seed),
                    "calibration_day": int(calibration_day),
                    "return": float(total),
                }
            )

    critic_before = agent.critic_validation(critic_gate_episodes)
    absolute_before = agent.critic_validation(absolute_gate_episodes)
    expected_value_before = _expected_value_calibration(
        absolute_before,
        deviation_days=gate_days,
        seeds_per_stratum=absolute_gate_seeds_per_stratum,
    )
    critic_prefit = agent.fit_critics_monte_carlo(
        steps=int(critic_steps),
        paired_advantage_steps=int(paired_advantage_steps),
        paired_advantage_weight=float(paired_advantage_weight),
        absolute_fit_group_id=absolute_group,
        paired_fit_group_id=support_group,
    )
    critic_after = agent.critic_validation(critic_gate_episodes)
    absolute_after = agent.critic_validation(absolute_gate_episodes)
    expected_value_after = _expected_value_calibration(
        absolute_after,
        deviation_days=gate_days,
        seeds_per_stratum=absolute_gate_seeds_per_stratum,
    )
    critic_gate = _critic_ranking_gate(
        support,
        replay_coverage,
        critic_before,
        critic_after,
        critic_gate_support_indices,
        critic_gate_episode_seeds,
        critic_gate_probe_days,
        gate_panel_coverage=gate_panel_coverage,
        absolute_fit_coverage=absolute_fit_weather_coverage,
        expected_value_calibration_before=expected_value_before,
        expected_value_calibration_after=expected_value_after,
    )
    agent.configure_safe_policy_improvement(
        actor_updates_enabled=False,
        warm_anchor_coordinates=None,
        anchor_coefficient=0.0,
        anchor_decay_updates=1.0,
        actor_update_interval=2,
    )
    initial_checkpoint = None
    if critic_gate["passed"]:
        initial_checkpoint = agent.capture_last_passing_state(
            f"{stage_name}_entering_policy_critic_gate_pass"
        )
    runtime = {
        "stage": stage_name,
        "episodes": critic_gate_episodes,
        "support_indices": critic_gate_support_indices,
        "episode_seeds": critic_gate_episode_seeds,
        "probe_days": critic_gate_probe_days,
        "gate_panel_coverage": gate_panel_coverage,
        "absolute_fit_coverage": absolute_fit_weather_coverage,
        "absolute_episodes": absolute_gate_episodes,
        "absolute_deviation_days": list(gate_days),
        "absolute_seeds_per_stratum": int(
            absolute_gate_seeds_per_stratum
        ),
        "support": support,
        "diagnostics_before": critic_before,
        "absolute_diagnostics_before": absolute_before,
        "replay_coverage": replay_coverage,
        "events": [],
        "pass_streak": int(bool(critic_gate["passed"])),
        "warm_gate_authorization": {
            "update": int(agent.updates),
            "passed": False,
            "gate_passed_current_version": bool(critic_gate["passed"]),
            "consecutive_passes": int(bool(critic_gate["passed"])),
            "actor_temperature_authorized_this_update": False,
            "actor_updates_enabled_for_next_episode": False,
            "gate": copy.deepcopy(critic_gate),
            "stage": stage_name,
            "source": "stage_specific_entering_policy_prefit_gate",
            "split": "heldout_train_critic_gate",
        },
    }
    return {
        "stage": stage_name,
        "reference_policy": "frozen_entering_recurrent_policy",
        "validation_used": False,
        "test_used": False,
        "local_action_support": support,
        "support_fit_rollouts": support_fit_rows,
        "absolute_fit_rollouts": absolute_fit_rows,
        "absolute_fit_weather_coverage": absolute_fit_weather_coverage,
        "replay_support_coverage": replay_coverage,
        "paired_crn_online_refresh_cache": paired_cache_contract,
        "critic_gate_rollouts": critic_gate_rows,
        "critic_gate_crossed_panel_coverage": gate_panel_coverage,
        "absolute_gate_rollouts": absolute_gate_rows,
        "critic_validation_before": critic_before,
        "critic_validation_after": critic_after,
        "absolute_critic_validation_before": absolute_before,
        "absolute_critic_validation_after": absolute_after,
        "expected_value_calibration_before": expected_value_before,
        "expected_value_calibration_after": expected_value_after,
        "critic_prefit": critic_prefit,
        "critic_coverage_and_ranking_gate": critic_gate,
        "initial_full_state_checkpoint": initial_checkpoint,
        "profile_split_contract": {
            "all_evidence_train_only": True,
            "validation_used": False,
            "test_used": False,
            "streams_pairwise_disjoint": True,
            "seed_streams": {
                name: sorted(values) for name, values in streams.items()
            },
        },
        "_critic_gate_runtime": runtime,
    }


def gate_stage(stage: str, evaluation: Mapping[str, Any]) -> Dict[str, Any]:
    policies = evaluation["policies"]
    paired = evaluation["paired"]
    if stage == "l0":
        baseline = float(policies["two_timescale_rule"]["mean"])
        tolerance = 0.01 * max(1e-9, abs(baseline))
        delta = float(paired["rl_minus_rule"]["mean"])
        return {
            "name": "L0 optimizer sanity",
            "passed": bool(delta >= -tolerance),
            "criterion": (
                "RL no more than 1% below the best fixed full-3D rule "
                "(base multiplier, surge gain, driver incentive)"
            ),
            "baseline": "best_fixed_full_3d_rule",
            "rl_minus_fixed_rule": paired["rl_minus_rule"],
            "tariff_static_reported_separately": {
                "rl_minus_tariff_static": paired["rl_minus_static"],
            },
            "delta": delta,
            "tolerance": tolerance,
        }
    if stage == "l1":
        identified = bool(evaluation["adaptive_opportunity_identified"])
        index = evaluation.get("adaptivity_index")
        return {
            "name": "L1 public shock adaptation",
            "passed": bool(identified and index is not None and float(index) >= 0.50),
            "criterion": (
                "recover >=50% of significant public-schedule switching gain; "
                "RL and schedule are both measured over the best fixed full-3D rule"
            ),
            "opportunity_identified": identified,
            "adaptivity_index": index,
        }
    if stage == "l2":
        opportunity = paired.get("oracle_minus_rule")
        if opportunity is None:
            raise ValueError("L2 gate requires the clairvoyant event-window oracle")
        identified = bool(evaluation["adaptive_opportunity_identified"])
        index = evaluation.get("adaptivity_index")
        return {
            "name": "L2 latent driver-supply shock",
            "passed": bool(
                identified and index is not None and float(index) >= 0.30
            ),
            "criterion": (
                "recover >=30% of the significant clairvoyant event-window "
                "gain, with both gains measured over the best fixed full-3D rule"
            ),
            "opportunity_identified": identified,
            "oracle_minus_fixed_rule": opportunity,
            "rl_minus_fixed_rule": paired["rl_minus_rule"],
            "tariff_static_reported_separately": {
                "oracle_minus_tariff_static": paired["oracle_minus_static"],
                "rl_minus_tariff_static": paired["rl_minus_static"],
            },
            "adaptivity_index": index,
            "oracle_is_fair_policy": False,
        }
    if stage == "l3":
        static_delta = paired["rl_minus_static"]
        rule_delta = paired["rl_minus_rule"]
        return {
            "name": "L3 strategic-rival commitment and retention",
            "passed": bool(
                float(static_delta["mean"]) > 0.0
                and float(rule_delta["mean"]) > 0.0
            ),
            "criterion": "beat continuous best static and tuned myopic two-timescale rule",
            "rl_minus_static": static_delta,
            "rl_minus_rule": rule_delta,
        }
    if stage == "mixed":
        delta = paired["rl_minus_rule"]
        return {
            "name": "mixed-shock production promotion",
            "passed": bool(
                float(delta["mean"]) > 0.0 and float(delta["t"]) >= 2.0
            ),
            "criterion": "beat tuned two-timescale rule with paired t >= 2",
            "paired_difference": delta,
        }
    raise ValueError(f"unknown learner rung {stage!r}")


def evaluate_retention(
    stage: str,
    agent: RecurrentSACAgent,
    test_environment_factory,
    test_seeds: Sequence[int],
    original_evaluation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Re-evaluate one frozen learner against already-fitted CRN baselines."""
    fits = original_evaluation["fits"]
    static_action = np.asarray(
        fits["best_static_tariff"]["action"], dtype=np.float64
    )
    rule_action = np.asarray(
        fits["two_timescale_rule"]["action"], dtype=np.float64
    )
    normal_action = np.asarray(
        fits["public_schedule"]["normal_action"], dtype=np.float64
    )
    event_action = np.asarray(
        fits["public_schedule"]["event_action"], dtype=np.float64
    )
    indicator = test_environment_factory().public_event_indicator
    static = evaluate_policy(
        test_environment_factory, lambda: FixedPolicy(static_action), test_seeds
    )
    rule = evaluate_policy(
        test_environment_factory, lambda: FixedPolicy(rule_action), test_seeds
    )
    schedule = evaluate_policy(
        test_environment_factory,
        lambda: PublicSchedulePolicy(normal_action, event_action, indicator),
        test_seeds,
    )
    learned = evaluate_recurrent_agent(
        test_environment_factory, agent, test_seeds, deterministic=True
    )
    result: Dict[str, Any] = {
        "stage": stage,
        "evaluation_seeds": [int(seed) for seed in test_seeds],
        "policies": {
            "best_static_tariff": _compact_evaluation(static),
            "two_timescale_rule": _compact_evaluation(rule),
            "public_schedule": _compact_evaluation(schedule),
            "recurrent_sac": _compact_evaluation(learned),
        },
        "paired": {
            "rl_minus_static": paired_difference(learned, static),
            "rl_minus_rule": paired_difference(learned, rule),
            "rl_minus_schedule": paired_difference(learned, schedule),
            "schedule_minus_static": paired_difference(schedule, static),
            "schedule_minus_rule": paired_difference(schedule, rule),
        },
    }
    if stage == "l2" and "clairvoyant_event_window_oracle" in fits:
        oracle_fit = fits["clairvoyant_event_window_oracle"]
        event_days_factory = test_environment_factory().clairvoyant_event_days
        oracle = evaluate_clairvoyant_event_schedule(
            test_environment_factory,
            oracle_fit["normal_action"],
            oracle_fit["event_action"],
            event_days_factory,
            test_seeds,
        )
        result["policies"]["clairvoyant_event_window_oracle"] = _compact_evaluation(
            oracle
        )
        result["policies"]["clairvoyant_event_window_oracle"].update(
            {
                "fair_comparison": False,
                "information": "latent shortage event/recovery window known by seed",
            }
        )
        result["paired"]["oracle_minus_static"] = paired_difference(oracle, static)
        result["paired"]["oracle_minus_rule"] = paired_difference(oracle, rule)
        result["paired"]["rl_minus_oracle"] = paired_difference(learned, oracle)
        opportunity = result["paired"]["oracle_minus_rule"]
    else:
        opportunity = result["paired"]["schedule_minus_rule"]
    denominator = float(opportunity["mean"])
    identified = bool(
        denominator > 0.0
        and np.isfinite(float(opportunity["standard_error"]))
        and denominator > 3.0 * float(opportunity["standard_error"])
    )
    result["adaptive_opportunity_identified"] = identified
    result["adaptivity_index"] = (
        float(result["paired"]["rl_minus_rule"]["mean"]) / denominator
        if identified
        else None
    )
    result["adaptivity_baseline"] = "best_fixed_full_3d_rule"
    result["gate"] = gate_stage(stage, result)
    return result


def plot_report(report: Mapping[str, Any], destination: Path) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    stages = [
        stage
        for stage in report["stages"]
        if report["stage_results"].get(stage, {}).get("validation_evaluation") is not None
    ]
    if not stages:
        return None
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for stage in stages:
        logs = report["stage_results"][stage]["training"]
        returns = np.asarray([row["return"] for row in logs], dtype=float)
        if returns.size:
            if returns.size < 6:
                axes[0].plot(
                    np.arange(returns.size),
                    returns,
                    marker="o",
                    label=stage.upper(),
                )
            else:
                window = min(10, max(2, returns.size // 5))
                smoothed = np.convolve(
                    returns, np.ones(window) / window, mode="valid"
                )
                axes[0].plot(
                    np.arange(window - 1, returns.size),
                    smoothed,
                    label=stage.upper(),
                )
    axes[0].set_title("Training return (raw / moving average)")
    axes[0].set_xlabel("Episode within stage")
    axes[0].set_ylabel("Episode return")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    labels = []
    means = []
    errors = []
    final_audit = report.get("final_test_audit", {})
    for stage in stages:
        evaluation = final_audit.get(
            stage, report["stage_results"][stage]["validation_evaluation"]
        )
        policies = evaluation["policies"]
        policy_names = [
            "best_static_tariff",
            "two_timescale_rule",
            "public_schedule",
        ]
        if "clairvoyant_event_window_oracle" in policies:
            policy_names.append("clairvoyant_event_window_oracle")
        policy_names.append("recurrent_sac")
        for policy_name in policy_names:
            labels.append(f"{stage.upper()}\n{policy_name.replace('_', ' ')}")
            means.append(float(policies[policy_name]["mean"]))
            error = float(policies[policy_name]["standard_error"])
            errors.append(error if np.isfinite(error) else 0.0)
    x = np.arange(len(labels))
    axes[1].bar(x, means, yerr=errors, capsize=2)
    axes[1].set_xticks(x, labels, rotation=65, ha="right", fontsize=8)
    axes[1].set_title(
        "One-shot held-out CRN final audit"
        if final_audit
        else "Validation-only rung evaluation"
    )
    axes[1].set_ylabel("Episode return")
    axes[1].grid(axis="y", alpha=0.25)

    for stage in stages:
        tail = report["stage_results"][stage].get("price_tail_gate", {})
        grid = tail.get("grid", [])
        if not grid:
            continue
        multiplier = np.asarray([row["multiplier"] for row in grid], dtype=float)
        tail_mean = np.asarray([row["mean"] for row in grid], dtype=float)
        tail_error = np.asarray(
            [
                row["standard_error"]
                if row.get("standard_error") is not None
                and np.isfinite(float(row["standard_error"]))
                else 0.0
                for row in grid
            ],
            dtype=float,
        )
        axes[2].errorbar(
            multiplier,
            tail_mean,
            yerr=tail_error,
            marker="o",
            capsize=2,
            label=stage.upper(),
        )
    axes[2].set_xscale("log", base=2)
    axes[2].set_title("Uncapped tariff price-tail preflight")
    axes[2].set_xlabel("Fixed base tariff multiplier")
    axes[2].set_ylabel("Episode return")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    if report.get("budget_name") == "smoke":
        figure.suptitle(
            "Diagnostic smoke run — wiring/numerics only, not promotion evidence",
            fontsize=11,
        )
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return str(destination)


def plot_shock_adaptation(
    report: Mapping[str, Any], destination: Path
) -> Optional[str]:
    """Plot learned controls around held-out shock onset."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    stages = []
    aligned: Dict[str, List[np.ndarray]] = {}
    before, after = 3, 5
    final_audit = report.get("final_test_audit", {})
    source = (
        final_audit
        if final_audit
        else {
            stage: result.get("validation_evaluation")
            for stage, result in report["stage_results"].items()
        }
    )
    for stage, evaluation in source.items():
        if not evaluation:
            continue
        episodes = evaluation["policies"]["recurrent_sac"]["episodes"]
        rows: List[np.ndarray] = []
        for episode in episodes:
            path = np.asarray(episode.get("daily_actions", []), dtype=float)
            shock = np.asarray(episode.get("active_shock_path", []), dtype=bool)
            indices = np.flatnonzero(shock)
            if path.ndim != 2 or path.shape[1] != 3 or indices.size == 0:
                continue
            onset = int(indices[0])
            window = np.full((before + after + 1, 3), np.nan, dtype=float)
            start = max(0, onset - before)
            stop = min(path.shape[0], onset + after + 1)
            left = before - (onset - start)
            window[left:left + (stop - start)] = path[start:stop]
            rows.append(window)
        if rows:
            stages.append(stage)
            aligned[stage] = rows
    if not stages:
        return None
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    labels = ("Base tariff multiplier", "Surge gain", "Driver incentive multiplier")
    x = np.arange(-before, after + 1)
    for stage in stages:
        data = np.stack(aligned[stage])
        mean = np.nanmean(data, axis=0)
        count = np.sum(np.isfinite(data), axis=0)
        error = np.nanstd(data, axis=0, ddof=0) / np.sqrt(np.maximum(count, 1))
        for index, axis in enumerate(axes):
            axis.plot(x, mean[:, index], marker="o", label=stage.upper())
            axis.fill_between(
                x,
                mean[:, index] - error[:, index],
                mean[:, index] + error[:, index],
                alpha=0.15,
            )
    for index, axis in enumerate(axes):
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set_title(labels[index])
        axis.set_xlabel("Days from first active shock")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Learned physical action")
    axes[-1].legend()
    title = (
        "Final held-out learned controls aligned to shock onset"
        if final_audit
        else "Validation learned controls aligned to shock onset"
    )
    if report.get("budget_name") == "smoke":
        title += " (diagnostic smoke)"
    figure.suptitle(title)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return str(destination)


def _resolve(value: Optional[int | float], default: int | float) -> int | float:
    return default if value is None else value


def _validate_critic_only_request(
    critic_only: bool, selected_stages: Sequence[str]
) -> None:
    if critic_only and tuple(selected_stages) != ("l0",):
        raise ValueError("--critic-only requires exactly --stages l0")


def run_critic_only_diagnostic(
    *,
    args,
    budget: Budget,
    calibration: Mapping[str, Any],
    calibration_path: Path,
    trip_bank_path: Path,
    profile_cache_path: Path,
    out_dir: Path,
    warm_start_rollouts: int,
    warm_start_clone_steps: int,
    warm_start_critic_steps: int,
    warm_paired_advantage_steps: int,
) -> Dict[str, Any]:
    """Run the quarantined L0 critic diagnostic without actor learning.

    This deliberately has no validation/test factory in its API.  Every
    environment uses the TRAIN rider split; the only policy-side optimization
    is the preregistered warm-rule behavior clone.  CPI construction,
    ``train_stage`` and SAC actor/temperature updates are unreachable.
    """
    native_profile_cache = _dataset_paths(str(args.city))[2].resolve()
    if Path(profile_cache_path).resolve() != native_profile_cache:
        raise ValueError(
            "critic-only diagnostics require the native-v28 uncapped profile cache"
        )
    started = time.time()
    shock_config = _shock_config("l0", calibration)
    core, provenance = build_v28_core(
        city=args.city,
        calibration_path=calibration_path,
        trip_bank_path=trip_bank_path,
        threshold_cache_path=profile_cache_path,
        seed=int(args.seed) + STAGE_OFFSETS["l0"],
        profile_pool_size=int(budget.profile_pool_size),
        simulation_sample_cap=int(budget.customers_per_block),
        shock_config=shock_config,
        steps_per_day=6,
        rider_memory=True,
        firm2_mode="static",
    )
    env_config = DailyEnvironmentConfig(
        episode_days=int(budget.episode_days),
        customers_per_block=int(budget.customers_per_block),
        steps_per_day=6,
    )
    scenario_factory = _scenario_factory("l0", budget.episode_days)

    def training_environment_factory(
        core_template=core,
        config=env_config,
        scenario=scenario_factory,
    ):
        return ContinuousMarketEnv(
            core_template,
            config=config,
            scenario_factory=scenario,
            profile_split="train",
            shock_sampling_mode="training_openings",
        )

    probe = training_environment_factory()
    action_transform = ActionTransform.market_controls()
    agent = RecurrentSACAgent(
        observation_dim=probe.observation_dim,
        action_transform=action_transform,
        config=RecurrentSACConfig(
            hidden_dim=128,
            replay_capacity=100_000,
            batch_size=int(budget.batch_size),
            burn_in=int(budget.burn_in),
            learn_length=int(budget.learn_length),
            gamma=1.0,
            actor_learning_rate=1.0e-4,
            critic_learning_rate=3.0e-4,
            temperature_learning_rate=3.0e-4,
            polyak_tau=0.005,
            device=str(args.device),
            seed=int(args.seed),
            observation_identity_indices=_observation_identity_indices(
                probe.observation_names
            ),
        ),
    )
    tail_seeds = _episode_seeds(
        int(args.seed), "l0", max(2, budget.benchmark_train_seeds), "tail"
    )
    price_tail = evaluate_uncapped_price_tail(
        training_environment_factory, tail_seeds
    )
    base_report: Dict[str, Any] = {
        "schema": "ride_response_v28_critic_only_diagnostic_v1",
        "mode": "critic_only",
        "city": args.city,
        "budget_name": args.budget,
        "budget": asdict(budget),
        "requested_stages": ["l0"],
        "provenance": provenance,
        "native_profile_cache_contract": {
            "required": True,
            "path": str(profile_cache_path.resolve()),
            "sha256": _sha256_file(profile_cache_path),
            "explicit_legacy_override_allowed": False,
        },
        "environment_contract": probe.contract(),
        "learner_config": asdict(agent.config),
        "action_transform": asdict(action_transform),
        "price_tail_gate": price_tail,
        "profile_split_contract": {
            "all_environment_rollouts": "train",
            "validation_factory_constructed": False,
            "validation_split_opened": False,
            "test_factory_constructed": False,
            "test_split_opened": False,
        },
        "execution_contract": {
            "episode_actor_acceptance_constructed": False,
            "train_stage_called": False,
            "sac_actor_updates_called": False,
            "temperature_updates_called": False,
            "behavior_clone_only_policy_optimization": True,
            "promotion_eligible": False,
            "quarantined_diagnostic": True,
        },
        "promotion_ready": False,
        "all_gates_passed": False,
    }
    if not bool(price_tail["passed"]):
        base_report.update(
            {
                "status": "critic_only_tail_preflight_failed",
                "warm_start": None,
                "critic_gate_passed": False,
                "checkpoint": None,
                "elapsed_seconds": float(time.time() - started),
            }
        )
        report_path = out_dir / "critic_only_report.json"
        report_path.write_text(
            json.dumps(_jsonable(base_report), indent=2, allow_nan=False)
        )
        return base_report

    training_fit_seeds = _episode_seeds(
        int(args.seed),
        "l0",
        max(2, budget.benchmark_train_seeds),
        "train_warm_fit",
    )
    paired_fit_seeds = _episode_seeds(
        int(args.seed), "l0", int(warm_start_rollouts), "warm"
    )
    rank_gate_seeds = _episode_seeds(
        int(args.seed), "l0", 8, "train_critic_rank"
    )
    absolute_fit_seeds = _episode_seeds(
        int(args.seed), "l0", 64, "train_absolute_fit"
    )
    probe_days = _critic_probe_days(budget.episode_days)
    absolute_gate_seeds = _episode_seeds(
        int(args.seed),
        "l0",
        8 * len(probe_days),
        "train_critic_absolute_gate",
    )
    unused_training_seeds = _episode_seeds(
        int(args.seed), "l0", budget.train_episodes, "train"
    )
    cem_config = CEMConfig(
        population=int(budget.cem_population),
        iterations=int(budget.cem_iterations),
        elite_fraction=0.20,
        initial_std=0.50,
        minimum_std=0.025,
        seed=int(args.seed) + STAGE_OFFSETS["l0"] + 991,
    )
    warm_start = warm_start_l0(
        agent,
        training_environment_factory,
        training_fit_seeds,
        paired_fit_seeds,
        rank_gate_seeds,
        unused_training_seeds,
        absolute_fit_seeds=absolute_fit_seeds,
        absolute_gate_seeds=absolute_gate_seeds,
        critic_probe_days=probe_days,
        cem_config=cem_config,
        clone_steps=int(warm_start_clone_steps),
        critic_steps=int(warm_start_critic_steps),
        paired_advantage_steps=int(warm_paired_advantage_steps),
        support_radius=float(args.warm_support_radius),
        paired_advantage_weight=float(args.warm_paired_advantage_weight),
        anchor_coefficient=float(args.warm_anchor_coefficient),
        anchor_decay_updates=float(args.warm_anchor_decay_updates),
    )
    warm_start.pop("_critic_gate_runtime")
    critic_gate_passed = bool(
        warm_start["critic_coverage_and_ranking_gate"]["passed"]
    )
    if agent.actor_updates_applied != 0:
        raise RuntimeError("critic-only diagnostic applied a SAC actor update")
    checkpoint_path = out_dir / "critic_only_diagnostic_unpromoted.pt"
    base_report.update(
        {
            "status": (
                "critic_only_gate_passed_nonpromotion"
                if critic_gate_passed
                else "critic_only_gate_failed"
            ),
            "warm_start": warm_start,
            "critic_gate_passed": critic_gate_passed,
            "checkpoint": {
                "path": str(checkpoint_path),
                "role": "quarantined_critic_only_nonpromotion",
                "promotion_ready": False,
                "generic_resumable_policy": False,
            },
            "action_diagnostics": agent.action_diagnostics(),
            "elapsed_seconds": float(time.time() - started),
        }
    )
    report_path = out_dir / "critic_only_report.json"
    base_report["report"] = str(report_path)
    report_path.write_text(
        json.dumps(_jsonable(base_report), indent=2, allow_nan=False)
    )
    agent.save(
        checkpoint_path,
        metadata={
            "artifact_role": "quarantined_critic_only_nonpromotion",
            "critic_gate_passed": critic_gate_passed,
            "promotion_ready": False,
            "validation_opened": False,
            "test_opened": False,
            "report": str(report_path),
        },
    )
    return base_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--city", choices=("nyc", "boston", "chicago"), default="chicago")
    parser.add_argument("--budget", choices=tuple(BUDGETS), default="pilot")
    parser.add_argument("--stages", default="l0,l1,l2,l3")
    parser.add_argument("--seed", type=int, default=28_028)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="artifacts/v28_rsac/run")
    parser.add_argument("--calibration", default="")
    parser.add_argument("--trip-bank", default="")
    parser.add_argument(
        "--profile-cache",
        default="",
        help=(
            "explicit threshold-cache override (including legacy diagnostic "
            "comparators); the default is the fail-closed native-v28 uncapped cache"
        ),
    )
    parser.add_argument("--train-episodes", type=int)
    parser.add_argument("--episode-days", type=int)
    parser.add_argument("--warmup-episodes", type=int)
    parser.add_argument("--warm-start-rollouts", type=int)
    parser.add_argument("--warm-start-clone-steps", type=int)
    parser.add_argument("--warm-start-critic-steps", type=int)
    parser.add_argument("--warm-paired-advantage-steps", type=int)
    parser.add_argument("--warm-support-radius", type=float, default=0.15)
    parser.add_argument("--warm-paired-advantage-weight", type=float, default=1.0)
    parser.add_argument("--warm-anchor-coefficient", type=float, default=10.0)
    parser.add_argument("--warm-anchor-decay-updates", type=float, default=100.0)
    parser.add_argument("--evaluation-seeds", type=int)
    parser.add_argument("--benchmark-train-seeds", type=int)
    parser.add_argument("--customers", type=int)
    parser.add_argument("--profile-pool-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--burn-in", type=int)
    parser.add_argument("--learn-length", type=int)
    parser.add_argument("--updates-per-day", type=float)
    parser.add_argument(
        "--exploration-commitment-mean-days", type=float, default=3.0
    )
    parser.add_argument("--cem-population", type=int)
    parser.add_argument("--cem-iterations", type=int)
    parser.add_argument("--run-mpc", action="store_true")
    parser.add_argument("--mpc-seeds", type=int, default=2)
    parser.add_argument("--keep-replay-between-stages", action="store_true")
    parser.add_argument(
        "--critic-only",
        action="store_true",
        help=(
            "run only the quarantined L0 TRAIN-side tail/CEM/warm-clone/critic "
            "diagnostic; never train the SAC actor or open validation/test"
        ),
    )
    parser.add_argument(
        "--diagnostic-continue",
        action="store_true",
        help="continue after a failed preflight/rung gate (default ladder fails fast)",
    )
    args = parser.parse_args()
    if (
        not np.isfinite(float(args.exploration_commitment_mean_days))
        or float(args.exploration_commitment_mean_days) < 1.0
    ):
        raise ValueError("exploration commitment mean must be finite and >= 1 day")
    if not np.isfinite(args.warm_support_radius) or args.warm_support_radius <= 0:
        raise ValueError("warm support radius must be finite and positive")
    if (
        not np.isfinite(args.warm_paired_advantage_weight)
        or args.warm_paired_advantage_weight < 0
    ):
        raise ValueError("warm paired advantage weight must be finite and non-negative")

    selected_stages = tuple(value.strip().lower() for value in args.stages.split(",") if value.strip())
    unknown = set(selected_stages) - set(STAGES)
    if not selected_stages or unknown:
        raise ValueError(f"stages must be a non-empty subset of {STAGES}; unknown={sorted(unknown)}")
    canonical = tuple(stage for stage in STAGES if stage in selected_stages)
    if len(set(selected_stages)) != len(selected_stages) or selected_stages != canonical:
        raise ValueError(f"stages must be unique and ordered as {STAGES}")
    if "mixed" in selected_stages and selected_stages != STAGES:
        raise ValueError(
            "mixed production must be requested after the complete L0-L3 "
            f"readiness ladder: --stages {','.join(STAGES)}"
        )
    _validate_critic_only_request(bool(args.critic_only), selected_stages)
    preset = BUDGETS[args.budget]
    budget = Budget(
        train_episodes=int(_resolve(args.train_episodes, preset.train_episodes)),
        episode_days=int(_resolve(args.episode_days, preset.episode_days)),
        warmup_episodes=int(_resolve(args.warmup_episodes, preset.warmup_episodes)),
        evaluation_seeds=int(_resolve(args.evaluation_seeds, preset.evaluation_seeds)),
        benchmark_train_seeds=int(_resolve(args.benchmark_train_seeds, preset.benchmark_train_seeds)),
        customers_per_block=int(_resolve(args.customers, preset.customers_per_block)),
        profile_pool_size=int(_resolve(args.profile_pool_size, preset.profile_pool_size)),
        batch_size=int(_resolve(args.batch_size, preset.batch_size)),
        burn_in=int(_resolve(args.burn_in, preset.burn_in)),
        learn_length=int(_resolve(args.learn_length, preset.learn_length)),
        updates_per_day=float(_resolve(args.updates_per_day, preset.updates_per_day)),
        cem_population=int(_resolve(args.cem_population, preset.cem_population)),
        cem_iterations=int(_resolve(args.cem_iterations, preset.cem_iterations)),
    )
    warm_defaults = {
        "smoke": (4, 25, 200, 200),
        "pilot": (4, 100, 500, 500),
        "full": (4, 300, 1_000, 1_000),
    }[args.budget]
    warm_start_rollouts = int(_resolve(args.warm_start_rollouts, warm_defaults[0]))
    warm_start_clone_steps = int(
        _resolve(args.warm_start_clone_steps, warm_defaults[1])
    )
    warm_start_critic_steps = int(
        _resolve(args.warm_start_critic_steps, warm_defaults[2])
    )
    warm_paired_advantage_steps = int(
        _resolve(args.warm_paired_advantage_steps, warm_defaults[3])
    )
    if warm_start_critic_steps < 0 or warm_paired_advantage_steps < 0:
        raise ValueError("warm critic phase budgets must be non-negative")
    default_calibration, default_trip_bank, default_profiles = _dataset_paths(args.city)
    calibration_path = Path(args.calibration or default_calibration)
    trip_bank_path = Path(args.trip_bank or default_trip_bank)
    profile_cache_path = Path(args.profile_cache or default_profiles)
    if args.critic_only and profile_cache_path.resolve() != default_profiles.resolve():
        raise ValueError(
            "--critic-only is fail-closed to the native-v28 uncapped profile cache; "
            "legacy/alternate --profile-cache overrides are not accepted"
        )
    if not profile_cache_path.is_file():
        source = "explicit --profile-cache override" if args.profile_cache else "native-v28 default"
        raise FileNotFoundError(
            f"{source} is unavailable: {profile_cache_path}; v28 does not fall back "
            "silently to a legacy bounded cache"
        )
    calibration = load_calibration_json(
        str(calibration_path), expected_city={
            "nyc": "New York City", "boston": "Boston", "chicago": "Chicago"
        }[args.city],
    )
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.critic_only:
        report = run_critic_only_diagnostic(
            args=args,
            budget=budget,
            calibration=calibration,
            calibration_path=calibration_path,
            trip_bank_path=trip_bank_path,
            profile_cache_path=profile_cache_path,
            out_dir=out_dir,
            warm_start_rollouts=warm_start_rollouts,
            warm_start_clone_steps=warm_start_clone_steps,
            warm_start_critic_steps=warm_start_critic_steps,
            warm_paired_advantage_steps=warm_paired_advantage_steps,
        )
        print(f"wrote {out_dir / 'critic_only_report.json'}")
        print(f"critic-only gate passed: {report['critic_gate_passed']}")
        print("promotion ready: False")
        return

    steps_per_day = 6
    episode_blocks = budget.episode_days * steps_per_day
    stage_count = len(selected_stages)
    readiness_stage_count = sum(stage != "mixed" for stage in selected_stages)
    production_stage_count = int("mixed" in selected_stages)
    # Base-price (6), high/low incentive (5+5), signed surge (7), and three
    # joint (5+5+5) tail axes.  These are actual environment rollouts, not
    # cheap algebra, so every repeated anchor is counted again.
    tail_grid_size = 38
    tail_seed_count = max(2, budget.benchmark_train_seeds)
    training_blocks = stage_count * budget.train_episodes * episode_blocks
    static_search_candidate_upper_bound = 65 + 9 * budget.cem_iterations
    warm_start_search_count = int("l0" in selected_stages)
    validation_static_search_episodes = (
        stage_count
        * static_search_candidate_upper_bound
        * budget.benchmark_train_seeds
    )
    training_warm_static_search_episodes = (
        warm_start_search_count
        * static_search_candidate_upper_bound
        * budget.benchmark_train_seeds
    )
    static_search_episodes = (
        validation_static_search_episodes + training_warm_static_search_episodes
    )
    validation_cem_optimizer_count = (
        2 * stage_count + int("l2" in selected_stages)
    )
    training_warm_cem_optimizer_count = warm_start_search_count
    cem_optimizer_count = (
        validation_cem_optimizer_count + training_warm_cem_optimizer_count
    )
    cem_candidate_episodes = (
        cem_optimizer_count
        * budget.cem_population
        * budget.cem_iterations
        * budget.benchmark_train_seeds
    )
    tail_episodes = stage_count * tail_grid_size * tail_seed_count
    validation_policy_episodes = (
        5 * stage_count + int("l2" in selected_stages)
    ) * budget.evaluation_seeds
    if "mixed" in selected_stages:
        retention_rung_count = 4
    elif "l3" in selected_stages:
        retention_rung_count = sum(
            stage in selected_stages for stage in ("l0", "l1", "l2")
        )
    else:
        retention_rung_count = 0
    validation_retention_episodes = (
        retention_rung_count
        * 5
        * budget.evaluation_seeds
    )
    shock_stratified_validation_episodes = (
        4 * budget.evaluation_seeds if "mixed" in selected_stages else 0
    )
    final_test_episodes = (
        (
            4 * stage_count
            + int("l2" in selected_stages)
            + (4 if "mixed" in selected_stages else 0)
        )
        * budget.evaluation_seeds
        if args.budget == "full"
        else 0
    )
    warm_support_action_count = 15
    warm_critic_rank_seed_count = 8
    warm_critic_probe_day_count = len(_critic_probe_days(budget.episode_days))
    warm_absolute_fit_seed_count = 64
    warm_absolute_gate_seeds_per_stratum = 8
    actor_acceptance_seed_count = int(
        ACTOR_ACCEPTANCE_SEEDS_PER_EPISODE[args.budget]
    )
    warm_prefill_episodes = (
        stage_count * warm_support_action_count * warm_start_rollouts
    )
    warm_critic_gate_episodes = (
        warm_support_action_count
        * warm_critic_rank_seed_count
        * warm_critic_probe_day_count
        * stage_count
    )
    warm_absolute_fit_episodes = (
        warm_absolute_fit_seed_count * stage_count
    )
    warm_absolute_gate_episodes = (
        warm_absolute_gate_seeds_per_stratum * warm_critic_probe_day_count
        * stage_count
    )
    warm_selection_episodes = 0
    actor_acceptance_episodes = (
        2
        * budget.train_episodes
        * actor_acceptance_seed_count
        * stage_count
    )
    warm_prefill_blocks = warm_prefill_episodes * episode_blocks
    nontraining_blocks = (
        cem_candidate_episodes
        + static_search_episodes
        + tail_episodes
        + validation_policy_episodes
        + validation_retention_episodes
        + shock_stratified_validation_episodes
        + final_test_episodes
        + warm_critic_gate_episodes
        + warm_absolute_gate_episodes
        + warm_selection_episodes
        + actor_acceptance_episodes
    ) * episode_blocks
    estimate = {
        "scope": "planned upper bound before fail-fast; perfect-state MPC listed separately",
        "readiness_ladder": {
            "stages": [stage for stage in selected_stages if stage != "mixed"],
            "training_market_blocks": int(
                readiness_stage_count
                * budget.train_episodes
                * episode_blocks
            ),
            "training_profile_row_evaluations": int(
                readiness_stage_count
                * budget.train_episodes
                * episode_blocks
                * budget.customers_per_block
            ),
        },
        "post_gate_mixed_production": {
            "selected": bool(production_stage_count),
            "training_market_blocks": int(
                production_stage_count
                * budget.train_episodes
                * episode_blocks
            ),
            "training_profile_row_evaluations": int(
                production_stage_count
                * budget.train_episodes
                * episode_blocks
                * budget.customers_per_block
            ),
            "shock_process": "all configured SHOCK_TYPES on empirical calendar",
            "requires_readiness_pass": True,
        },
        "training_environment_days": training_blocks // 6,
        "expected_new_behavior_action_draws": int(
            math.ceil(
                (training_blocks // 6)
                / float(args.exploration_commitment_mean_days)
            )
        ),
        "daily_macro_action_applications": training_blocks // 6,
        "training_market_blocks": training_blocks,
        "training_profile_row_evaluations": training_blocks * budget.customers_per_block,
        "l0_warm_start_rollout_episodes": int(warm_prefill_episodes),
        "l0_warm_support_action_count": int(warm_support_action_count),
        "l0_heldout_train_critic_gate_episodes": int(warm_critic_gate_episodes),
        "l0_independent_center_absolute_fit_episodes": int(
            warm_absolute_fit_episodes
        ),
        "l0_heldout_center_absolute_gate_episodes": int(
            warm_absolute_gate_episodes
        ),
        "l0_critic_rank_cross": {
            "seeds": int(warm_critic_rank_seed_count),
            "days": [int(day) for day in _critic_probe_days(budget.episode_days)],
            "support_actions": int(warm_support_action_count),
        },
        "l0_train_selection_policy_episodes": int(warm_selection_episodes),
        "l0_actor_acceptance_policy_episode_upper_bound": int(
            actor_acceptance_episodes
        ),
        "l0_actor_acceptance_seeds_per_episode": int(
            actor_acceptance_seed_count
        ),
        "l0_actor_acceptance_familywise_alpha": 0.05,
        "stage_specific_critic_safety": {
            "stage_count": int(stage_count),
            "train_local_support_episodes": int(warm_prefill_episodes),
            "train_absolute_fit_episodes": int(warm_absolute_fit_episodes),
            "heldout_train_rank_episodes": int(warm_critic_gate_episodes),
            "heldout_train_absolute_gate_episodes": int(
                warm_absolute_gate_episodes
            ),
            "actor_acceptance_policy_episode_upper_bound": int(
                actor_acceptance_episodes
            ),
            "validation_used": False,
            "test_used": False,
        },
        "l0_paired_crn_prefill_market_blocks": int(warm_prefill_blocks),
        "l0_independent_absolute_fit_market_blocks": int(
            warm_absolute_fit_episodes * episode_blocks
        ),
        "l0_warm_start_market_blocks": int(
            warm_prefill_blocks + warm_absolute_fit_episodes * episode_blocks
        ),
        "l0_warm_start_clone_steps": int(warm_start_clone_steps),
        "l0_warm_start_critic_steps": int(warm_start_critic_steps),
        "l0_warm_paired_advantage_steps": int(warm_paired_advantage_steps),
        "l0_warm_critic_optimizer_steps_per_twin": int(
            warm_start_critic_steps + 2 * warm_paired_advantage_steps
        ),
        "l0_warm_phase2_substeps": [
            "paired_advantage_head",
            "absolute_value_head_refresh",
        ],
        "l0_warm_support_radius_coordinates": float(args.warm_support_radius),
        "l0_warm_paired_advantage_weight": float(
            args.warm_paired_advantage_weight
        ),
        "approximate_gradient_updates": int(
            len(selected_stages)
            * budget.train_episodes
            * budget.episode_days
            * budget.updates_per_day
        ),
        "cem_candidate_episodes": int(cem_candidate_episodes),
        "cem_candidate_market_blocks": int(cem_candidate_episodes * episode_blocks),
        "training_warm_rule_cem_candidate_episodes": int(
            training_warm_cem_optimizer_count
            * budget.cem_population
            * budget.cem_iterations
            * budget.benchmark_train_seeds
        ),
        "validation_comparator_cem_candidate_episodes": int(
            validation_cem_optimizer_count
            * budget.cem_population
            * budget.cem_iterations
            * budget.benchmark_train_seeds
        ),
        "training_warm_static_search_episode_upper_bound": int(
            training_warm_static_search_episodes
        ),
        "validation_static_search_episode_upper_bound": int(
            validation_static_search_episodes
        ),
        "adaptive_static_search_episode_upper_bound": int(static_search_episodes),
        "adaptive_static_search_market_blocks_upper_bound": int(
            static_search_episodes * episode_blocks
        ),
        "price_tail_gate_episodes": int(tail_episodes),
        "validation_policy_episodes": int(validation_policy_episodes),
        "validation_retention_policy_episodes": int(
            validation_retention_episodes
        ),
        "shock_stratified_validation_policy_episodes": int(
            shock_stratified_validation_episodes
        ),
        "one_shot_final_test_policy_episodes": int(final_test_episodes),
        "test_split_deferred_to_full_budget": bool(args.budget != "full"),
        "all_nontraining_market_blocks_excluding_mpc": int(nontraining_blocks),
        "all_profile_row_evaluations_excluding_mpc": int(
            (
                training_blocks
                + warm_prefill_blocks
                + warm_absolute_fit_episodes * episode_blocks
                + nontraining_blocks
            )
            * budget.customers_per_block
        ),
        "perfect_state_mpc_enabled": bool(args.run_mpc),
        "perfect_state_mpc_branch_market_blocks_upper_bound": int(
            stage_count
            * min(budget.evaluation_seeds, max(1, int(args.mpc_seeds)))
            * budget.episode_days
            * max(8, budget.cem_population // 2)
            * max(2, budget.cem_iterations - 1)
            * 4
            * 7
            * steps_per_day
        ) if args.run_mpc else 0,
    }
    print("[budget] " + json.dumps(estimate), flush=True)

    action_transform = ActionTransform.market_controls()
    agent: Optional[RecurrentSACAgent] = None
    provenance: Optional[Dict[str, Any]] = None
    contract: Optional[Dict[str, Any]] = None
    stage_results: Dict[str, Any] = {}
    retention_specs: Dict[str, Dict[str, Any]] = {}
    started = time.time()

    for stage_index, stage in enumerate(selected_stages):
        if stage == "mixed":
            prerequisites = ("l0", "l1", "l2", "l3")
            ready = bool(
                args.budget in {"pilot", "full"}
                and all(name in stage_results for name in prerequisites)
                and all(
                    stage_results[name]["gate"]["passed"]
                    and stage_results[name]["gate"].get(
                        "promotion_eligible", False
                    )
                    for name in prerequisites
                )
            )
            if not ready:
                print(
                    "[mixed] BLOCKED: post-gate production requires a "
                    "promotion-eligible L0-L3 pass in this run",
                    flush=True,
                )
                stage_results[stage] = {
                    "status": "blocked_by_readiness_ladder",
                    "gate": {
                        "name": "mixed-shock production prerequisite",
                        "passed": False,
                        "criterion": "promotion-eligible L0-L3 pass",
                    },
                    "training": [],
                }
                break
        print(f"\n=== {stage.upper()} ===", flush=True)
        shock_config = _shock_config(stage, calibration)
        core, stage_provenance = build_v28_core(
            city=args.city,
            calibration_path=calibration_path,
            trip_bank_path=trip_bank_path,
            threshold_cache_path=profile_cache_path,
            seed=int(args.seed) + STAGE_OFFSETS[stage],
            profile_pool_size=int(budget.profile_pool_size),
            simulation_sample_cap=int(budget.customers_per_block),
            shock_config=shock_config,
            steps_per_day=6,
            rider_memory=True,
            firm2_mode=(
                "adaptive_best_response_aggressive"
                if stage in {"l3", "mixed"}
                else "static"
            ),
        )
        env_config = DailyEnvironmentConfig(
            episode_days=int(budget.episode_days),
            customers_per_block=int(budget.customers_per_block),
            steps_per_day=6,
        )
        scenario_factory = _scenario_factory(stage, budget.episode_days)

        def training_environment_factory(
            core_template=core, config=env_config, scenario=scenario_factory
        ):
            return ContinuousMarketEnv(
                core_template,
                config=config,
                scenario_factory=scenario,
                profile_split="train",
                shock_sampling_mode="training_openings",
            )

        def validation_environment_factory(
            core_template=core, config=env_config, scenario=scenario_factory
        ):
            return ContinuousMarketEnv(
                core_template,
                config=config,
                scenario_factory=scenario,
                profile_split="validation",
                shock_sampling_mode="natural",
            )

        def test_environment_factory(
            core_template=core, config=env_config, scenario=scenario_factory
        ):
            return ContinuousMarketEnv(
                core_template,
                config=config,
                scenario_factory=scenario,
                profile_split="test",
                shock_sampling_mode="natural",
            )

        def shock_stratified_validation_environment_factory(
            core_template=core, config=env_config, scenario=scenario_factory
        ):
            return ContinuousMarketEnv(
                core_template,
                config=config,
                scenario_factory=scenario,
                profile_split="validation",
                shock_sampling_mode="training_openings",
            )

        def shock_stratified_test_environment_factory(
            core_template=core, config=env_config, scenario=scenario_factory
        ):
            return ContinuousMarketEnv(
                core_template,
                config=config,
                scenario_factory=scenario,
                profile_split="test",
                shock_sampling_mode="training_openings",
            )

        probe = training_environment_factory()
        if agent is None:
            agent = RecurrentSACAgent(
                observation_dim=probe.observation_dim,
                action_transform=action_transform,
                config=RecurrentSACConfig(
                    hidden_dim=128,
                    replay_capacity=100_000,
                    batch_size=int(budget.batch_size),
                    burn_in=int(budget.burn_in),
                    learn_length=int(budget.learn_length),
                    gamma=1.0,
                    actor_learning_rate=1e-4,
                    critic_learning_rate=3e-4,
                    temperature_learning_rate=3e-4,
                    polyak_tau=0.005,
                    device=str(args.device),
                    seed=int(args.seed),
                    observation_identity_indices=_observation_identity_indices(
                        probe.observation_names
                    ),
                ),
            )
            contract = probe.contract()
            provenance = stage_provenance
        elif not args.keep_replay_between_stages:
            # The shock generator is a new behavior distribution.  Preserve
            # learned weights as curriculum initialization, but do not train
            # off stale replay from a differently versioned environment.
            agent.reset_replay(seed=int(args.seed) + STAGE_OFFSETS[stage] + 71)

        train_seeds = _episode_seeds(
            int(args.seed), stage, budget.train_episodes, "train"
        )
        validation_fit_seeds = _episode_seeds(
            int(args.seed), stage, budget.benchmark_train_seeds, "validation_fit"
        )
        validation_gate_seeds = _episode_seeds(
            int(args.seed), stage, budget.evaluation_seeds, "validation_gate"
        )
        test_seeds = _episode_seeds(
            int(args.seed), stage, budget.evaluation_seeds, "test"
        )
        tail_seeds = _episode_seeds(
            int(args.seed), stage, max(2, budget.benchmark_train_seeds), "tail"
        )
        critic_gate_seeds = _episode_seeds(
            int(args.seed),
            stage,
            8,
            "train_critic_rank",
        )
        absolute_fit_seeds = _episode_seeds(
            int(args.seed), stage, 64, "train_absolute_fit"
        )
        critic_probe_days = _critic_probe_days(budget.episode_days)
        absolute_gate_seeds = _episode_seeds(
            int(args.seed),
            stage,
            8 * len(critic_probe_days),
            "train_critic_absolute_gate",
        )
        train_selection_seeds = _episode_seeds(
            int(args.seed),
            stage,
            max(2, budget.benchmark_train_seeds),
            "train_selection",
        )
        actor_accept_seeds_per_episode = int(
            ACTOR_ACCEPTANCE_SEEDS_PER_EPISODE[args.budget]
        )
        actor_accept_flat_seeds = _episode_seeds(
            int(args.seed),
            stage,
            int(budget.train_episodes) * actor_accept_seeds_per_episode,
            "train_actor_accept",
        )
        actor_accept_seed_blocks = [
            actor_accept_flat_seeds[
                index * actor_accept_seeds_per_episode :
                (index + 1) * actor_accept_seeds_per_episode
            ]
            for index in range(int(budget.train_episodes))
        ]
        cem_config = CEMConfig(
            population=int(budget.cem_population),
            iterations=int(budget.cem_iterations),
            elite_fraction=0.20,
            initial_std=0.50,
            minimum_std=0.025,
            seed=int(args.seed) + STAGE_OFFSETS[stage] + 991,
        )
        stage_started = time.time()
        price_tail = evaluate_uncapped_price_tail(
            validation_environment_factory,
            tail_seeds,
        )
        print(
            f"[{stage}] price-tail={'PASS' if price_tail['passed'] else 'FAIL'} "
            f"best={price_tail['best_multiplier']:.3g}x "
            f"endpoint={price_tail['grid'][-1]['mean']:+.3f}",
            flush=True,
        )
        if not price_tail["passed"] and not args.diagnostic_continue:
            stage_results[stage] = {
                "status": "preflight_failed",
                "training_seeds": train_seeds,
                "training": [],
                "price_tail_gate": price_tail,
                "gate": {
                    "name": "uncapped price-tail preflight",
                    "passed": False,
                    "criterion": price_tail["criterion"],
                },
                "shock_config": shock_config.as_dict(),
                "provenance": stage_provenance,
                "elapsed_seconds": float(time.time() - stage_started),
            }
            print(
                f"[{stage}] fail-fast: structural price-tail gate failed; "
                "use --diagnostic-continue only for diagnosis",
                flush=True,
            )
            break
        warm_start: Optional[Dict[str, Any]] = None
        stage_safety: Optional[Dict[str, Any]] = None
        critic_gate_runtime: Optional[Dict[str, Any]] = None
        actor_gate_recheck = None
        actor_acceptance: Optional[EpisodeActorAcceptance] = None
        initial_safe_gate_failed = False
        if stage == "l0":
            warm_fit_seeds = _episode_seeds(
                int(args.seed),
                stage,
                max(2, budget.benchmark_train_seeds),
                "train_warm_fit",
            )
            warm_seeds = _episode_seeds(
                int(args.seed), stage, warm_start_rollouts, "warm"
            )
            warm_start = warm_start_l0(
                agent,
                training_environment_factory,
                warm_fit_seeds,
                warm_seeds,
                critic_gate_seeds,
                train_seeds,
                absolute_fit_seeds=absolute_fit_seeds,
                absolute_gate_seeds=absolute_gate_seeds,
                critic_probe_days=critic_probe_days,
                cem_config=cem_config,
                clone_steps=warm_start_clone_steps,
                critic_steps=warm_start_critic_steps,
                paired_advantage_steps=warm_paired_advantage_steps,
                support_radius=float(args.warm_support_radius),
                paired_advantage_weight=float(args.warm_paired_advantage_weight),
                anchor_coefficient=float(args.warm_anchor_coefficient),
                anchor_decay_updates=float(args.warm_anchor_decay_updates),
            )
            critic_gate_runtime = warm_start.pop("_critic_gate_runtime")
            actor_gate_recheck = make_train_critic_gate_rechecker(
                critic_gate_runtime
            )
            initial_safe_gate_failed = not bool(
                warm_start["critic_coverage_and_ranking_gate"]["passed"]
            )
            if not initial_safe_gate_failed:
                actor_acceptance = EpisodeActorAcceptance(
                    agent,
                    training_environment_factory,
                    actor_accept_seed_blocks,
                    critic_gate_runtime,
                    stage=stage,
                    forbidden_seeds=[
                        *train_seeds,
                        *validation_fit_seeds,
                        *validation_gate_seeds,
                        *test_seeds,
                        *tail_seeds,
                        *critic_gate_seeds,
                        *absolute_fit_seeds,
                        *absolute_gate_seeds,
                        *train_selection_seeds,
                        *warm_fit_seeds,
                        *warm_seeds,
                    ],
                    familywise_alpha=0.05,
                    incumbent_checkpoint_directory=(
                        out_dir / "economic_incumbents"
                    ),
                )
            print(
                "[l0 warm-start] "
                f"full-rule={warm_start['target_action']} "
                f"clone_loss={warm_start['behavior_clone']['final_loss']:.6g} "
                f"critic_loss={warm_start['critic_prefit']['final_loss']:.6g}",
                flush=True,
            )
        else:
            stage_fit_seeds = _episode_seeds(
                int(args.seed), stage, warm_start_rollouts, "warm"
            )
            stage_safety = prepare_stage_critic_safety(
                agent,
                training_environment_factory,
                stage=stage,
                rollout_seeds=stage_fit_seeds,
                critic_gate_seeds=critic_gate_seeds,
                absolute_fit_seeds=absolute_fit_seeds,
                absolute_gate_seeds=absolute_gate_seeds,
                training_episode_seeds=train_seeds,
                critic_probe_days=critic_probe_days,
                critic_steps=warm_start_critic_steps,
                paired_advantage_steps=warm_paired_advantage_steps,
                support_radius=float(args.warm_support_radius),
                paired_advantage_weight=float(args.warm_paired_advantage_weight),
            )
            critic_gate_runtime = stage_safety.pop("_critic_gate_runtime")
            actor_gate_recheck = make_train_critic_gate_rechecker(
                critic_gate_runtime
            )
            initial_safe_gate_failed = not bool(
                stage_safety["critic_coverage_and_ranking_gate"]["passed"]
            )
            if not initial_safe_gate_failed:
                actor_acceptance = EpisodeActorAcceptance(
                    agent,
                    training_environment_factory,
                    actor_accept_seed_blocks,
                    critic_gate_runtime,
                    stage=stage,
                    forbidden_seeds=[
                        *train_seeds,
                        *validation_fit_seeds,
                        *validation_gate_seeds,
                        *test_seeds,
                        *tail_seeds,
                        *critic_gate_seeds,
                        *absolute_fit_seeds,
                        *absolute_gate_seeds,
                        *train_selection_seeds,
                        *stage_fit_seeds,
                    ],
                    familywise_alpha=0.05,
                    incumbent_checkpoint_directory=(
                        out_dir / "economic_incumbents" / stage
                    ),
                )
            print(
                f"[{stage} critic refresh] gate="
                f"{'PASS' if not initial_safe_gate_failed else 'FAIL'} "
                f"absolute_UCB={stage_safety['expected_value_calibration_after']['maximum_stratum_normalized_bias_ucb']:.4f} "
                f"rank={stage_safety['critic_coverage_and_ranking_gate']['spearman']:.4f}",
                flush=True,
            )
        if initial_safe_gate_failed:
            training = []
        else:
            training = train_stage(
                agent,
                training_environment_factory,
                train_seeds,
                # L1+ already has stage-local absolute/support replay.  A
                # neutral random warmup would invalidate that locality and is
                # therefore reserved for the L0 construction only.
                warmup_episodes=(
                    int(budget.warmup_episodes) if stage == "l0" else 0
                ),
                updates_per_day=float(budget.updates_per_day),
                seed=int(args.seed) + STAGE_OFFSETS[stage],
                exploration_commitment_mean_days=float(
                    args.exploration_commitment_mean_days
                ),
                warmup_center_coordinates=(
                    None
                    if warm_start is None
                    else np.asarray(warm_start["target_coordinates"], dtype=np.float64)
                ),
                warmup_coordinate_std=(
                    None
                    if warm_start is None
                    else np.full(
                        agent.action_dim,
                        float(args.warm_support_radius) / 2.5,
                        dtype=np.float64,
                    )
                ),
                actor_gate_recheck=actor_gate_recheck,
                actor_acceptance=actor_acceptance,
            )
        safety_report = warm_start if stage == "l0" else stage_safety
        if safety_report is not None:
            assert critic_gate_runtime is not None
            safety_report["post_training_critic_gate_rechecks"] = list(
                critic_gate_runtime["events"]
            )
            attempted_gate = (
                critic_gate_runtime["events"][-1]
                if critic_gate_runtime["events"]
                else {
                    "passed": False,
                    "gate_passed_current_version": False,
                    "consecutive_passes": int(
                        critic_gate_runtime["pass_streak"]
                    ),
                    "actor_temperature_authorized_this_update": False,
                    "actor_updates_enabled_for_next_episode": False,
                    "reason": "no post-training recheck was run",
                }
            )
            safety_report["post_training_critic_gate_attempted"] = copy.deepcopy(
                attempted_gate
            )
            safety_report["post_training_critic_gate"] = (
                actor_acceptance.current_incumbent_gate()
                if actor_acceptance is not None
                else copy.deepcopy(attempted_gate)
            )
            safety_report["candidate_actor_updates_applied"] = int(
                agent.actor_updates_applied
            )
            safety_report["episode_actor_acceptance"] = (
                None if actor_acceptance is None else actor_acceptance.summary()
            )
        terminal_safe_failure = bool(
            initial_safe_gate_failed or agent.training_terminal_failure
        )
        if terminal_safe_failure:
            reason = (
                "initial held-out TRAIN critic gate failed"
                if initial_safe_gate_failed
                else str(agent.training_terminal_failure_reason)
            )
            verified_incumbent_rollback = None
            if actor_acceptance is not None:
                verified_incumbent_rollback = agent.restore_economic_incumbent(
                    "terminal diagnostic artifact restored to the last "
                    "economically verified incumbent"
                )
            stage_results[stage] = {
                "status": "terminal_safe_critic_failure",
                "training_seeds": train_seeds,
                "training": training,
                "price_tail_gate": price_tail,
                "warm_start": warm_start,
                "stage_critic_safety": stage_safety,
                "train_only_safe_policy_selection": None,
                "validation_evaluation": None,
                "diagnostic_artifact_policy": (
                    "economically_verified_incumbent"
                    if verified_incumbent_rollback is not None
                    else "invalid_unverified_diagnostic"
                ),
                "verified_incumbent_rollback": verified_incumbent_rollback,
                "gate": {
                    "name": "same-version held-out TRAIN critic safety gate",
                    "passed": False,
                    "promotion_eligible": False,
                    "terminal": True,
                    "criterion": (
                        "every TD+TRAIN-fit paired-refresh critic version must "
                        "pass before actor/temperature learning"
                    ),
                    "reason": reason,
                },
                "shock_config": shock_config.as_dict(),
                "provenance": stage_provenance,
                "elapsed_seconds": float(time.time() - stage_started),
            }
            print(f"[{stage}] TERMINAL safe-critic failure: {reason}", flush=True)
            break
        train_safe_selection: Optional[Dict[str, Any]] = None
        train_selected_checkpoint: Optional[str] = None
        if actor_acceptance is not None:
            acceptance_summary = actor_acceptance.summary()
            accepted_rows = [
                row
                for row in acceptance_summary["events"]
                if row["actor_proposal"] and row["accepted"]
            ]
            train_safe_selection = {
                "selected": (
                    "accepted_actor_incumbent"
                    if accepted_rows
                    else (
                        "warm_reference_incumbent"
                        if stage == "l0"
                        else "entering_stage_incumbent"
                    )
                ),
                "retained_candidate": bool(accepted_rows),
                "criterion": (
                    "episode-boundary TRAIN-only paired CRN one-sided "
                    "Bonferroni LCB must be strictly positive"
                ),
                "paired_lower_confidence_bound": (
                    float(accepted_rows[-1]["paired_lcb"])
                    if accepted_rows
                    else 0.0
                ),
                "actor_gain_report": acceptance_summary,
                "selection_seeds": actor_accept_flat_seeds,
                "split": "train_actor_accept",
                "validation_used": False,
                "test_used": False,
                "full_incumbent_state_selected_online": True,
            }
            print(
                f"[{stage} actor acceptance] "
                f"selected={train_safe_selection['selected']} "
                f"LCB={train_safe_selection['paired_lower_confidence_bound']:+.4f}",
                flush=True,
            )
            selected_path = out_dir / f"policy_train_selected_after_{stage}.pt"
            agent.save(
                selected_path,
                metadata={
                    "stage": stage,
                    "selection_split": "train_actor_accept",
                    "validation_used_for_selection": False,
                    "test_used_for_selection": False,
                    "selection": train_safe_selection,
                },
            )
            train_selected_checkpoint = str(selected_path)
        evaluation = evaluate_stage(
            stage,
            agent,
            validation_environment_factory,
            validation_fit_seeds,
            validation_gate_seeds,
            cem_config=cem_config,
            run_mpc=bool(args.run_mpc),
            mpc_seeds=int(args.mpc_seeds),
        )
        gate = gate_stage(stage, evaluation)
        if stage == "l0":
            assert warm_start is not None and train_safe_selection is not None
            actor_gain_report = train_safe_selection["actor_gain_report"]
            learner_readiness = _l0_learner_readiness(
                warm_start["post_training_critic_gate"],
                actor_gain_report,
            )
            gate["economic_gate_passed"] = bool(gate["passed"])
            gate["learner_safe_improvement_ready"] = learner_readiness
            gate["learner_safe_improvement_criterion"] = (
                "current critic gate passed and actor proposals were evaluated by "
                "fresh TRAIN-only paired-CRN acceptance; L0 may retain the warm "
                "incumbent when validation remains within 1% of the full-3D rule"
            )
            gate["positive_actor_gain_reported"] = bool(
                actor_gain_report["positive_actor_gain_reported"]
            )
            gate["warm_incumbent_preservation_allowed_at_l0"] = True
            gate["later_rungs_require_positive_adaptive_gain"] = True
            gate["deployed_policy"] = train_safe_selection["selected"]
            gate["fallback_is_terminal_for_later_stages"] = bool(
                not learner_readiness
            )
            gate["passed"] = bool(gate["passed"] and learner_readiness)
        else:
            assert stage_safety is not None and train_safe_selection is not None
            actor_gain_report = train_safe_selection["actor_gain_report"]
            learner_readiness = _adaptive_stage_learner_readiness(
                stage_safety["post_training_critic_gate"],
                actor_gain_report,
            )
            gate["economic_gate_passed"] = bool(gate["passed"])
            gate["learner_safe_improvement_ready"] = learner_readiness
            gate["learner_safe_improvement_criterion"] = (
                "stage-specific current critic gate passed and at least one "
                "adaptive actor proposal achieved a strictly positive "
                "multiplicity-corrected TRAIN-only paired-CRN LCB"
            )
            gate["positive_actor_gain_reported"] = bool(
                actor_gain_report["positive_actor_gain_reported"]
            )
            gate["entering_incumbent_preservation_allowed"] = False
            gate["deployed_policy"] = train_safe_selection["selected"]
            gate["fallback_is_terminal_for_later_stages"] = bool(
                not learner_readiness
            )
            gate["passed"] = bool(gate["passed"] and learner_readiness)
        gate["promotion_eligible"] = bool(args.budget in {"pilot", "full"})
        if args.budget == "smoke":
            gate["statistical_status"] = (
                "underpowered wiring/numerics diagnostic; cannot promote a full run"
            )
        shock_stratified_validation_stress: Optional[Dict[str, Any]] = None
        if stage == "mixed":
            shock_stratified_validation_stress = evaluate_retention(
                stage,
                agent,
                shock_stratified_validation_environment_factory,
                validation_gate_seeds,
                evaluation,
            )
            shock_stratified_validation_stress.update(
                {
                    "split": "validation",
                    "scenario_sampling": "shock_stratified_training_openings",
                    "used_for_rung_gate": False,
                }
            )
        validation_retention: Dict[str, Any] = {}
        retention_checkpoint = bool(
            stage == "mixed"
            or (stage == "l3" and "mixed" not in selected_stages)
        )
        if retention_checkpoint:
            for prior_stage, prior_spec in retention_specs.items():
                retained = evaluate_retention(
                    prior_stage,
                    agent,
                    prior_spec["validation_environment_factory"],
                    prior_spec["validation_gate_seeds"],
                    prior_spec["validation_evaluation"],
                )
                retained["split"] = "validation"
                retained["used_for_continuation"] = True
                validation_retention[prior_stage] = retained
                print(
                    f"[{stage} retention:{prior_stage}] "
                    f"{'PASS' if retained['gate']['passed'] else 'FAIL'}",
                    flush=True,
                )
            retention_passed = bool(
                all(row["gate"]["passed"] for row in validation_retention.values())
            )
            gate["prior_rung_retention"] = {
                "passed": retention_passed,
                "rungs": {
                    name: bool(row["gate"]["passed"])
                    for name, row in validation_retention.items()
                },
            }
            gate["passed"] = bool(gate["passed"] and retention_passed)
        checkpoint = out_dir / (
            f"policy_after_{stage}.pt"
            if gate["passed"]
            else f"policy_diagnostic_after_{stage}.pt"
        )
        agent.save(
            checkpoint,
            metadata={
                "artifact_role": (
                    "passed_stage_policy"
                    if gate["passed"]
                    else "failed_stage_diagnostic"
                ),
                "stage": stage,
                "city": args.city,
                "budget": args.budget,
                "provenance": stage_provenance,
                "environment_contract": probe.contract(),
                "gate": gate,
            },
        )
        stage_results[stage] = {
            "training_seeds": train_seeds,
            "training": training,
            "price_tail_gate": price_tail,
            "warm_start": warm_start,
            "stage_critic_safety": stage_safety,
            "train_only_safe_policy_selection": train_safe_selection,
            "train_selected_checkpoint_before_validation": train_selected_checkpoint,
            "validation_evaluation": evaluation,
            "shock_stratified_validation_stress": shock_stratified_validation_stress,
            "gate": gate,
            "validation_prior_rung_retention": validation_retention,
            "checkpoint": str(checkpoint),
            "shock_config": shock_config.as_dict(),
            "provenance": stage_provenance,
            "elapsed_seconds": float(time.time() - stage_started),
        }
        retention_specs[stage] = {
            "validation_environment_factory": validation_environment_factory,
            "validation_fit_seeds": tuple(validation_fit_seeds),
            "validation_gate_seeds": tuple(validation_gate_seeds),
            "validation_evaluation": evaluation,
            "test_environment_factory": test_environment_factory,
            "shock_stratified_test_environment_factory": (
                shock_stratified_test_environment_factory
            ),
            "test_seeds": tuple(test_seeds),
        }
        print(
            f"[{stage}] gate={'PASS' if gate['passed'] else 'FAIL'} "
            f"RL={evaluation['policies']['recurrent_sac']['mean']:+.3f} "
            f"rule={evaluation['policies']['two_timescale_rule']['mean']:+.3f} "
            f"tariff-static={evaluation['policies']['best_static_tariff']['mean']:+.3f} "
            f"schedule={evaluation['policies']['public_schedule']['mean']:+.3f}",
            flush=True,
        )
        partial = {
            "schema": "ride_response_v28_learner_ladder_v9",
            "city": args.city,
            "budget_name": args.budget,
            "budget": asdict(budget),
            "stages": list(selected_stages),
            "completed_stages": list(stage_results),
            "provenance": provenance,
            "environment_contract": contract,
            "learner_config": asdict(agent.config),
            "exploration_commitment_mean_days": float(
                args.exploration_commitment_mean_days
            ),
            "action_transform": asdict(action_transform),
            "compute_estimate": estimate,
            "stage_results": stage_results,
            "action_diagnostics": agent.action_diagnostics(),
            "elapsed_seconds": float(time.time() - started),
        }
        (out_dir / "report.partial.json").write_text(
            json.dumps(_jsonable(partial), indent=2, allow_nan=False)
        )
        terminal_safe_fallback = bool(
            not gate.get("learner_safe_improvement_ready", False)
        )
        if terminal_safe_fallback:
            print(
                f"[{stage}] terminal safe fallback: later rungs cannot continue "
                "with an unaccepted actor or mismatched post-training critics",
                flush=True,
            )
            break
        if not gate["passed"] and not args.diagnostic_continue:
            print(
                f"[{stage}] fail-fast: rung gate failed; later stages were not run",
                flush=True,
            )
            break

    assert agent is not None
    attempted_stages = list(stage_results)
    trained_stages = list(retention_specs)
    completed_stages = [
        stage
        for stage in trained_stages
        if bool(stage_results[stage]["gate"]["passed"])
    ]
    all_requested_completed = trained_stages == list(selected_stages)
    validation_gates_passed = bool(
        all_requested_completed
        and all(stage_results[stage]["gate"]["passed"] for stage in trained_stages)
    )
    canonical_readiness = ("l0", "l1", "l2", "l3")
    canonical_validation_passed = bool(
        all(stage in retention_specs for stage in canonical_readiness)
        and all(
            stage_results[stage]["gate"]["passed"]
            and stage_results[stage]["gate"].get("promotion_eligible", False)
            for stage in canonical_readiness
        )
    )
    # The test split is opened once, only after all training, optimizer
    # selection, continuation decisions, and validation-retention checks have
    # frozen.  No result below can trigger more training in this run.
    final_test_audit: Dict[str, Any] = {}
    final_test_stress_audit: Dict[str, Any] = {}
    open_final_test = bool(
        args.budget == "full"
        and validation_gates_passed
        and canonical_validation_passed
    )
    if open_final_test:
        for audit_stage, audit_spec in retention_specs.items():
            audited = evaluate_retention(
                audit_stage,
                agent,
                audit_spec["test_environment_factory"],
                audit_spec["test_seeds"],
                audit_spec["validation_evaluation"],
            )
            audited.update(
                {
                    "split": "test",
                    "one_shot": True,
                    "used_for_training_or_continuation": False,
                    "learner_checkpoint": "final_frozen_policy",
                }
            )
            final_test_audit[audit_stage] = audited
            if audit_stage == "mixed":
                stress_audited = evaluate_retention(
                    audit_stage,
                    agent,
                    audit_spec["shock_stratified_test_environment_factory"],
                    audit_spec["test_seeds"],
                    audit_spec["validation_evaluation"],
                )
                stress_audited.update(
                    {
                        "split": "test",
                        "one_shot_suite": True,
                        "scenario_sampling": "shock_stratified_training_openings",
                        "used_for_training_or_continuation": False,
                    }
                )
                final_test_stress_audit[audit_stage] = stress_audited
            print(
                f"[final test:{audit_stage}] "
                f"{'PASS' if audited['gate']['passed'] else 'FAIL'}",
                flush=True,
            )
    required_final_audit = list(canonical_readiness)
    if "mixed" in selected_stages:
        required_final_audit.append("mixed")
    final_retention_passed = bool(
        required_final_audit
        and all(stage in final_test_audit for stage in required_final_audit)
        and all(
            final_test_audit[stage]["gate"]["passed"]
            for stage in required_final_audit
        )
    )
    report = {
        "schema": "ride_response_v28_learner_ladder_v9",
        "city": args.city,
        "budget_name": args.budget,
        "budget": asdict(budget),
        "requested_stages": list(selected_stages),
        "stages": completed_stages,
        "attempted_stages": attempted_stages,
        "provenance": provenance,
        "environment_contract": contract,
        "learner_config": asdict(agent.config),
        "exploration_commitment_mean_days": float(
            args.exploration_commitment_mean_days
        ),
        "action_transform": asdict(action_transform),
        "compute_estimate": estimate,
        "stage_results": stage_results,
        "final_test_audit": final_test_audit,
        "final_test_shock_stratified_stress_audit": final_test_stress_audit,
        "test_split_contract": {
            "opened": bool(final_test_audit),
            "opening_condition": (
                "full budget, canonical L0-L3 and all requested validation "
                "gates passed after training froze"
            ),
            "deferred_to_full_budget": bool(args.budget != "full"),
            "one_shot": True,
            "used_for_training_or_continuation": False,
        },
        "all_gates_passed": bool(
            validation_gates_passed
        ),
        "promotion_ready": bool(
            args.budget in {"pilot", "full"}
            and not args.diagnostic_continue
            and all_requested_completed
            and canonical_validation_passed
            and (args.budget == "pilot" or final_retention_passed)
        ),
        "canonical_l0_l3_validation_passed": canonical_validation_passed,
        "final_test_retention_passed": final_retention_passed,
        "fail_fast_enabled": not bool(args.diagnostic_continue),
        "action_diagnostics": agent.action_diagnostics(),
        "elapsed_seconds": float(time.time() - started),
    }
    figure = plot_report(report, out_dir / "learner_ladder.png")
    report["figure"] = figure
    adaptation_figure = plot_shock_adaptation(
        report, out_dir / "shock_aligned_actions.png"
    )
    report["shock_adaptation_figure"] = adaptation_figure
    promotable_artifact = bool(report["promotion_ready"])
    final_policy_path = out_dir / (
        "policy_final.pt"
        if promotable_artifact
        else "policy_diagnostic_unpromoted.pt"
    )
    report["final_policy_artifact"] = {
        "path": str(final_policy_path),
        "role": (
            "promotion_ready_final_policy"
            if promotable_artifact
            else "quarantined_unpromoted_diagnostic"
        ),
        "promotion_ready": promotable_artifact,
        "generic_resumable_policy": False,
        "economic_incumbent_included": bool(
            agent.economic_incumbent_contract()["available"]
        ),
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False))
    agent.save(
        final_policy_path,
        metadata={
            "artifact_role": report["final_policy_artifact"]["role"],
            "city": args.city,
            "budget": args.budget,
            "requested_stages": list(selected_stages),
            "completed_stages": completed_stages,
            "all_gates_passed": report["all_gates_passed"],
            "promotion_ready": promotable_artifact,
            "report": str(report_path),
        },
    )
    print(f"\nwrote {report_path}")
    if figure:
        print(f"wrote {figure}")
    if adaptation_figure:
        print(f"wrote {adaptation_figure}")
    print(f"all gates passed: {report['all_gates_passed']}")


if __name__ == "__main__":
    main()
