from __future__ import annotations

"""Reproducible result compiler for the three RideResponse contributions."""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


BENCHMARKS = (
    "pi_price_gap",
    "mpc_grid",
    "region_supply_demand",
    "surge_driver_incentive",
    "queue_service_threshold",
)
INCOME_ORDER = ("<50k", "50k-100k", "100k-200k", "200k+")


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    _ensure_parent(path)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite(values: Iterable[object]) -> np.ndarray:
    result: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return np.asarray(result, dtype=float)


def _mean_ci(
    values: Sequence[float],
    *,
    rng: np.random.Generator,
    replicates: int = 1000,
) -> Tuple[float, float, float]:
    array = _finite(values)
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(array))
    if array.size == 1:
        return mean, mean, mean
    draws = rng.integers(0, array.size, size=(replicates, array.size))
    bootstrap = np.mean(array[draws], axis=1)
    return mean, float(np.percentile(bootstrap, 2.5)), float(
        np.percentile(bootstrap, 97.5)
    )


def _moving_block_mean_ci(
    values: Sequence[float],
    *,
    rng: np.random.Generator,
    replicates: int = 1000,
) -> Tuple[float, float, float]:
    array = _finite(values)
    if array.size < 2:
        return _mean_ci(array, rng=rng, replicates=replicates)
    block = int(max(2, round(math.sqrt(array.size))))
    blocks_needed = int(math.ceil(array.size / block))
    starts = rng.integers(0, array.size, size=(replicates, blocks_needed))
    offsets = np.arange(block, dtype=int)
    sampled = array[(starts[:, :, None] + offsets[None, None, :]) % array.size]
    bootstrap = np.mean(sampled.reshape(replicates, -1)[:, : array.size], axis=1)
    return (
        float(np.mean(array)),
        float(np.percentile(bootstrap, 2.5)),
        float(np.percentile(bootstrap, 97.5)),
    )


def _linear_calibration(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """HC3 inference for predicted = intercept + slope * observed price."""
    design = np.column_stack([np.ones(actual.size), actual])
    beta, _, _, _ = np.linalg.lstsq(design, predicted, rcond=None)
    inverse = np.linalg.pinv(design.T @ design)
    residual = predicted - design @ beta
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
    adjusted = residual / np.maximum(1e-8, 1.0 - leverage)
    covariance = inverse @ (
        design.T @ ((adjusted * adjusted)[:, None] * design)
    ) @ inverse
    standard_error = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    return {
        "calibration_intercept": float(beta[0]),
        "calibration_intercept_hc3_ci95_low": float(beta[0] - 1.96 * standard_error[0]),
        "calibration_intercept_hc3_ci95_high": float(beta[0] + 1.96 * standard_error[0]),
        "calibration_slope": float(beta[1]),
        "calibration_slope_hc3_ci95_low": float(beta[1] - 1.96 * standard_error[1]),
        "calibration_slope_hc3_ci95_high": float(beta[1] + 1.96 * standard_error[1]),
    }


def _two_sample_ks(left: np.ndarray, right: np.ndarray) -> float:
    points = np.sort(np.unique(np.concatenate([left, right])))
    left_sorted = np.sort(left)
    right_sorted = np.sort(right)
    return float(np.max(np.abs(
        np.searchsorted(left_sorted, points, side="right") / left_sorted.size
        - np.searchsorted(right_sorted, points, side="right") / right_sorted.size
    )))


def _categorical_percentages(values: Iterable[object]) -> Tuple[List[str], List[float]]:
    counts = Counter(str(value) for value in values)
    labels = sorted(counts)
    total = max(1, sum(counts.values()))
    return labels, [100.0 * counts[label] / total for label in labels]


def _bar_percent(axis: object, labels: Sequence[str], values: Sequence[float], title: str) -> None:
    axis.bar(range(len(labels)), values, color="tab:blue", alpha=0.82)
    axis.set_xticks(range(len(labels)), labels, rotation=28, ha="right")
    axis.set_ylabel("Population share (%)")
    axis.set_title(title)
    axis.grid(axis="y", linestyle="--", alpha=0.25)


def _load_profiles(path: str) -> List[Dict[str, object]]:
    profiles: List[Dict[str, object]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "PriceThreshold" not in record:
                raise ValueError(f"profile {line_number} has no PriceThreshold")
            profiles.append(record)
    if not profiles:
        raise ValueError(f"no profiles found in {path}")
    return profiles


def _robust_ols(
    y: np.ndarray,
    columns: Mapping[str, np.ndarray],
) -> List[Dict[str, object]]:
    names = ["intercept", *columns.keys()]
    design = np.column_stack([np.ones(y.size), *columns.values()])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    inverse = np.linalg.pinv(design.T @ design)
    residual = y - design @ beta
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
    adjusted = residual / np.maximum(1e-8, 1.0 - leverage)
    meat = design.T @ ((adjusted * adjusted)[:, None] * design)
    covariance = inverse @ meat @ inverse
    standard_error = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    total_sum_squares = float(np.sum((y - np.mean(y)) ** 2))
    residual_sum_squares = float(np.sum(residual * residual))
    r_squared = float(
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0.0
        else 0.0
    )
    adjusted_r_squared = float(
        1.0
        - (1.0 - r_squared)
        * (y.size - 1)
        / max(1, y.size - design.shape[1])
    )
    p_values = np.asarray([
        math.erfc(abs(float(coefficient / error)) / math.sqrt(2.0))
        if error > 0.0
        else 0.0
        for coefficient, error in zip(beta, standard_error)
    ])
    adjusted_p = np.ones_like(p_values)
    order = np.argsort(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = float((len(p_values) - rank) * p_values[index])
        running = max(running, candidate)
        adjusted_p[index] = min(1.0, running)
    rows: List[Dict[str, object]] = []
    for name, coefficient, error, p_value, p_holm in zip(
        names, beta, standard_error, p_values, adjusted_p
    ):
        rows.append(
            {
                "term": name,
                "coefficient_dollars": float(coefficient),
                "hc3_standard_error": float(error),
                "ci95_low": float(coefficient - 1.96 * error),
                "ci95_high": float(coefficient + 1.96 * error),
                "p_value_normal_approximation": float(p_value),
                "p_value_holm_adjusted": float(p_holm),
                "n": int(y.size),
                "r_squared": r_squared,
                "adjusted_r_squared": adjusted_r_squared,
            }
        )
    return rows


def analyze_crowdllm(cache_path: str, out_dir: str, seed: int) -> Dict[str, object]:
    profiles = _load_profiles(cache_path)
    rng = np.random.default_rng(seed)
    thresholds = _finite(profile["PriceThreshold"] for profile in profiles)
    sources = Counter(str(profile.get("PriceThresholdSource", "missing")) for profile in profiles)
    rationale_present = np.asarray([
        bool(str(profile.get("PriceThresholdRationale", "")).strip())
        for profile in profiles
    ])
    unique, counts = np.unique(thresholds, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    valid = (thresholds >= 0.50) & (thresholds <= 5.00)
    boundary = np.isclose(thresholds, 0.50) | np.isclose(thresholds, 5.00)
    income_thresholds = {
        level: _finite(
            profile["PriceThreshold"]
            for profile in profiles
            if str(profile["IncomeBracket"]) == level
        )
        for level in INCOME_ORDER
    }
    income_means = [float(np.mean(income_thresholds[level])) for level in INCOME_ORDER]
    income_extreme_difference = float(income_means[-1] - income_means[0])
    pooled_extreme_sd = float(
        np.sqrt(
            (
                np.var(income_thresholds[INCOME_ORDER[0]], ddof=1)
                + np.var(income_thresholds[INCOME_ORDER[-1]], ddof=1)
            )
            / 2.0
        )
    )
    female_thresholds = _finite(
        profile["PriceThreshold"]
        for profile in profiles
        if str(profile.get("Gender", "")) == "Female"
    )
    nonfemale_thresholds = _finite(
        profile["PriceThreshold"]
        for profile in profiles
        if str(profile.get("Gender", "")) != "Female"
    )
    returning_thresholds = _finite(
        profile["PriceThreshold"]
        for profile in profiles
        if str(profile.get("LoyaltyType", "")) == "Returning"
    )
    new_thresholds = _finite(
        profile["PriceThreshold"]
        for profile in profiles
        if str(profile.get("LoyaltyType", "")) == "New"
    )
    large_household_thresholds = _finite(
        profile["PriceThreshold"]
        for profile in profiles
        if int(profile.get("HouseholdSize", 1) or 1) >= 5
    )
    small_household_thresholds = _finite(
        profile["PriceThreshold"]
        for profile in profiles
        if int(profile.get("HouseholdSize", 1) or 1) <= 2
    )
    context_adjustments = _finite(
        profile.get("ThresholdContextAdjustment", float("nan"))
        for profile in profiles
    )
    context_unique, context_counts = np.unique(context_adjustments, return_counts=True)
    expected_context_adjustments = _finite(
        profile.get("ThresholdExpectedContextAdjustment", float("nan"))
        for profile in profiles
    )
    raw_gpt_context_adjustments = _finite(
        profile.get("ThresholdRawGPTContextAdjustment", float("nan"))
        for profile in profiles
    )
    cold_start_ride_count = int(
        sum(len(list(profile.get("ColdstartRides", []))) for profile in profiles)
    )
    summary = {
        "profiles": int(len(profiles)),
        "cold_start_rides": cold_start_ride_count,
        "threshold_source_gpt_rate": float(sources.get("gpt", 0) / len(profiles)),
        "threshold_rationale_nonempty_rate": float(np.mean(rationale_present)),
        "threshold_valid_rate": float(np.mean(valid)),
        "threshold_boundary_mass": float(np.mean(boundary)),
        "threshold_unique_values": int(unique.size),
        "threshold_entropy_nats": entropy,
        "threshold_mean": float(np.mean(thresholds)),
        "threshold_std": float(np.std(thresholds, ddof=1)),
        "threshold_p05": float(np.percentile(thresholds, 5)),
        "threshold_p25": float(np.percentile(thresholds, 25)),
        "threshold_median": float(np.median(thresholds)),
        "threshold_p75": float(np.percentile(thresholds, 75)),
        "threshold_p95": float(np.percentile(thresholds, 95)),
        "threshold_min": float(np.min(thresholds)),
        "threshold_max": float(np.max(thresholds)),
        "income_adjacent_monotone_steps": int(
            sum(right > left for left, right in zip(income_means[:-1], income_means[1:]))
        ),
        "income_adjacent_steps_total": int(len(INCOME_ORDER) - 1),
        "income_high_minus_low_dollars": income_extreme_difference,
        "income_high_vs_low_cohens_d": float(
            income_extreme_difference / pooled_extreme_sd
            if pooled_extreme_sd > 0.0
            else float("nan")
        ),
        "female_minus_nonfemale_raw_dollars": float(
            np.mean(female_thresholds) - np.mean(nonfemale_thresholds)
        ),
        "returning_minus_new_raw_dollars": float(
            np.mean(returning_thresholds) - np.mean(new_thresholds)
        ),
        "large_minus_small_household_raw_dollars": float(
            np.mean(large_household_thresholds) - np.mean(small_household_thresholds)
        ),
        "context_adjustment_unique_values": int(context_unique.size),
        "context_adjustment_largest_class_rate": float(
            np.max(context_counts) / len(context_adjustments)
        ),
        "context_classification_accuracy": float(
            np.mean(np.isclose(context_adjustments, expected_context_adjustments))
        ),
        "raw_gpt_context_classification_accuracy": float(
            np.mean(np.isclose(
                raw_gpt_context_adjustments,
                expected_context_adjustments,
            ))
        ),
        "context_correction_rate": float(np.mean([
            bool(profile.get("ThresholdContextWasCorrected", False))
            for profile in profiles
        ])),
    }
    _write_csv(os.path.join(out_dir, "part1_crowdllm_summary.csv"), [summary])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].hist(thresholds, bins=np.linspace(0.50, 5.00, 19), color="tab:blue", alpha=0.82)
    axes[0].axvline(summary["threshold_median"], color="tab:red", linestyle="--", label=f"median=${summary['threshold_median']:.2f}")
    axes[0].set_xlabel("Fare-gap salience threshold ($)")
    axes[0].set_ylabel("Profiles")
    axes[0].set_title("GPT-generated threshold distribution")
    axes[0].legend(loc="best")
    sorted_thresholds = np.sort(thresholds)
    axes[1].step(sorted_thresholds, np.arange(1, thresholds.size + 1) / thresholds.size, where="post", color="tab:blue")
    axes[1].set_xlabel("Fare-gap salience threshold ($)")
    axes[1].set_ylabel("Cumulative population share")
    axes[1].set_title("Empirical cumulative distribution")
    for axis in axes:
        axis.grid(linestyle="--", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, "part1_crowdllm_threshold_distribution.png")
    _ensure_parent(path)
    fig.savefig(path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes[0, 0].hist(_finite(p["Age"] for p in profiles), bins=18, color="tab:blue", alpha=0.82)
    axes[0, 0].set(title="Age", xlabel="Years", ylabel="Profiles")
    labels, values = _categorical_percentages(p["IncomeBracket"] for p in profiles)
    labels = [label for label in INCOME_ORDER if label in labels]
    income_counts = Counter(str(p["IncomeBracket"]) for p in profiles)
    values = [100.0 * income_counts[label] / len(profiles) for label in labels]
    _bar_percent(axes[0, 1], labels, values, "Income bracket")
    labels, values = _categorical_percentages(p["EmploymentStatus"] for p in profiles)
    _bar_percent(axes[0, 2], labels, values, "Employment")
    labels, values = _categorical_percentages(p["HouseholdSize"] for p in profiles)
    _bar_percent(axes[1, 0], labels, values, "Household size")
    labels, values = _categorical_percentages(p["LoyaltyType"] for p in profiles)
    _bar_percent(axes[1, 1], labels, values, "Loyalty type")
    axes[1, 2].hist(_finite(p["LoyaltyStrength"] for p in profiles), bins=20, color="tab:blue", alpha=0.82)
    axes[1, 2].set(title="Loyalty strength", xlabel="Strength [0, 1]", ylabel="Profiles")
    for axis in axes.flat:
        axis.grid(axis="y", linestyle="--", alpha=0.25)
    fig.suptitle("Sampled CrowdLLM rider demographics")
    fig.tight_layout()
    path = os.path.join(out_dir, "part1_crowdllm_demographics.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)

    rides = [ride for profile in profiles for ride in list(profile.get("ColdstartRides", []))]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    day_counts = Counter(int(ride["DayOfWeek"]) for ride in rides)
    _bar_percent(axes[0, 0], [str(i) for i in range(7)], [100.0 * day_counts[i] / len(rides) for i in range(7)], "Day of week (0=Mon)")
    hour_counts = Counter(int(ride["Hour"]) for ride in rides)
    _bar_percent(axes[0, 1], [str(i) for i in range(24)], [100.0 * hour_counts[i] / len(rides) for i in range(24)], "Hour")
    axes[0, 1].tick_params(axis="x", labelrotation=90)
    labels, values = _categorical_percentages(ride["Weather"] for ride in rides)
    _bar_percent(axes[0, 2], labels, values, "Weather")
    labels, values = _categorical_percentages(bool(ride["Airport"]) for ride in rides)
    _bar_percent(axes[1, 0], labels, values, "Airport ride")
    labels, values = _categorical_percentages(ride["Service"] for ride in rides)
    _bar_percent(axes[1, 1], labels, values, "Service")
    axes[1, 2].hist(_finite(ride["DistanceMiles"] for ride in rides), bins=30, color="tab:blue", alpha=0.82)
    axes[1, 2].set(title="Travel distance", xlabel="Miles", ylabel="Rides")
    for axis in axes.flat:
        axis.grid(axis="y", linestyle="--", alpha=0.25)
    fig.suptitle(f"Cold-start ride natures (n={len(rides):,})")
    fig.tight_layout()
    path = os.path.join(out_dir, "part1_crowdllm_ride_natures.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)

    association_rows: List[Dict[str, object]] = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    groups = [
        ("IncomeBracket", list(INCOME_ORDER), axes[0, 0], "Income bracket"),
        ("LoyaltyType", ["New", "Returning"], axes[0, 1], "Loyalty type"),
        (
            "EmploymentStatus",
            sorted({str(p["EmploymentStatus"]) for p in profiles}),
            axes[1, 0],
            "Employment",
        ),
    ]
    for field, categories, axis, title in groups:
        means: List[float] = []
        low: List[float] = []
        high: List[float] = []
        for category in categories:
            values = [float(p["PriceThreshold"]) for p in profiles if str(p[field]) == category]
            mean, ci_low, ci_high = _mean_ci(values, rng=rng)
            means.append(mean)
            low.append(ci_low)
            high.append(ci_high)
            association_rows.append({"attribute": field, "level": category, "count": len(values), "threshold_mean": mean, "ci95_low": ci_low, "ci95_high": ci_high})
        axis.errorbar(range(len(categories)), means, yerr=[np.asarray(means) - low, np.asarray(high) - means], fmt="o", capsize=4, color="tab:blue")
        axis.set_xticks(range(len(categories)), categories, rotation=25, ha="right")
        axis.set_ylabel("Mean threshold ($)")
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
    loyalty = _finite(p["LoyaltyStrength"] for p in profiles)
    bins = np.linspace(0.0, 1.0, 11)
    bin_index = np.clip(np.digitize(loyalty, bins, right=False) - 1, 0, 9)
    bin_means = [float(np.mean(thresholds[bin_index == index])) if np.any(bin_index == index) else float("nan") for index in range(10)]
    axes[1, 1].plot((bins[:-1] + bins[1:]) / 2.0, bin_means, marker="o", color="tab:blue")
    axes[1, 1].set(title="Threshold by loyalty strength", xlabel="Loyalty strength bin", ylabel="Mean threshold ($)")
    axes[1, 1].grid(linestyle="--", alpha=0.25)
    fig.suptitle("CrowdLLM threshold construct diagnostics (95% bootstrap CI)")
    fig.tight_layout()
    path = os.path.join(out_dir, "part1_crowdllm_construct_diagnostics.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    _write_csv(os.path.join(out_dir, "part1_crowdllm_group_statistics.csv"), association_rows)

    age = _finite(p["Age"] for p in profiles)
    household = _finite(p["HouseholdSize"] for p in profiles)
    strength = _finite(p["LoyaltyStrength"] for p in profiles)
    income = np.asarray([INCOME_ORDER.index(str(p["IncomeBracket"])) for p in profiles], dtype=float)
    returning = np.asarray([float(str(p["LoyaltyType"]) == "Returning") for p in profiles])
    employed = np.asarray([float(str(p["EmploymentStatus"]) == "Employed") for p in profiles])
    student = np.asarray([float(str(p["EmploymentStatus"]) == "Student") for p in profiles])
    female = np.asarray([float(str(p["Gender"]) == "Female") for p in profiles])
    columns = {
        "age_z": (age - age.mean()) / max(age.std(), 1e-8),
        "household_size_z": (household - household.mean()) / max(household.std(), 1e-8),
        "income_ordinal_z": (income - income.mean()) / max(income.std(), 1e-8),
        "returning_rider": returning,
        "loyalty_strength_z": (strength - strength.mean()) / max(strength.std(), 1e-8),
        "employed": employed,
        "student": student,
        "female_protected_attribute": female,
    }
    regression_rows = _robust_ols(thresholds, columns)
    _write_csv(os.path.join(out_dir, "part1_crowdllm_construct_regression.csv"), regression_rows)

    ride_summary: List[Dict[str, object]] = []
    for field in ("DayOfWeek", "Hour", "Weather", "Airport", "Service"):
        counts = Counter(str(ride[field]) for ride in rides)
        for level, count in sorted(counts.items()):
            ride_summary.append({"attribute": field, "level": level, "count": count, "share": count / len(rides)})
    _write_csv(os.path.join(out_dir, "part1_crowdllm_ride_nature_statistics.csv"), ride_summary)
    return summary


def analyze_crowdllm_robustness(
    primary_path: str,
    robustness_specs: Sequence[str],
    rationale_path: str,
    out_dir: str,
) -> List[Dict[str, object]]:
    """Audit marginal and paired prompt stability and retain rationale evidence."""
    populations: List[Tuple[str, List[Dict[str, object]]]] = [
        ("primary", _load_profiles(primary_path))
    ]
    for spec in robustness_specs:
        if "=" not in spec:
            raise ValueError("robustness cache must use label=jsonl_path")
        label, path = spec.split("=", 1)
        populations.append((label, _load_profiles(path)))
    primary_profiles = populations[0][1]
    primary = _finite(row["PriceThreshold"] for row in primary_profiles)
    rows: List[Dict[str, object]] = []
    for label, profiles in populations:
        values = _finite(row["PriceThreshold"] for row in profiles)
        sources = Counter(str(row.get("PriceThresholdSource", "missing")) for row in profiles)
        paired_count = min(len(primary_profiles), len(profiles))
        paired_primary = _finite(
            primary_profiles[index]["PriceThreshold"] for index in range(paired_count)
        )
        paired_values = _finite(
            profiles[index]["PriceThreshold"] for index in range(paired_count)
        )
        paired_error = paired_values - paired_primary
        paired_primary_context = _finite(
            primary_profiles[index].get("ThresholdContextAdjustment", float("nan"))
            for index in range(paired_count)
        )
        paired_context = _finite(
            profiles[index].get("ThresholdContextAdjustment", float("nan"))
            for index in range(paired_count)
        )
        paired_correlation = (
            float(np.corrcoef(paired_primary, paired_values)[0, 1])
            if paired_count > 1
            and np.std(paired_primary) > 0.0
            and np.std(paired_values) > 0.0
            else float("nan")
        )
        rows.append({
            "population": label,
            "profiles": len(profiles),
            "gpt_source_rate": float(sources.get("gpt", 0) / len(profiles)),
            "threshold_mean": float(np.mean(values)),
            "threshold_std": float(np.std(values, ddof=1)),
            "threshold_median": float(np.median(values)),
            "threshold_p05": float(np.percentile(values, 5)),
            "threshold_p95": float(np.percentile(values, 95)),
            "ks_distance_from_primary": (
                0.0 if label == "primary" else _two_sample_ks(primary, values)
            ),
            "absolute_mean_shift_from_primary": float(
                0.0 if label == "primary" else abs(np.mean(values) - np.mean(primary))
            ),
            "paired_profiles": paired_count,
            "paired_threshold_mae": float(np.mean(np.abs(paired_error))),
            "paired_threshold_rmse": float(np.sqrt(np.mean(np.square(paired_error)))),
            "paired_threshold_bias": float(np.mean(paired_error)),
            "paired_threshold_correlation": paired_correlation,
            "paired_exact_match_rate": float(np.mean(np.isclose(paired_error, 0.0))),
            "paired_context_adjustment_mae": float(
                np.mean(np.abs(paired_context - paired_primary_context))
            ),
            "paired_context_exact_match_rate": float(
                np.mean(np.isclose(paired_context, paired_primary_context))
            ),
        })
    _write_csv(os.path.join(out_dir, "part1_crowdllm_robustness.csv"), rows)

    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    for label, profiles in populations:
        values = np.sort(_finite(row["PriceThreshold"] for row in profiles))
        axis.step(
            values,
            np.arange(1, values.size + 1) / values.size,
            where="post",
            label=label,
        )
    axis.set(
        xlabel="Fare-gap salience threshold ($)",
        ylabel="Cumulative population share",
        title="CrowdLLM threshold stability on repeated identical prompts",
    )
    axis.grid(linestyle="--", alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "part1_crowdllm_seed_robustness.png"),
        dpi=180,
    )
    plt.close(fig)

    if rationale_path:
        rationale_profiles = _load_profiles(rationale_path)
        rationale_sources = Counter(
            str(row.get("PriceThresholdSource", "missing"))
            for row in rationale_profiles
        )
        rationale_present = np.asarray([
            bool(str(row.get("PriceThresholdRationale", "")).strip())
            for row in rationale_profiles
        ])
        _write_csv(
            os.path.join(out_dir, "part1_crowdllm_rationale_audit_summary.csv"),
            [{
                "profiles": len(rationale_profiles),
                "gpt_source_rate": float(
                    rationale_sources.get("gpt", 0) / len(rationale_profiles)
                ),
                "rationale_nonempty_rate": float(np.mean(rationale_present)),
                "same_prompt_inputs_as_primary": True,
                "threshold_mean": float(np.mean(_finite(
                    row["PriceThreshold"] for row in rationale_profiles
                ))),
            }],
        )
        rationale_rows = [{
            "profile_id": row.get("ProfileID", index),
            "profile_market": row.get("ProfileMarket", ""),
            "income_bracket": row.get("IncomeBracket", ""),
            "household_size": row.get("HouseholdSize", ""),
            "loyalty_type": row.get("LoyaltyType", ""),
            "loyalty_strength": row.get("LoyaltyStrength", ""),
            "price_threshold": row.get("PriceThreshold", ""),
            "threshold_source": row.get("PriceThresholdSource", ""),
            "rationale": row.get("PriceThresholdRationale", ""),
        } for index, row in enumerate(rationale_profiles)]
        _write_csv(
            os.path.join(out_dir, "part1_crowdllm_rationale_audit_sample.csv"),
            rationale_rows,
        )
    return rows


def analyze_benchmarks(prefix: str, out_dir: str, seed: int) -> List[Dict[str, object]]:
    rng = np.random.default_rng(seed)
    summary_rows = _read_csv(f"{prefix}_benchmark_summary.csv")
    summary_by_mode = {row["benchmark"]: row for row in summary_rows}
    output_rows: List[Dict[str, object]] = []
    for benchmark in BENCHMARKS:
        rows = _read_csv(f"{prefix}_benchmark_{benchmark}_trajectory.csv")
        profit_gap = _finite(row["profit_gap_per_request"] for row in rows)
        share_gap = _finite(row["completed_share_gap"] for row in rows)
        f1_profit = _finite(row["firm1_profit_per_request"] for row in rows)
        f2_profit = _finite(row["firm2_profit_per_request"] for row in rows)
        f1_share = _finite(row["firm1_completed_share"] for row in rows)
        f2_share = _finite(row["firm2_completed_share"] for row in rows)
        profit_mean, profit_low, profit_high = _moving_block_mean_ci(profit_gap, rng=rng)
        share_mean, share_low, share_high = _moving_block_mean_ci(share_gap, rng=rng)
        reported = summary_by_mode[benchmark]
        output_rows.append({
            "benchmark": benchmark,
            "steps": len(rows),
            "firm1_profit_mean": float(np.mean(f1_profit)),
            "firm2_profit_mean": float(np.mean(f2_profit)),
            "profit_gap_mean": profit_mean,
            "profit_gap_block_ci95_low": profit_low,
            "profit_gap_block_ci95_high": profit_high,
            "firm1_completed_share_mean": float(np.mean(f1_share)),
            "firm2_completed_share_mean": float(np.mean(f2_share)),
            "share_gap_mean": share_mean,
            "share_gap_block_ci95_low": share_low,
            "share_gap_block_ci95_high": share_high,
            "profit_dominance_rate": float(np.mean(profit_gap > 0.0)),
            "share_dominance_rate": float(np.mean(share_gap > 0.0)),
            "late_objective_advantage": float(reported["final_quarter_advantage"]),
            "rolling_objective_dominance_rate": float(reported["rolling_dominance_rate"]),
            "firm1_fulfillment_rate": float(reported["firm1_fulfillment_rate"]),
            "completion_coverage": float(reported["completion_coverage"]),
            "unserved_demand_rate": float(reported["unserved_demand_rate"]),
            "mean_price_gap_abs_error": float(reported["mean_price_gap_abs_error"]),
            "selected_action_feasibility_rate": float(
                reported["selected_action_feasibility_rate"]
            ),
            "passes_objective": str(reported["passes_objective"]).lower() == "true",
        })
    _write_csv(os.path.join(out_dir, "part2_benchmark_key_statistics.csv"), output_rows)

    labels = [name.replace("_", "\n") for name in BENCHMARKS]
    x = np.arange(len(BENCHMARKS))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes[0, 0].bar(x - 0.18, [row["firm1_profit_mean"] for row in output_rows], 0.36, label="Firm 1", color="tab:blue")
    axes[0, 0].bar(x + 0.18, [row["firm2_profit_mean"] for row in output_rows], 0.36, label="Benchmark", color="tab:orange")
    axes[0, 0].set_ylabel("Contribution profit / request ($)")
    axes[0, 0].legend(loc="best")
    profit_means = np.asarray([row["profit_gap_mean"] for row in output_rows], dtype=float)
    axes[0, 1].errorbar(x, profit_means, yerr=[profit_means - [row["profit_gap_block_ci95_low"] for row in output_rows], [row["profit_gap_block_ci95_high"] for row in output_rows] - profit_means], fmt="o", capsize=4, color="tab:blue")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.9)
    axes[0, 1].set_ylabel("Profit advantage / request ($)")
    axes[1, 0].bar(x - 0.18, [100.0 * row["firm1_completed_share_mean"] for row in output_rows], 0.36, label="Firm 1", color="tab:blue")
    axes[1, 0].bar(x + 0.18, [100.0 * row["firm2_completed_share_mean"] for row in output_rows], 0.36, label="Benchmark", color="tab:orange")
    axes[1, 0].set_ylabel("Completed rides / incoming request (%)")
    axes[1, 0].legend(loc="best")
    share_means = 100.0 * np.asarray([row["share_gap_mean"] for row in output_rows], dtype=float)
    axes[1, 1].errorbar(x, share_means, yerr=[share_means - 100.0 * np.asarray([row["share_gap_block_ci95_low"] for row in output_rows]), 100.0 * np.asarray([row["share_gap_block_ci95_high"] for row in output_rows]) - share_means], fmt="o", capsize=4, color="tab:blue")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.9)
    axes[1, 1].set_ylabel("Completed-share advantage (pp)")
    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
    fig.suptitle("Firm 1 policy performance against five benchmark controllers")
    fig.tight_layout()
    path = os.path.join(out_dir, "part2_benchmark_performance.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    axes[0].bar(x, 100.0 * np.asarray([
        row["firm1_fulfillment_rate"] for row in output_rows
    ]), color="tab:blue", alpha=0.82)
    axes[0].set_ylabel("Firm 1 fulfillment (%)")
    axes[1].bar(x, 100.0 * np.asarray([
        row["unserved_demand_rate"] for row in output_rows
    ]), color="tab:orange", alpha=0.82)
    axes[1].set_ylabel("Market unserved demand (%)")
    axes[2].bar(x, 100.0 * np.asarray([
        row["selected_action_feasibility_rate"] for row in output_rows
    ]), color="tab:green", alpha=0.82)
    axes[2].set_ylabel("Selected actions feasible (%)")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
    fig.suptitle("Service and policy-feasibility diagnostics by benchmark")
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "part2_benchmark_service_diagnostics.png"),
        dpi=180,
    )
    plt.close(fig)
    return output_rows


def analyze_dataset_comparisons(
    specs: Sequence[str],
    out_dir: str,
    seed: int,
) -> List[Dict[str, object]]:
    comparisons: List[Tuple[str, str, List[Dict[str, str]]]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError("dataset comparison must use label=csv_path")
        label, path = spec.split("=", 1)
        comparisons.append((label, path, _read_csv(path)))
    if len(comparisons) != 5:
        raise ValueError("exactly five objective-mixture comparisons are required")

    summaries: List[Dict[str, object]] = []
    quantile_rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for label, path, rows in comparisons:
        paired_prices: List[Tuple[float, float, float]] = []
        for row in rows:
            try:
                triple = (
                    float(row["actual_paid"]),
                    float(row["rl_predicted_price"]),
                    float(row["anchor_tariff_price"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in triple):
                paired_prices.append(triple)
        if not paired_prices:
            raise ValueError(f"invalid price comparison: {path}")
        price_matrix = np.asarray(paired_prices, dtype=float)
        actual, predicted, anchor = price_matrix.T
        error = predicted - actual
        absolute_error = np.abs(error)
        anchor_absolute_error = np.abs(anchor - actual)
        mae, mae_low, mae_high = _mean_ci(absolute_error, rng=rng)
        anchor_mae, anchor_mae_low, anchor_mae_high = _mean_ci(
            anchor_absolute_error,
            rng=rng,
        )
        improvement, improvement_low, improvement_high = _mean_ci(
            anchor_absolute_error - absolute_error,
            rng=rng,
        )
        predicted_mean, predicted_mean_low, predicted_mean_high = _mean_ci(
            predicted,
            rng=rng,
        )
        correlation = float(np.corrcoef(actual, predicted)[0, 1])
        action_counter = Counter(str(row.get("policy_action_label", "missing")) for row in rows)
        action_probabilities = np.asarray(list(action_counter.values()), dtype=float) / len(rows)
        action_entropy = float(-np.sum(action_probabilities * np.log(action_probabilities)))
        top_action, top_action_count = action_counter.most_common(1)[0]
        calibration = _linear_calibration(actual, predicted)
        quantile_errors: List[float] = []
        for probability in (0.10, 0.25, 0.50, 0.75, 0.90):
            actual_quantile = float(np.quantile(actual, probability))
            predicted_quantile = float(np.quantile(predicted, probability))
            quantile_error = predicted_quantile - actual_quantile
            quantile_errors.append(abs(quantile_error))
            quantile_rows.append({
                "objective": label,
                "quantile": probability,
                "actual_price": actual_quantile,
                "predicted_price": predicted_quantile,
                "prediction_minus_actual": quantile_error,
            })
        summaries.append({
            "objective": label,
            "rows": int(actual.size),
            "price_mae": mae,
            "price_mae_bootstrap_ci95_low": mae_low,
            "price_mae_bootstrap_ci95_high": mae_high,
            "price_rmse": float(np.sqrt(np.mean(error * error))),
            "price_bias": float(np.mean(error)),
            "price_correlation": correlation,
            **calibration,
            "fare_quantile_mae": float(np.mean(quantile_errors)),
            "anchor_price_mae": anchor_mae,
            "anchor_price_mae_bootstrap_ci95_low": anchor_mae_low,
            "anchor_price_mae_bootstrap_ci95_high": anchor_mae_high,
            "mae_improvement_over_anchor": improvement,
            "mae_improvement_bootstrap_ci95_low": improvement_low,
            "mae_improvement_bootstrap_ci95_high": improvement_high,
            "actual_price_mean": float(np.mean(actual)),
            "predicted_price_mean": predicted_mean,
            "predicted_price_mean_bootstrap_ci95_low": predicted_mean_low,
            "predicted_price_mean_bootstrap_ci95_high": predicted_mean_high,
            "unique_policy_actions": len(action_counter),
            "policy_action_entropy_nats": action_entropy,
            "top_policy_action": top_action,
            "top_policy_action_share": float(top_action_count / len(rows)),
        })
    _write_csv(os.path.join(out_dir, "part3_kaggle_objective_summary.csv"), summaries)
    _write_csv(
        os.path.join(out_dir, "part3_fare_quantile_calibration.csv"),
        quantile_rows,
    )

    keyed_predictions: List[Dict[Tuple[str, str], float]] = []
    keyed_actuals: List[Dict[Tuple[str, str], float]] = []
    for label, path, rows in comparisons:
        predictions_for_objective: Dict[Tuple[str, str], float] = {}
        actuals_for_objective: Dict[Tuple[str, str], float] = {}
        for row in rows:
            key = (
                str(row.get("source_path", row.get("source_file", ""))),
                str(row.get("source_stream_position", "")),
            )
            try:
                actual_value = float(row["actual_paid"])
                predicted_value = float(row["rl_predicted_price"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(actual_value) and math.isfinite(predicted_value)):
                continue
            if key in predictions_for_objective:
                raise ValueError(f"duplicate Kaggle record key in {path}: {key}")
            predictions_for_objective[key] = predicted_value
            actuals_for_objective[key] = actual_value
        keyed_predictions.append(predictions_for_objective)
        keyed_actuals.append(actuals_for_objective)
    common_keys = set(keyed_predictions[0])
    if any(set(mapping) != common_keys for mapping in keyed_predictions[1:]):
        raise ValueError("objective comparisons do not contain the same Kaggle records")
    ordered_keys = sorted(common_keys)
    for key in ordered_keys:
        actual_values = [mapping[key] for mapping in keyed_actuals]
        if not np.allclose(actual_values, actual_values[0], rtol=0.0, atol=1e-10):
            raise ValueError(f"actual price mismatch across objective files: {key}")
    prediction_matrix = np.column_stack([
        np.asarray([mapping[key] for key in ordered_keys], dtype=float)
        for mapping in keyed_predictions
    ])
    endpoint_difference = prediction_matrix[:, -1] - prediction_matrix[:, 0]
    endpoint_mean, endpoint_low, endpoint_high = _mean_ci(
        endpoint_difference,
        rng=rng,
    )
    objective_differences = np.diff(prediction_matrix, axis=1)
    paired_summary = {
        "common_kaggle_rows": int(prediction_matrix.shape[0]),
        "p90_minus_p10_mean_predicted_price": endpoint_mean,
        "p90_minus_p10_bootstrap_ci95_low": endpoint_low,
        "p90_minus_p10_bootstrap_ci95_high": endpoint_high,
        "ride_share_monotone_nondecreasing_with_profit_weight": float(
            np.mean(np.all(objective_differences >= -1e-10, axis=1))
        ),
        "ride_share_monotone_nonincreasing_with_profit_weight": float(
            np.mean(np.all(objective_differences <= 1e-10, axis=1))
        ),
        "ride_share_with_any_objective_price_difference": float(
            np.mean(np.ptp(prediction_matrix, axis=1) > 1e-10)
        ),
    }
    _write_csv(
        os.path.join(out_dir, "part3_paired_objective_response.csv"),
        [paired_summary],
    )

    all_actual = np.concatenate([_finite(row["actual_paid"] for row in rows) for _, _, rows in comparisons])
    all_predicted = np.concatenate([_finite(row["rl_predicted_price"] for row in rows) for _, _, rows in comparisons])
    lower = float(min(np.min(all_actual), np.min(all_predicted)))
    upper = float(max(np.max(all_actual), np.max(all_predicted)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    for axis, (label, _, rows), summary in zip(axes.flat, comparisons, summaries):
        actual = _finite(row["actual_paid"] for row in rows)
        predicted = _finite(row["rl_predicted_price"] for row in rows)
        axis.scatter(actual, predicted, s=10, alpha=0.35, color="tab:blue", linewidths=0)
        axis.plot([lower, upper], [lower, upper], color="tab:red", linestyle="--", linewidth=1.2)
        axis.set_title(f"{label}\nMAE=${summary['price_mae']:.2f}, r={summary['price_correlation']:.3f}")
        axis.grid(linestyle="--", alpha=0.20)
    axes.flat[-1].axis("off")
    for axis in axes[:, 0]:
        axis.set_ylabel("Predicted price ($)")
    for axis in axes[-1, :2]:
        axis.set_xlabel("Observed Kaggle price ($)")
    for axis in axes.flat[:-1]:
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
    fig.suptitle("Kaggle price predictions across objective mixtures (common 5,000-ride sample)")
    fig.tight_layout()
    path = os.path.join(out_dir, "part3_kaggle_price_scatter_all_objectives.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)

    profit_shares = np.asarray([10, 30, 50, 70, 90], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(profit_shares, [row["price_mae"] for row in summaries], marker="o", color="tab:blue")
    axes[0].axhline(
        float(np.mean([row["anchor_price_mae"] for row in summaries])),
        color="tab:red",
        linestyle="--",
        linewidth=1.1,
        label="anchor tariff",
    )
    axes[0].set_ylabel("Price MAE ($)")
    axes[0].legend(loc="best")
    axes[1].plot(profit_shares, [row["price_bias"] for row in summaries], marker="o", color="tab:orange")
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[1].set_ylabel("Mean price bias ($)")
    axes[2].plot(profit_shares, [row["price_correlation"] for row in summaries], marker="o", color="tab:green")
    axes[2].set_ylabel("Price correlation")
    for axis in axes:
        axis.set_xlabel("Profit objective weight (%)")
        axis.grid(linestyle="--", alpha=0.25)
    fig.suptitle("Objective preference and out-of-sample price behavior")
    fig.tight_layout()
    path = os.path.join(out_dir, "part3_objective_price_tradeoff.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)

    predicted_means = np.asarray(
        [row["predicted_price_mean"] for row in summaries],
        dtype=float,
    )
    predicted_low = np.asarray(
        [row["predicted_price_mean_bootstrap_ci95_low"] for row in summaries],
        dtype=float,
    )
    predicted_high = np.asarray(
        [row["predicted_price_mean_bootstrap_ci95_high"] for row in summaries],
        dtype=float,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].errorbar(
        profit_shares,
        predicted_means,
        yerr=[predicted_means - predicted_low, predicted_high - predicted_means],
        marker="o",
        capsize=4,
        color="tab:blue",
    )
    axes[0].set(xlabel="Profit objective weight (%)", ylabel="Mean predicted price ($)")
    axes[0].grid(linestyle="--", alpha=0.25)
    axes[1].hist(endpoint_difference, bins=30, color="tab:purple", alpha=0.82)
    axes[1].axvline(0.0, color="black", linewidth=0.9)
    axes[1].set(
        xlabel="P90/C10 minus P10/C90 predicted price ($)",
        ylabel="Kaggle rides",
    )
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)
    fig.suptitle("Paired objective response on the common Kaggle ride sample")
    fig.tight_layout()
    path = os.path.join(out_dir, "part3_paired_objective_response.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return summaries


def analyze_training_progress(
    specs: Sequence[str],
    out_dir: str,
) -> List[Dict[str, object]]:
    if len(specs) != 5:
        raise ValueError("exactly five objective-mixture training prefixes are required")
    progress_rows: List[Dict[str, object]] = []
    validation_rows: List[Dict[str, object]] = []
    exploration_rows: List[Dict[str, object]] = []
    run_summary_rows: List[Dict[str, object]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError("training prefix must use label=report_prefix")
        label, prefix = spec.split("=", 1)
        rows = _read_csv(f"{prefix}_training_diagnostics.csv")
        if len(rows) != 9150:
            raise ValueError(f"expected 9,150 training rows for {label}; got {len(rows)}")
        dynamic_rows = [
            row for row in rows if row.get("training_stage") == "dynamic_opponent"
        ]
        if len(dynamic_rows) != 8235:
            raise ValueError(
                f"expected 8,235 dynamic-stage rows for {label}; got {len(dynamic_rows)}"
            )
        for cycle_index in range(9):
            cycle = dynamic_rows[cycle_index * 915 : (cycle_index + 1) * 915]
            summary: Dict[str, object] = {
                "objective": label,
                "population_cycle": cycle_index + 1,
                "training_day_end": int(float(cycle[-1]["batch"])) + 1,
            }
            for output_name, source_name in (
                ("objective_reward_mean", "avg_reward"),
                ("revenue_per_request_mean", "avg_rev_per_request"),
                ("profit_per_request_mean", "avg_profit_per_request"),
                ("profit_advantage_per_request_mean", "avg_profit_advantage_per_request"),
                ("firm1_completed_share_mean", "avg_completed_share"),
                ("firm2_completed_share_mean", "avg_firm2_completed_share"),
            ):
                values = _finite(row[source_name] for row in cycle)
                summary[output_name] = float(np.mean(values))
            progress_rows.append(summary)
        for row in rows:
            if row.get("validation_score", "") not in ("", "nan", "NaN"):
                validation_rows.append({
                    "objective": label,
                    "training_day": int(float(row["batch"])) + 1,
                    "validation_score": float(row["validation_score"]),
                    "heldout_critic_td_explained_variance": float(
                        row["validation_critic_td_explained_variance"]
                    ),
                    "heldout_critic_td_mae": float(row["validation_critic_td_mae"]),
                    "validation_feasible_mode_fraction": float(
                        row["validation_feasible_mode_fraction"]
                    ),
                    "validation_action_diversity": float(
                        row["validation_action_diversity"]
                    ),
                    "validation_action_switch_rate": float(
                        row["validation_action_switch_rate"]
                    ),
                    "validation_policy_dynamicity": float(
                        row["validation_policy_dynamicity"]
                    ),
                })
            if row.get("ppo_update_performed", "").lower() == "true":
                exploration_rows.append({
                    "objective": label,
                    "training_day": int(float(row["batch"])) + 1,
                    "exploration_rate": float(row["ppo_exploration_rate"]),
                    "entropy_coefficient": float(row["ppo_ent_coeff"]),
                    "policy_entropy_fraction": float(row["ppo_policy_entropy_fraction"]),
                    "raw_argmax_concentration": float(
                        row["ppo_raw_argmax_concentration"]
                    ),
                    "state_action_sensitivity": float(
                        row["ppo_state_action_sensitivity"]
                    ),
                    "rollout_action_diversity": float(
                        row["ppo_rollout_action_diversity"]
                    ),
                    "on_policy_critic_explained_variance": float(
                        row["ppo_explained_variance"]
                    ),
                })
        objective_population = [
            row for row in progress_rows if row["objective"] == label
        ]
        objective_validation = [
            row for row in validation_rows if row["objective"] == label
        ]
        objective_exploration = [
            row for row in exploration_rows if row["objective"] == label
        ]
        with open(f"{prefix}_run_config.json", encoding="utf-8") as handle:
            run_config = json.load(handle)
        benchmark_path = f"{prefix}_benchmark_summary.csv"
        benchmark_rows = (
            _read_csv(benchmark_path) if os.path.exists(benchmark_path) else []
        )
        reward = dict(run_config.get("active_reward", {}) or {})
        backtest = dict(run_config.get("competitive_backtest", {}) or {})
        run_args = dict(run_config.get("args", {}) or {})
        credit_assignment = dict(
            run_config.get("credit_assignment", {}) or {}
        )
        decision_rows = [
            row for row in rows
            if int(float(row.get("raw_policy_top_action", -1))) >= 0
        ]
        if not decision_rows:
            raise ValueError(f"no policy decision diagnostics for {label}")
        late_decisions = decision_rows[-max(1, len(decision_rows) // 10):]
        late_argmax_counts = Counter(
            int(float(row["raw_policy_top_action"]))
            for row in late_decisions
        )
        late_argmax_action, late_argmax_count = late_argmax_counts.most_common(1)[0]
        run_summary_rows.append({
            "objective": label,
            "policy_objective": run_args.get("policy_objective", ""),
            "effective_seed": run_config.get("effective_seed", ""),
            "training_rows": len(rows),
            "evaluation_rows": len(_read_csv(f"{prefix}_evaluation_diagnostics.csv")),
            "benchmark_rows": len(benchmark_rows),
            "own_profit_weight": reward.get("own_profit_weight", ""),
            "profit_lead_weight": reward.get("profit_advantage_weight", ""),
            "own_completed_share_weight": reward.get("market_share_level_weight", ""),
            "completed_share_lead_weight": reward.get(
                "market_share_competitiveness_weight",
                "",
            ),
            "ppo_updates": len(objective_exploration),
            "actor_initialization_contract": credit_assignment.get(
                "actor_initialization_contract",
                "",
            ),
            "initial_deterministic_action": int(float(
                decision_rows[0]["raw_policy_top_action"]
            )),
            "late_deterministic_action_mode": int(late_argmax_action),
            "late_deterministic_action_mode_rate": float(
                late_argmax_count / len(late_decisions)
            ),
            "exploration_initial": objective_exploration[0]["exploration_rate"],
            "exploration_minimum": float(min(
                row["exploration_rate"] for row in objective_exploration
            )),
            "exploration_final_update": objective_exploration[-1]["exploration_rate"],
            "validation_score_initial": objective_validation[0]["validation_score"],
            "validation_score_final": objective_validation[-1]["validation_score"],
            "validation_score_best": float(max(
                row["validation_score"] for row in objective_validation
            )),
            "validation_score_change": float(
                objective_validation[-1]["validation_score"]
                - objective_validation[0]["validation_score"]
            ),
            "heldout_critic_td_ev_initial": objective_validation[0][
                "heldout_critic_td_explained_variance"
            ],
            "heldout_critic_td_ev_final": objective_validation[-1][
                "heldout_critic_td_explained_variance"
            ],
            "validation_action_diversity_final": objective_validation[-1][
                "validation_action_diversity"
            ],
            "validation_action_switch_rate_final": objective_validation[-1][
                "validation_action_switch_rate"
            ],
            "validation_policy_dynamicity_final": objective_validation[-1][
                "validation_policy_dynamicity"
            ],
            "state_action_sensitivity_initial_update": objective_exploration[0][
                "state_action_sensitivity"
            ],
            "state_action_sensitivity_final_update": objective_exploration[-1][
                "state_action_sensitivity"
            ],
            "cycle1_profit_per_request": objective_population[0][
                "profit_per_request_mean"
            ],
            "cycle9_profit_per_request": objective_population[-1][
                "profit_per_request_mean"
            ],
            "cycle1_profit_advantage": objective_population[0][
                "profit_advantage_per_request_mean"
            ],
            "cycle9_profit_advantage": objective_population[-1][
                "profit_advantage_per_request_mean"
            ],
            "deployment_backtest_passed": bool(backtest.get("passed", False)),
            "late_objective_advantage": backtest.get("late_objective_advantage", ""),
            "late_objective_rolling_dominance_rate": backtest.get(
                "late_objective_rolling_dominance_rate",
                "",
            ),
            "benchmarks_passed": int(sum(
                str(row["passes_objective"]).lower() == "true"
                for row in benchmark_rows
            )),
            "worst_benchmark_final_quarter_advantage": (
                float(min(
                    float(row["final_quarter_advantage"])
                    for row in benchmark_rows
                ))
                if benchmark_rows else ""
            ),
        })
    _write_csv(
        os.path.join(out_dir, "supplemental_training_population_cycles.csv"),
        progress_rows,
    )
    _write_csv(
        os.path.join(out_dir, "supplemental_training_validation_progress.csv"),
        validation_rows,
    )
    _write_csv(
        os.path.join(out_dir, "supplemental_training_exploration_progress.csv"),
        exploration_rows,
    )
    _write_csv(
        os.path.join(out_dir, "supplemental_objective_run_summary.csv"),
        run_summary_rows,
    )

    labels = [spec.split("=", 1)[0] for spec in specs]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for label in labels:
        population = [row for row in progress_rows if row["objective"] == label]
        validation = [row for row in validation_rows if row["objective"] == label]
        exploration = [row for row in exploration_rows if row["objective"] == label]
        axes[0, 0].plot(
            [row["population_cycle"] for row in population],
            [row["profit_advantage_per_request_mean"] for row in population],
            marker="o",
            label=label,
        )
        axes[0, 1].plot(
            [row["training_day"] for row in validation],
            [row["validation_score"] for row in validation],
            marker="o",
            label=label,
        )
        axes[1, 0].plot(
            [row["training_day"] for row in validation],
            [row["heldout_critic_td_explained_variance"] for row in validation],
            marker="o",
            label=label,
        )
        axes[1, 1].plot(
            [row["training_day"] for row in exploration],
            [row["exploration_rate"] for row in exploration],
            marker="o",
            label=label,
        )
    axes[0, 0].axhline(0.0, color="black", linewidth=0.9)
    axes[0, 0].set(
        xlabel="Complete dynamic-opponent population cycle",
        ylabel="Mean profit advantage / request ($)",
    )
    axes[0, 1].set(xlabel="Training day", ylabel="Held-out objective score")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.9)
    axes[1, 0].set(xlabel="Training day", ylabel="Held-out critic TD explained variance")
    axes[1, 1].set(xlabel="Training day", ylabel="Behavior exploration rate")
    for axis in axes.flat:
        axis.grid(linestyle="--", alpha=0.25)
    axes[0, 0].legend(loc="best", ncol=2)
    fig.suptitle("Learning progress across the five objective-mixture policies")
    fig.tight_layout()
    path = os.path.join(out_dir, "supplemental_training_progress.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for label in labels:
        validation = [row for row in validation_rows if row["objective"] == label]
        exploration = [row for row in exploration_rows if row["objective"] == label]
        axes[0, 0].plot(
            [row["training_day"] for row in validation],
            [row["validation_action_diversity"] for row in validation],
            marker="o",
            label=label,
        )
        axes[0, 1].plot(
            [row["training_day"] for row in validation],
            [row["validation_policy_dynamicity"] for row in validation],
            marker="o",
            label=label,
        )
        axes[1, 0].plot(
            [row["training_day"] for row in exploration],
            [row["state_action_sensitivity"] for row in exploration],
            marker="o",
            label=label,
        )
        axes[1, 1].plot(
            [row["training_day"] for row in exploration],
            [row["raw_argmax_concentration"] for row in exploration],
            marker="o",
            label=label,
        )
    axes[0, 0].set(xlabel="Training day", ylabel="Validation action diversity")
    axes[0, 1].set(xlabel="Training day", ylabel="Validation policy dynamicity")
    axes[1, 0].set(xlabel="Training day", ylabel="State-action sensitivity")
    axes[1, 1].set(xlabel="Training day", ylabel="Raw argmax concentration")
    for axis in axes.flat:
        axis.grid(linestyle="--", alpha=0.25)
    axes[0, 0].legend(loc="best", ncol=2)
    fig.suptitle("Policy-learning and adaptive-response diagnostics")
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "supplemental_policy_learning_diagnostics.png"),
        dpi=180,
    )
    plt.close(fig)
    return progress_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold_cache", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--benchmark_prefix", default="")
    parser.add_argument("--robustness_cache", action="append", default=[])
    parser.add_argument("--rationale_cache", default="")
    parser.add_argument("--dataset_comparison", action="append", default=[])
    parser.add_argument("--training_prefix", action="append", default=[])
    parser.add_argument("--seed", type=int, default=104729)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    crowd = analyze_crowdllm(args.threshold_cache, args.out_dir, args.seed)
    print("[Part 1] " + json.dumps(crowd, sort_keys=True))
    if args.robustness_cache or args.rationale_cache:
        robustness = analyze_crowdllm_robustness(
            args.threshold_cache,
            args.robustness_cache,
            args.rationale_cache,
            args.out_dir,
        )
        print("[Part 1 robustness] " + json.dumps(robustness, sort_keys=True))
    if args.benchmark_prefix:
        benchmark = analyze_benchmarks(args.benchmark_prefix, args.out_dir, args.seed + 1)
        print("[Part 2] " + json.dumps(benchmark, sort_keys=True))
    if args.dataset_comparison:
        dataset = analyze_dataset_comparisons(
            args.dataset_comparison,
            args.out_dir,
            args.seed + 2,
        )
        print("[Part 3] " + json.dumps(dataset, sort_keys=True))
    if args.training_prefix:
        progress = analyze_training_progress(args.training_prefix, args.out_dir)
        print("[Training progress] " + json.dumps(progress, sort_keys=True))


if __name__ == "__main__":
    main()
