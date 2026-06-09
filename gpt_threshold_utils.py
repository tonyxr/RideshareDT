#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 08:28:07 2026

@author: Xiaoru Shi
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping


GPT_THRESHOLD_COUNTER_KEYS = (
    "batches_total",
    "batches_skipped_no_key",
    "batches_attempted",
    "batches_succeeded",
    "batches_partial",
    "batches_failed",
    "profiles_requested",
    "profiles_gpt",
    "profiles_fallback",
)


def new_gpt_threshold_usage_counts() -> Dict[str, int]:
    """Create a fresh GPT threshold utilization counter dictionary."""
    return {key: 0 for key in GPT_THRESHOLD_COUNTER_KEYS}


def increment_gpt_threshold_usage(counts: MutableMapping[str, int], key: str, amount: int = 1) -> None:
    """Increment a GPT threshold utilization counter, tolerating older saved key maps."""
    if key not in GPT_THRESHOLD_COUNTER_KEYS:
        raise KeyError(f"Unknown GPT threshold usage counter: {key}")
    counts[key] = int(counts.get(key, 0)) + int(amount)


def clip_price_threshold(value: Any, lo: float = 0.50, hi: float = 5.00) -> float:
    """Convert a threshold to float and clamp it to the GPT bootstrap contract."""
    value_f = float(value)
    if value_f < lo:
        return float(lo)
    if value_f > hi:
        return float(hi)
    return float(value_f)


def build_threshold_profile(
    profile: Mapping[str, Any],
    coldstart_rides: Iterable[Mapping[str, Any]],
    threshold: Mapping[str, Any],
) -> Dict[str, Any]:
    """Attach a generated/fallback price threshold to a rider profile for later simulation use."""
    return {
        **dict(profile),
        "ColdstartRides": list(coldstart_rides),
        "PriceThreshold": clip_price_threshold(threshold["price_threshold"]),
        "PriceThresholdRationale": threshold.get("rationale", ""),
        "PriceThresholdSource": str(threshold.get("source", "fallback")),
    }


def format_gpt_threshold_usage_summary(counts: Mapping[str, int]) -> str:
    """Format non-zero utilization counters plus success/fallback percentages for run logs."""
    parts: List[str] = [
        f"{key}={int(counts.get(key, 0))}"
        for key in GPT_THRESHOLD_COUNTER_KEYS
        if int(counts.get(key, 0))
    ]

    attempted = int(counts.get("batches_attempted", 0))
    successful_batches = int(counts.get("batches_succeeded", 0)) + int(counts.get("batches_partial", 0))
    if attempted > 0:
        parts.append(f"batch_success_rate={100.0 * successful_batches / attempted:.1f}%")

    requested = int(counts.get("profiles_requested", 0))
    if requested > 0:
        parts.append(f"profile_gpt_rate={100.0 * int(counts.get('profiles_gpt', 0)) / requested:.1f}%")
        parts.append(f"profile_fallback_rate={100.0 * int(counts.get('profiles_fallback', 0)) / requested:.1f}%")

    return ", ".join(parts)


def diagnose_gpt_threshold_usage(
    counts: Mapping[str, int],
    error_counts: Mapping[str, int],
    max_retries: int,
) -> List[str]:
    """Return concise diagnostic notes for unexpectedly low GPT threshold utilization."""
    notes: List[str] = []
    attempted = int(counts.get("batches_attempted", 0))
    failed = int(counts.get("batches_failed", 0))
    skipped_no_key = int(counts.get("batches_skipped_no_key", 0))
    requested = int(counts.get("profiles_requested", 0))
    fallback = int(counts.get("profiles_fallback", 0))

    if skipped_no_key > 0 and attempted == 0:
        notes.append("no OpenAI API key was available, so every profile used deterministic fallback thresholds")
    if attempted > 0 and failed / max(1, attempted) >= 0.50:
        notes.append(
            "most GPT threshold batches failed before returning usable rows; check network/API connectivity, proxy stability, and rate limits"
        )
    if requested > 0 and fallback / max(1, requested) >= 0.50:
        notes.append("fallback thresholds dominate this bootstrap, so GPT utilization is low for this run")
    if attempted > 0 and failed > 0 and int(max_retries) <= 0:
        notes.append("gpt_threshold_max_retries=0 disables retry recovery for transient RemoteDisconnected/connection-refused errors")
    if error_counts:
        top_reason = max(error_counts.items(), key=lambda kv: int(kv[1]))[0]
        notes.append(f"most frequent GPT fallback reason: {top_reason}")

    return notes
