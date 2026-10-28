# CRISP-Ride Revision 14 change report

Revision 14 was forked from `rideshare_v26r_revision12` with `runs`, Python bytecode, and pytest caches excluded. Revision 12 and all earlier revision trees were left unchanged.

## Diff scope

`diff -rq --exclude runs --exclude __pycache__ --exclude .pytest_cache rideshare_v26r_revision12 rideshare_v26r_revision14` reports only these differences:

- Modified: `Core.py`, `PPOAgent.py`, `platform_mdp.py`.
- Added: `tests/test_revision14.py`.
- Added: `scripts/run_revision14_verify.py`, `scripts/r14_identity_check.py`, `scripts/r14_probe_report.py`, `scripts/best_static_tariff_search.py`.
- Added: this report.

No market, demand, supply, shock, rival, scenario, reward, curriculum, learning/exploration schedule, warm-start, or evaluation-harness source was edited. In particular, `pricing_models.py`, `joint_demand.py`, `choice_models.py`, `MarketInteraction.py`, `market_dynamics.py`, `driver_supply.py`, `market_shocks.py`, `scripts/v26r_case_eval.py`, and `scripts/v26r_cases.py` are byte-identical to Revision 12.

## Per-function changes

### `platform_mdp.py`

- `ObservationConfig`: added the four opt-in Revision 14 settings for momentum and opponent observations.
- Module constants and `_window_delta`, `_window_trend`: added the registered signal ordering, dimensions, scales, one-day delta, and multi-day trend transforms.
- `PlatformObservationModel.__init__`: extends observation width only when the corresponding flags are enabled.
- `regime_feature_groups`: appends the opponent slot to the recurrent opponent group and routes other opt-in fields to the immediate group.
- `reset`, `snapshot`, `restore`: initialize and round-trip the new telemetry, gap, rival-index, and rival-move histories. Restore remains backward compatible with Revision 12 snapshots.
- `ingest`: appends delayed public telemetry only when a Revision 14 observation feature is active.
- `_append_momentum_history`, `_momentum_slot`: construct the 15 registered momentum features.
- `record_rival_index`, `_opponent_slot`: implement two-record move confirmation and the 11 registered public opponent features.
- `build_observation`: appends momentum and opponent slots after the Revision 12 frame.

The requested audit found no code in `pricing_models.py` or `PPOAgent.py` that assumes the opponent feature group contains exactly four indices.

### `Core.py`

- `Core.__init__`: added and validated the nine registered NMCC/observation parameters; preserves Revision 12 defaults when all new flags are off; validates the opponent-information contract; supplies NMCC baseline and scaling modes to the actor only when branch credit is active.
- `v26r_settings`: records the eight reproducibility settings required by the specification.
- `_publish_firm2_leader_quotes`: retains the exact Revision 12 public rival-index expression and optionally records it for Firm 1's opponent slot.
- `_counterfactual_day_values`: always includes hold and the sampled action, fills with least-evaluated actions, truncates at the episode boundary, uses day-level discounting, centers on hold by default, restores the simulator after every branch, and emits the registered probe diagnostics.
- Firm 1 training decision site: restricts NMCC to configured stages, applies every-k cost control, and passes the remaining episode horizon.
- PPO/daily diagnostics: records NMCC valid fraction, advantage RMS, GAE sign agreement, call count, elapsed time, and the most recent branch summary.
- `_baseline_firm1_action`, `_post_firm1_coefficient_targets`: add the evaluation-only `static_tariff:@path` comparator.
- `_trace_row`: adds both firms' 4-mile/15-minute reference-fare indices and Firm 1's per-coefficient anchor ratios.
- Argument parser, artifact reload, alias mapping, and `Core(...)` construction: add the registered Revision 14 flags, restore observation contracts from artifacts, and map NMCC aliases onto the existing counterfactual machinery.

### `PPOAgent.py`

- `_counterfactual_actor_advantages`: supports the registered supplied-hold baseline and sign-preserving RMS scaling while retaining Revision 12's mean/z-score defaults.
- `update`: reads the opt-in NMCC modes, blends only valid branch rows, and returns valid-fraction, advantage-RMS, and GAE-sign-agreement diagnostics.

### Verification and evaluation scripts

- `r14_identity_check.py`: deterministic 20-day Revision 12/14 identity check over rewards, actions, top-action labels, shocks, and PPO/KL rows.
- `r14_probe_report.py`: validates hold/sample invariants, classifies candidates relative to the current coefficient ratio, reports class-by-regime branch advantages and the preregistered signal statistic.
- `best_static_tariff_search.py`: performs two validation-only coordinate-ascent sweeps in the specified coefficient order and writes both the best differentiated fixed tariff and frozen-RL tariff.
- `run_revision14_verify.py`: implements the six gated phases, detached-run records, 600-day smoke gates, four 4,000-day ablations, held-out case sweeps, and P1–P6 analysis.

## Deviations

There are no intentional model or experimental-design deviations from the attached specification. The full-run harness reports the required follow-up seeds 251013 and 251014 when the public seed-251012 policy clears P1–P3; it does not silently claim replication from a single seed.

## Test record

- Python compilation (`Core.py`, `PPOAgent.py`, `platform_mdp.py`, and all four new scripts): passed.
- `pytest -q tests/test_revision14.py`: 8 passed.
- Full Revision 12/Revision 14 suite comparison and the staged verification phases: recorded under the run directory named by `runs/latest_revision14_verify_path.txt`. The phase `status.json` files and this section are updated from the completed evidence before handoff.
