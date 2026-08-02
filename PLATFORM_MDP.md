# Platform pricing MDP

The market simulator remains the source of rider, trip, weather, driver, and
choice outcomes. `platform_mdp.py` is the information and control boundary
between that simulator and a pricing firm.

## Information available to the policy

Firm1 receives:

- its active fare coefficients;
- delayed/noisy own operational KPIs (requests, completed requests, revenue,
  contribution profit, fulfillment, acceptance, wait, driver utilization);
- the observed mix of trip distance, duration, airport, and long-trip demand;
- delayed/noisy public competitor quote probes for 0–2, 2–5, 5–10, and 10+
  mile representative trips, including quote age and uncertainty;
- its own recent intervention history.

The policy does **not** receive rider threshold parameters, exact competitor
coefficients, competitor profit, competitor controller state, or exact current
segment gaps.

## Objective and constraints

The policy has one selected objective and one corresponding reward mechanism.
Profit maximization and market-share competitiveness are not additive terms in
the same named reward. They retain identical state features, action features,
policy network, PPO implementation, training process, diagnostic columns, and
graph layout.

For incoming request count `N`, completed rides `C_i`, paid fare `p_ij`, and
variable contribution cost `c_ij`, platform profit is:

`pi_i = sum(j in C_i, p_ij - c_ij) / N`.

Relative profit is exactly `delta_pi = pi_1 - pi_2`, in dollars per incoming
request. It is not market share, revenue, margin, or profit per completed ride.
The v6-based profit objective is:

`R_profit = w_abs * asinh(pi_1 / s_abs) + w_rel * q(pi_1) * tanh(delta_pi / s_rel)`

where `q(pi_1) = 1 - exp(-max(pi_1, 0) / s_quality)`. The quality gate means
beating the rival while Firm1 is unprofitable cannot earn a large relative
profit reward.

For completed market shares `s_1` and `s_2`, the fully separate competitiveness
reward is:

`R_share = w_level * s_1 + (1 - w_level) * 0.5 * (1 + tanh(((s_1 - s_2) - target_share_gap) / share_gap_scale))`.

The first term discourages winning a tiny residual market; the second gives a
dense learning signal for reaching and exceeding Firm 1's required completed-
share lead. The named competitiveness preset uses a ten-percentage-point target
gap. Quote-gap MAE, RMSE, p90 error, and in-tolerance rate remain visible
diagnostics but do not enter this reward.

Competitiveness reward contains no profit component. Positive own contribution
profit is enforced as a checkpoint and deployment-certification requirement,
so a loss-making market-share win cannot be deployed but profit cannot
overpower the share objective during learning.

The active constrained-MDP costs remain fulfillment, wait, and margin. Quote-
gap costs are retained as diagnostics and are not hidden reward terms.

### Objective profiles

Profit maximization—the default, restoring the v6 `1.0/0.10` reward:

```bash
--policy_objective profit_maximization
```

Market-share competitiveness (sustain Firm 1's completed-share lead; enforce
positive Firm 1 profit in certification):

```bash
--policy_objective competitiveness
```

Balanced:

```bash
--policy_objective balanced
```

Balanced mode explicitly combines normalized profit and market-share utilities. Use
`--policy_objective custom` only for a backward-compatible expert-weighted
combination. Saved policies restore the named objective, mechanism, and all
underlying values.

Checkpoint ranking is objective-specific: profit policies rank realized profit
reward, while competitiveness policies rank completed-share advantage. An
ineligible snapshot is retained only as diagnostic metadata and can
never replace or be restored as the deployable best checkpoint. Certification
requires both an eligible validation snapshot and a passing final durability
audit. Profit runs require a consistent contribution-profit lead of at least
`$0.25` per incoming request; competitiveness runs require a consistent
completed-share lead at or above the configured target and positive own
contribution profit; balanced runs require positive share and profit leads.

## Training

The default `--training_curriculum staged` runs:

1. `foundation`: frozen opponent and broad exploration;
2. `robustness`: slower-moving opponent and increased constraint pressure;
3. `competition`: normal opponent cadence;
4. `consolidation`: low exploration and a long uninterrupted rehearsal against
   the exact evaluation opponent, so checkpoint selection observes its
   late-stage response.

Exploration, entropy, and learning rate use smooth transitions at stage
boundaries. The bounded rational gap utility and long uninterrupted
consolidation stage reduce the reset-driven spikes seen in direct training.
Reward graphs use one uniform trailing expected-reward line and 95% standard-
error band, with raw market outcomes shown as faint unconnected points; PPO
still optimizes the exact unsmoothed objective reward. Reward convergence and
economic checkpoint eligibility are reported separately.
`--training_curriculum direct` remains available only as an ablation
that keeps the benchmark opponent active in repeated long episodes.

Hold remains a normal action in every stage. The default cadence checks for an
update every 10 training days and requires at least 128 pricing decisions,
accumulating across boundaries as needed. This produces many more optimizer
updates than the earlier 256-decision/20-day setup.

Action features describe projected own fares relative to the public city
anchor; they do not reveal which action approaches a hand-authored competitor
price gap. State/action specialization is computed over three economically
meaningful fare-impact groups (lower, neutral, higher), using the same
broad-market exposure weights and five-cent threshold as the deployment audit.
Because action-feature impacts are stored as dollars divided by 20, the
optimizer converts that threshold to matching normalized units. This prevents
nominal action-ID changes or low-exposure airport-fee chatter from masquerading
as a dynamic policy. Its default weight is `0.10`, and the specialization
distribution is sharpened only for this regularizer so low-ranked probability
tails cannot earn credit while deterministic argmax stays constant. During
optimization it is weighted by positive PPO advantage: changing economic
direction in an unprofitable state earns no specialization bonus.

## Train and save

Profit-maximizing policy:

```bash
python Core.py \
  --run_experiment \
  --firm1_mode RL \
  --firm2_mode adaptive_best_response_aggressive \
  --training_curriculum staged \
  --policy_objective profit_maximization \
  --state_action_mi_weight 0.10 \
  --checkpoint_validation_horizon 1000 \
  --checkpoint_validation_interval_days 600 \
  --checkpoint_validation_customers 100 \
  --eval_policy_mode argmax \
  --require_competitive_backtest \
  --trained_model_out artifacts/platform_policy.pt \
  --report_prefix artifacts/platform_train
```

Market-share-competitiveness policy, using the same model and training stages:

```bash
python Core.py \
  --run_experiment \
  --firm1_mode RL \
  --firm2_mode adaptive_best_response_aggressive \
  --choice_mode cognitive \
  --threshold_profile_source cached \
  --threshold_cache_path artifacts/threshold_profiles_30k.jsonl \
  --policy_objective competitiveness \
  --training_curriculum staged \
  --train_timesteps 3000 \
  --train_steps_per_day 4 \
  --train_customers 200 \
  --ppo_update_interval_days 10 \
  --ppo_min_rollout_transitions 128 \
  --ppo_batch_size 128 \
  --checkpoint_validation_horizon 1000 \
  --checkpoint_validation_interval_days 600 \
  --checkpoint_validation_customers 100 \
  --eval_timesteps 1000 \
  --eval_customers 200 \
  --eval_policy_mode argmax \
  --eval_guardrail_mode off \
  --deterministic_torch \
  --experiment_seed 4500 \
  --require_competitive_backtest \
  --trained_model_out artifacts/price_competitiveness_v6plus.pt \
  --trained_model_id price-competitiveness-v6plus-seed4500 \
  --profiles_out artifacts/price_competitiveness_v6plus_profiles.csv \
  --report_prefix artifacts/price_competitiveness_v6plus
```

Every candidate is first evaluated and subjected to the competitive durability
audit. Only a passing candidate is written as a timestamped immutable policy
under `artifacts/trained_models/` and appended to `manifest.jsonl`; a failed
candidate is never certified or registered. If an explicit
`--trained_model_out` path is supplied, a failed candidate may be written there
for debugging with `certification_status=uncertified`.
`--trained_model_out` is an optional mutable convenience alias; overwriting the
alias never removes older archived models. The artifact includes policy and
optimizer weights, normalizers, action mapping, frame-stack shape, MDP
configuration, and validation metadata. It deliberately excludes the terminal
tariff, opponent, observation queues, and dual state so evaluation starts from
a fresh market rather than a leaked training world.

Checkpoint validation uses the real opponent update cadence and nine held-out
rollouts: static, five independently seeded aggressive best responses, PI
price-gap, queue/service threshold, and region supply/demand. Profit checkpoints
must clear the profit-gap target in every rollout; competitiveness checkpoints
must clear the completed-share-gap target in every rollout and remain
profitable; balanced checkpoints must maintain positive share and profit leads.
Action diversity and tariff movement remain diagnostics, not eligibility
requirements: a stable dominant policy is allowed to converge. The final
1,000-step audit applies the same objective-specific business criteria plus
reward retention.

## Evaluate the same policy against benchmarks

```bash
for opponent in pi_price_gap mpc_grid region_supply_demand surge_driver_incentive queue_service_threshold; do
  python Core.py \
    --eval_only \
    --trained_model_in platform_train \
    --firm2_mode "$opponent" \
    --report_prefix "artifacts/eval_${opponent}"
done
```

`--trained_model_in` accepts an explicit `.pt` path, a registry model id/archive
filename, or `latest`. `--list_trained_models` prints all available archive
records. Evaluation restores the artifact's exact MDP configuration, leaves the
selected opponent untouched, performs no optimizer update, and uses the learned
policy directly. The default action mode is deterministic `argmax`, making
deployment independent of a sampling seed and exposing whether the network
actually learned a state-dependent response. Stochastic modes are available
only for explicit sensitivity analysis; the legacy `top2_margin` selector has
been removed from both the command-line workflow and the policy runtime. The default evaluation guardrail is
`log_only`, so it cannot silently replace policy actions.

## Validate prices against NYC ride rows

```bash
python Core.py \
  --eval_only \
  --dataset_only \
  --trained_model_in platform_train \
  --firm2_mode static \
  --compare_with_dataset \
  --comparison_policy_mode argmax \
  --comparison_dataset_seed 104729 \
  --comparison_duration_mode actual_if_available \
  --comparison_limit 5000 \
  --comparison_out artifacts/platform_train_nyc_prices.csv
```

For each valid ride, the validator starts from the NYC anchor tariff, injects
the observed hour/day, distance, duration, airport, weather, and service
context, executes one independent learned policy action, and computes a fare.
It records the selected action, action confidence, magnitude, resulting tariff
coefficients, predicted price, and paid passenger fare. Rows do not change the
next row's state. Reports include MAE/RMSE/MAPE/bias, correlation, $2/$5
coverage, invalid-row counts, and improvement relative to the unmodified NYC
anchor tariff. Parquet row groups are read round-robin so a bounded sample is
not confined to the start of a chronologically sorted file. Set a different
`--comparison_dataset_seed` to randomize row-group order, bounded batch offsets,
and row order for an independent sample. Use
`--comparison_duration_mode predicted_only` as a leakage stress test that
prevents observed trip duration from entering the price calculation.

For a repeatable multi-seed robustness report:

```bash
python run_dataset_robustness.py --rows 3000
```

This runs four independent policy/dataset seed pairs under both observed-
duration and predicted-duration-only pricing, then writes scatter facets,
cross-seed metrics, error distributions, row-level CSVs, and a JSON summary.
