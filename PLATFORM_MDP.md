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

The scalar reward is always positive and contains only normalized business
utility: contribution profit, revenue, completed demand, and service quality.
There is no hold penalty, intervention bonus, price-gap reward, or oscillation
penalty in the scalar reward.

The active weights are independently configurable and normalized to sum to
one:

```text
--positive_reward_profit_weight
--positive_reward_revenue_weight
--positive_reward_completed_demand_weight
--positive_reward_service_weight
```

All must be finite and non-negative, and at least one must be positive. Profit
and revenue are measured per incoming request, not per completed ride, so
raising prices while losing demand does not automatically improve either
component. Completed demand and service provide direct supporting signals for
sustainable volume and execution quality.

Price-gap overpricing/underpricing, four distance-segment gaps, fulfillment,
wait, margin, and oscillation are ten separate soft costs. PPO learns cost
critics and uses bounded primal-dual multipliers. Related gap multipliers are
normalized as one family so segment detail cannot overpower business utility.

## Staged training

`--training_curriculum staged` runs:

1. `foundation`: frozen opponent and broad exploration;
2. `robustness`: slower-moving opponent and increased constraint pressure;
3. `competition`: normal opponent cadence;
4. `consolidation`: lower exploration/learning rate and a longer action hold.

Hold remains a normal action in every stage. The action cadence and oscillation
cost teach intervention maturity without making hold intrinsically preferable.

## Train and save

```bash
python Core.py \
  --run_experiment \
  --firm1_mode RL \
  --firm2_mode queue_service_threshold \
  --training_curriculum staged \
  --positive_reward_profit_weight 0.38 \
  --positive_reward_revenue_weight 0.22 \
  --positive_reward_completed_demand_weight 0.20 \
  --positive_reward_service_weight 0.20 \
  --trained_model_out artifacts/platform_policy.pt \
  --report_prefix artifacts/platform_train
```

Every training run writes a timestamped immutable policy under
`artifacts/trained_models/` and appends its identity to `manifest.jsonl`.
`--trained_model_out` is an optional mutable convenience alias; overwriting the
alias never removes older archived models. The artifact includes policy and
optimizer weights, normalizers, action mapping, frame-stack shape, MDP
configuration, and validation metadata. It deliberately excludes the terminal
tariff, opponent, observation queues, and dual state so evaluation starts from
a fresh market rather than a leaked training world.

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
policy directly. The default action mode is seeded `top2_margin`: it samples
between the two leaders only when their probability gap is below 0.05, avoiding
brittle argmax tie-breaking while remaining reproducible. Use
`--eval_policy_mode argmax` for the deterministic baseline. The default
evaluation guardrail is `log_only`, so it cannot silently replace policy
actions.

## Validate prices against NYC ride rows

```bash
python Core.py \
  --eval_only \
  --dataset_only \
  --trained_model_in platform_train \
  --firm2_mode static \
  --compare_with_dataset \
  --comparison_policy_mode argmax \
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
not confined to the start of a chronologically sorted file.
