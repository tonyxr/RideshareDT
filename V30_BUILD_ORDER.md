# Model v30 build order and frozen contracts

## Scope

V30 is an opt-in extension of the current v29 configuration. It is not a new
simulator or a second learner. `--v30` composes the existing v29/v28/v27/v26
market with the original categorical PPO stack, then adds only:

1. four closed-loop observation fields;
2. a profit-only dominance objective with one terminal profit-lead reward; and
3. external data, calibration, evaluation, shadow-serving, and cycle contracts
   needed to operate the simulator as a continuously updated digital twin.

The simulator, market shocks, rider choice, profile-specific rider memory,
driver supply, five independent tariff actions, and PPO implementation remain
the current inherited implementations. No conservative-policy-improvement
controller, fallback actor, action projection, or economic upper bound is part
of v30 learning. Firm 1's five coefficient multipliers retain the v29 lower
floor of `1.0` relative to the calibrated city anchors.

## Build and acceptance order

Each layer must pass before the next layer is changed.

### 0. Frozen inheritance

- `--v30` implies `--v29`, which implies v28/v27/v26.
- Cognitive choice and rider memory are mandatory.
- The action set is `hold` plus an up/down action for each of the five tariff
  coefficients. One action changes one coefficient only.
- Coefficients use the existing additive step semantics, an anchor-relative
  floor of `1.0`, and no economic upper bound. Only non-finite numerical
  operations are rejected atomically.
- PPO receives the exact discounted semi-MDP reward sum over a held action.

Acceptance: v25-v29 regression suites and action-floor tests remain green.

### 1. Data identity and event ledger

`digital_twin/ledger.py` stores decisions, delayed outcomes, and observations
idempotently in a transactional ledger. `digital_twin/identity_store.py` keeps
immutable pseudonymous rider/driver traits separate from versioned experience
state. Direct PII keys are rejected.

Acceptance: replay is idempotent, conflicting duplicates fail, late outcomes
join to decisions, event timestamps are UTC-aware, tariff payloads are
non-negative, and physical duration/probability domains are enforced.

### 2. Simulator/production observation parity

`digital_twin/observation.py` uses `PlatformObservationModel.feature_names()`
as the only base schema source. It appends exactly four fields:

- log forecast/current demand ratio;
- log forecast/current supply ratio;
- log state uncertainty; and
- log data age in decision intervals.

The production mapping and simulator vector produce the same ordered vector
and schema hash. V30 requires the concise v29 foresight frame.

Acceptance: exact vector parity, finite transforms, stable names/hash, and
artifact load rejection on a schema mismatch.

### 3. Online state and parameter assimilation

`digital_twin/state_estimator.py` provides a small online filter for demand,
supply, public rival quote index, and shock probability. Internal forecasts
are projected to their natural domains; invalid external measurements still
fail. `digital_twin/calibration.py` provides recursive calibration and residual
drift diagnostics. `digital_twin/twin_ensemble.py` maintains probabilities
over versioned plausible structural parameterizations.

Acceptance: missing-data propagation, uncertainty growth, trend forecasting,
snapshot/restore parity, convergence, drift detection, ensemble posterior
updates, and deterministic restore.

### 4. Counterfactual evaluation

`digital_twin/fixed_comparator.py` searches all five fixed coefficient
multipliers independently, with every multiplier constrained to `>=1.0` and no
economic upper cap. `digital_twin/off_policy.py` reports unclipped IPS, SNIPS,
doubly robust estimates, effective sample size, overlap warnings, and
cluster-bootstrap intervals.

This five-dimensional comparator is the scientific fixed-policy comparator.
The older uniform static grid in `Core.py` remains a cheaper lower-bound
diagnostic and must not be described as the best fixed tariff.

Acceptance: known optima, deterministic common-seed replay, finite-tail
failure handling, propensity validation, low-overlap warnings, and clustered
uncertainty.

### 5. Shadow operation and controlled experiments

`digital_twin/policy_service.py` builds the exact registered production
observation, records the full action distribution, and writes a recommendation
to the ledger without changing a live tariff. `digital_twin/experiments.py`
provides stable pseudonymous randomized assignment with an exact logged
propensity.

Acceptance: schema mismatch fails, invalid probabilities fail, shadow output
is deterministic, no deployment flag is set, and assignment is stable.

### 6. Concise MDP reward and terminal foresight

The corrected dense v30 reward directly matches monetary profit dominance:

```text
(own contribution profit - rival contribution profit) / profit-advantage scale
```

No share, quote-gap, movement, service, or guardrail term sends PPO credit.
At a true training-episode boundary the tracker adds one unclipped terminal
reward equal to the final-quarter mean contribution-profit advantage times
the episode length in market steps and `terminal_profit_dominance_weight`.
Unlike the previous unscaled episode mean, this emphasizes late outcomes on
the scale of the dense return. That reward enters the same discounted
semi-MDP transition and propagates through PPO returns. Validation applies the
same terminal term once.

The two public clock features distinguish remaining time and horizon length.
One critic-only tail accumulator resolves the path-dependent terminal reward
without exposing rival profit to the actor. Return-RMS-normalized critic
regression preserves raw value predictions when statistics change; GAE stays
in economic units. The simulator, shocks, independent coefficient actions and
environment-then-opponent curriculum are unchanged. Absolute own profit is
reported separately; relative-profit optimization is not an absolute-profit
guarantee. Old checkpoints require their original observation/reward contract.

Economic dominance certification is separate from learning: full and tail own
profit must be positive, full and tail profit advantage non-negative, and the
rolling profit-dominance rate must meet its declared floor. It never changes
the actor during training.

Acceptance: reward-component isolation, monotonic profit response, one-time
terminal accounting/reset, unclipped extremes, exact PPO diagnostic parity,
and validation/checkpoint objective parity.

### 7. Continuous optimization cycle

`digital_twin/orchestrator.py` commits these stages atomically:

```text
ingest -> assimilate_state -> calibrate -> update_twin
       -> train -> evaluate -> shadow
```

`scripts/run_v30_cycle.py` runs provider-specific commands from a content-
hashed configuration and resumes at the first unfinished stage. A completed
shadow cycle is not deployment. Deployment requires a separate explicit
authorization with an authorizer, change ticket, and policy artifact hash.
This is an operational change-control boundary, not a learning guardrail.

Acceptance: stage-prefix integrity, crash resume, immutable configuration,
artifact hashing, explicit deployment, and a composed data-to-shadow cycle
covering every v30 component.

### 8. Training ladder

1. Unit and component tests.
2. v25-v29 regression suites.
3. Tiny v30 wiring run.
4. Diagnostic v30 smoke with train/validation only.
5. Five-dimensional fixed comparator and benchmark evaluation.
6. Powered environment-first training.
7. Opponent training.
8. One frozen final-test evaluation.
9. Shadow cycle; any live deployment remains an external authorization.

`scripts/run_v30_smoke_chicago.sh` is deliberately non-promotional and seals
the final test partition. `scripts/run_v30_full_chicago.sh` is a detached full
job and must not be started merely because component tests pass.

## Current operational boundaries

- Periodic collaborating-firm feeds must be mapped into the registered event
  schemas by provider-specific ingestion commands.
- The event ledger reference implementation is SQLite; a production event bus
  may replace it while preserving the same canonical contracts.
- The fixed five-dimensional comparator is a separate evaluation component;
  Core's inherited uniform multiplier grid is not an equivalent substitute.
- Shadow recommendations never mutate production tariffs.
- V30 does not claim exact mid-market simulator-state resume. The optimization
  cycle resumes at committed stage boundaries.

## Reward/critic/foresight correction tests — 2026-10-12

Implemented sequentially, without restarting or modifying the already-loaded
seed-300127 evaluation, its launcher, reports, or scheduled-check settings:

1. Reward alignment: 69 tests passed, including the observed day-3000 versus
   day-6000 reversal and the pilot's late profit-lead loss. Dense reward now
   equals the reported profit lead rather than rewarding absolute profit growth
   despite a worse lead. Other named reward objectives are unchanged.
2. Critic: 69 tests passed. Across three seeds and return scales 1, 100, and
   10,000, held-out known-return explained variance exceeded 0.95 and relative
   RMSE stayed below 0.04. Tests cover raw-output preservation, unchanged actor,
   saved statistics, actual PPO target units and update counts. Joint learning
   from public state plus terminal context also exceeds 0.95 held-out EV.
3. Foresight: tail reward accounting, clock observability, critic-only history,
   snapshot restoration, and delayed-payoff PPO learning are tested. All three
   delayed-payoff seeds put over 90% probability on the long-term winning
   opening action despite its worse immediate payoff. Validation now includes
   the final held action and terminal credit in its critic TD diagnostic.

Combined regression: **297 passed**. Evidence and JUnit results are in
`artifacts/v30/reward_critic_foresight_fixes_20261012/`.

A native-Terminal Chicago smoke run used existing calibration, trip bank and
cached profiles: 24 days, four blocks/day, 40 requests/block, environment then
aggressive adaptive rival, seed 301012. Exit status was zero; terminal boundaries
were days 5, 10, 15, 20 and 24, with five PPO updates. Final evaluation used only
16 steps of the validation partition, not the final-test partition. Own-minus-
rival profit per request was +0.120652 overall and +0.291010 in the last quarter.
These are integration results, not evidence of sustained benchmark dominance.
The five in-market critic EV values were +0.2969, +0.1405, -0.0194, -0.3022,
-0.0043; successful synthetic value fitting does not certify market convergence.

The first smoke exposed a legacy CLI bug that redirected absolute report
prefixes into the repository root, refreshing root-level `chicago_*` convenience
reports. Their previous untracked contents were not recoverably captured. The
versioned ongoing-run directory was unaffected. Absolute destinations are now
honored, regression-tested, and verified by a second isolated smoke run; the
first smoke's diagnostics were preserved under `first_smoke/`.

No action cap, new profit floor, CPI controller, fallback policy, or performance
gate was added. No full-budget replacement run or scheduled monitor was started.
The simulator, shock process, rider memory and coefficient action semantics
remain unchanged. Fresh training is required for the new observation and value
contract; old-run comparisons must use their archived source, not this version.
