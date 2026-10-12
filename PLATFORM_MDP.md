# RideResponse economic-objective PPO and v30 closed-loop digital twin

## v30 closed-loop extension (2026-10)

V30 is an opt-in composition over v29, not a replacement simulator or learner.
It preserves the five independent tariff actions, cognitive rider choice and
identity-specific memory, the v27 market/shock engine, and the original PPO
stack. It appends four online forecast/freshness fields and two public episode
clock fields to the concise public observation (31 fields with shocks). The
dense reward is `(own contribution profit - rival contribution profit) /
profit_advantage_scale`. Absolute profits remain separate reported outcomes;
maximizing relative profit does not guarantee positive absolute profit.

At a true boundary, the terminal reward is `weight * episode_market_steps *
mean(final-quarter profit lead) / profit_advantage_scale`. This explicitly
values maintaining late leadership, on the scale of the dense episode sum.
PPO discounts it at the boundary in the held-action return; it is not clipped.
The actor sees remaining-episode fraction and log horizon length. The critic
also receives one training-only accumulated tail-lead scalar through its
existing context encoder, not rival identity, exact tariffs, or future events.

V30 uses output-preserving running return-RMS normalization for the value head
and value loss. Values, advantages, GAE and diagnostics remain in raw economic
units; actor probabilities and rewards are not normalized or capped. The RMS
statistics are saved with the network. Older objectives keep their original
value architecture. The changed observation/reward/value contract requires a
fresh v30 model; old artifacts must be evaluated with their saved source.

The external closed loop is implemented by the `digital_twin/` package:
transactional events and pseudonymous identity state, online state estimation,
recursive calibration and drift monitoring, a versioned twin ensemble, a full
five-dimensional fixed comparator, off-policy estimators, shadow-only policy
serving, stable experiments, and a crash-safe periodic-cycle orchestrator.
The complete ordered contract and acceptance ladder are in
`V30_BUILD_ORDER.md`.

V30 learning has no CPI controller, fallback actor, performance action mask,
or economic upper tariff cap. The v29 anchor-relative coefficient floor remains
`1.0`; numerical overflow is rejected rather than clipped. Operational shadow
and deployment authorization occur after learning and do not modify PPO.

V30 evaluates the categorical actor by sampling its raw distribution, exactly
as during training (`sample_raw`, seeded). Argmax remains an explicit diagnostic
ablation: repeating the most probable *increment* is not equivalent to holding
a target tariff. V30 PPO uses actual discounted episodic rewards and the terminal
bonus without subtracting the current rollout's reward mean. Legacy continuing
tasks retain their differential-return behavior. The return contract is saved;
old differential-value checkpoints cannot silently resume under a changed
critic target. Online estimator timestamps advance by
`24 / steps_per_day` hours per market block, matching forecast horizons.

## v27 market engine and stochastic shocks (2026-10)

`--v27` (implies `--v26`) keeps the learner and the v26 evaluation contract and
replaces the market step.  Without `--v27` the code reproduces v26 exactly
(the Boston v26 smoke is bit-identical before and after the v27 change).  The
settings are stored in the run config and model artifact under `v26.v27`.

### Market engine (`market_engine.py`)

| Component | v26 | v27 |
|---|---|---|
| Rider choice | One sampled choice per rider (cognitive utility) | Expected flows: each rider contributes P(no ride), P(Firm 1), P(Firm 2) from the *same* utility, vectorized (unit-tested to 1e-8 against `CognitiveChoiceModel.choose`) |
| Demand model | Cognitive | Cognitive (default) or `--demand_model nested_logit`: mixed nested logit (outside option vs a rideshare nest), price in units of the city reference fare, a kink beyond the rider's CrowdLLM threshold, income/household heterogeneity |
| Driver acceptance | Deterministic cutoff (accept if probability >= 0.55) | Expected acceptance probability (same earnings ratio) |
| Capacity | One trip per online driver per step | Driver time: each online driver supplies the block's minutes; a trip uses pickup + trip minutes.  Pickup = 3.5 / sqrt(1 - utilization) minutes (capped at 16), so pickup times rise as idle supply falls; when demand cannot be carried, requests are rationed at the sustainable maximum (the lower, stable equilibrium is used when two exist) |
| Riders' ETA | Noisy draw per rider | Platform ETA: last step's pickup, then one correction with this step's solved pickup |
| Fleet calibration | 95% anchor fulfillment, one-trip capacity | Drivers per base request so that both firms at the anchor tariff run at `--target_utilization` (0.60) over a calendar week (request-weighted); the anchor-market earnings per online hour become the participation reference, so the anchor market is a steady state |
| Multi-homing | None | `--driver_multihoming` (0.5) of drivers run both apps and are shared capacity *within* each step: their time goes to each platform in proportion to the driver minutes its requests need, tilted toward the app paying more per engaged minute (`--allocation_sensitivity` 3); an outage also cuts the platform's reach.  The rest single-home (half per platform) |
| Driver participation | Online drivers react to last step's engaged-hour earnings | Each group x (1 + 0.25 tanh((E - E_ref)/(0.75 E_ref))), with E an exponentially smoothed (rate 1/12 per step, about two days) earnings per *online* hour: market-wide for multi-homers, the own platform's for single-homers |
| Supply cycle | Tracks 50% of the hourly demand cycle | Tracks 50% of the hourly *and* the weekly cycle |
| Noise | Choice and dispatch Monte Carlo | None beyond real randomness (request volume, rider mix, trips, weather, shocks) |

Rider-side metrics keep v26 units (sampled riders, money per base request);
driver time and earnings use realized requests.  `FirmMetrics` counts are
expected values (floats).

### Shocks (`market_shocks.py`, `--shocks calibrated|stress`)

A dated calendar (the weekday follows the date) drives a `ShockProcess` that
lives inside `MarketInteraction`, so counterfactual snapshots copy it and
rebuilt validation/benchmark markets re-create it from their seeds (common
shock paths across policies).  Shock draws use their own stream.

| Shock | Public? | Channels |
|---|---|---|
| Holiday travel wave (Thanksgiving, Christmas-New Year, July 4, Memorial, Labor Day; 3-12 days) | yes (calendar) | airport trip odds x2.2 at the edges, commute volume x0.75, airport riders less price sensitive, fleet x0.9 |
| Winter storm (Dec-Mar hazards, 1-3 days, forecast one day ahead with misses/false alarms) | yes (warning, active) | weather = snow; transit suspended (p=0.5) or degraded; request volume x0.55 (replacing the weather demand factor of snow, which the ride samples cannot identify for storm days); half of the suppressed requests are deferred and released over two days after the storm; trips x1.35 longer; fleet x0.65; reservation fares x1.25 |
| Large event (evening blocks) | announced (p=0.7) | city-wide volume x1.15-1.45, lower price sensitivity |
| Transit disruption (1-4 days) | yes | outside option dearer and less convenient |
| Driver shortage (2-6 days) | latent | fleet x0.6-0.8, higher reservation wage |
| Platform outage (1-3 blocks, either firm) | latent | that firm's capacity x0.2-0.4 |
| Rival promotion (3-7 days) | latent (in-app coupons) | Firm 2 rider fares x0.8 |
| Demand drift | latent | AR(1) daily log volume (persistence 0.95, sd 0.03) |

`--shock_intensity` scales every effect (0 = neutral); `--shock_types`
restricts the set; `stress` raises arrival rates.  Training openings
oversample holiday and storm-season windows (50%); evaluation uses its own
natural-calendar shock path.  `--emergency_surge_cap c` caps fares at c x the
anchor fare during storms (off by default).

### Observation and critic

With shocks on, the public frame gains six features (`holiday_active`,
`holiday_proximity`, `storm_warning`, `storm_active`, `transit_availability`,
`event_announced`): 30 instead of 24.  `--no-public_shock_features` keeps the
slots but zeroes them (information ablation with the same network).  The
privileged critic additionally sees six latent channels (fleet multiplier,
each firm's capacity multiplier, the rival promotion, log volume, deferred
backlog).  Artifacts with a different width are rejected at load.

### Diagnostics

`scripts/v27_shock_diagnostics.py` evaluates, per shock type, static
multipliers, event switching, and an open-loop oracle schedule (fit on
training seeds, scored on test seeds, against a placebo window) on common
random numbers.  See LOG.md (2026-10-07) for the Boston results.

## v26 environment and evaluation contract (2026-10)

`--v26` keeps the v25 learner (public recurrent actor, centralized TD(0)
critic, Monte-Carlo actor credit, behavior-mixture PPO) and changes the market
it learns in and the way it is evaluated.  Every switch is stored in the run
config (`v26`) and in the model artifact (`v26_settings`).  Without `--v26`
the code reproduces v25.

### Market (city-calibrated, sequential)

| Component | v25 | v26 |
|---|---|---|
| Operational trips | `Exponential(4 mi)`, 18/25 mph, same in every city | Empirical trip bank (50k whole-file rows per city): distance, duration, airport, service, conditioned on hour and weekday/weekend |
| Tariff anchors | Hard-coded city presets | Fitted from observed fares/quotes (`city_market_calibration.fit_tariff_anchors`); presets via `--preset_tariff_anchors` |
| Tariff bounds/steps | NYC dollar bounds | Relative to each city's anchors |
| Calendar | i.i.d. hour each step, i.i.d. weekday/weather each day | Consecutive weekdays, Markov weather (stationary law = NOAA frequencies, persistence 0.70), ordered intra-day blocks (`--train_steps_per_day 6` = 4-hour blocks) |
| Request volume | Constant per step | Calibrated hour-of-day x weekday x weather intensity, Poisson noise; money metrics per *base* request so busy blocks weigh more |
| Driver supply | `max(0.25, customers/500)` floor; rush factor 1.18 | Linear in base customers (request/driver ratio invariant to run size); online drivers track 50% of the hourly demand cycle; earnings feedback unchanged |
| Firm symmetry | Firm 2 supply x0.96, pay share 0.70 vs 0.72 | Identical; pay share from observed NYC driver pay where available |
| Unserved riders | Lost | Re-request on the other platform with prob. 0.75 (0.25 if above reservation fare) |
| Rider price level | NYC-dollar reservation and outside-option constants | Scaled by the city price-level index (NYC = 1) and the city transit fare |
| Validation markets | Rebuilt from presets (calibration silently dropped) | Rebuilt with the active calibration (`_rebuild_market`) |
| Rider utility firm identity | Non-loyal riders lean to Firm 2 near ties (-0.03/-0.06); stressed riders lean to Firm 1 (+0.02) | Both constants 0 under symmetric firms (`CognitiveChoiceModel.make_symmetric`) |
| Driver pay basis | Share of the rider's own fare (a rider discount cuts pay and acceptance one-for-one) | Share of the city-anchor fare for the same trip/context (`--driver_pay_basis anchor`): platform-funded discounts, upfront-pay style |
| Fleet size | 260 drivers per 500 requests (hard-coded) | Bisection to 95% rider fulfillment at the anchor parity (`--capacity_target_fulfillment`), times `--fleet_scale` |
| `demand_intensity` observation | Product of tariff surcharge multipliers | Calibrated expected request volume of the block |
| Rider memory (opt-in) | None | `--rider_memory`: smoothed price/reliability reputations shift platform choice and the outside option (forward-looking demand channel) |
| Supply feedback | Always on | `--disable_supply_feedback` freezes the online-driver response (channel ablation) |
| Validation noise | Choice RNG neither restored nor reseeded per rollout | Choice, dispatch, and observer RNGs reseeded per rollout (CRN across policies) and restored afterwards |

### Learning protocol

- Decisions every market step (`--firm1_action_interval_steps 1`); the
  observation hour/day therefore matches the block being priced.
- Training opponents exclude the five benchmark controllers
  (`--exclude_benchmarks_from_training`); checkpoint selection validates on the
  training classes only.  Each training opening draws a perturbed
  parameterization of its controller (`--training_opponent_perturbation 0.15`).
- Each opening reseeds market, rider-choice, driver, and observer streams from
  (training seed, opening seed) (`--environment_reseed_per_episode`); runs use
  several training seeds (`scripts/run_v26_experiment.py`, default 4500-4502).
- Opponent-episode, stage, and horizon cuts are time-limit truncations that
  bootstrap from the value of the last observed state
  (`--truncation_bootstrap`); PPO segments baselines/weights on done OR
  truncated.  (v25 stored them as terminals with zero value.)

### Evaluation

- Five benchmark controllers are held out of training and selection.
- OOD: `--ood_perturbed_variants K` seeded perturbations of each benchmark.
- Firm-1-seat baselines on the RL rollout seeds (`--baseline_suite`):
  static grid (`static:m`, best-in-hindsight static oracle), `opening_hold`,
  and `myopic_greedy` (one-step common-random-number oracle with the true
  simulator).  RL minus myopic is the forward-looking value; RL minus best
  static is the value of adaptivity.  Written to `<prefix>_v26_baselines.json`.
- Observed-fare diagnostic (`--observed_fare_mode trajectory`) prices each
  observed ride with a tariff the frozen policy actually posted in a matching
  block/weekend/weather state during evaluation, and reports the best constant
  anchor multiplier (fit on even rows, scored on odd rows) as the baseline.

### CrowdLLM thresholds

`--gpt_threshold_mode llm_numeric` asks the model for the numeric rider-level
threshold (bounded to [$0.50, $5.00]; gender/marital status withheld; no
gradient stated in the prompt) and retains it; the v10 rubric is only a
labelled fallback.  `--threshold_ablation {constant_mean, shuffled, rubric}`
rewrites thresholds after the (threshold-independent) train/held-out rider
partition.  `scripts/external_threshold_benchmark.py` tests any threshold
population against Chicago TNP shared-trip authorization (1.49M pooling-era
trips, 18 PUMAs, ACS PUMS income).

### Generic city onboarding

A city needs a `config/city_datasets.json` entry with a `field_map` (column
names/units for price, distance, duration, timestamps, airport rule) plus NOAA
weather, ACS PUMS, and TIGER PUMA files; `scripts/prepare_city_data.py` then
produces the calibration and trip bank.  Seattle is catalogued with
`status: unavailable_no_public_trip_level_rides`.

---


Version 23 is a continuing-market POMDP with two separately trained objectives,
simple objective-specific rewards, coherent absolute whole-tariff actions, and
long-horizon recurrent PPO credit.

Version 23 remains the default compatibility mode. `--credit_assignment_v24`
enables the feature-gated credit-assignment stack described below without
changing the action set, economic reward formula, checkpoint objective, or
deployment actor. It does not enable constrained-MDP optimization.

`--credit_assignment_v25` includes the v24 features and corrects the causal
contracts exposed by the v24 audit: exact semi-MDP reward/exposure arithmetic,
the actual next decision state for bootstrap and potential shaping, a direct
centralized value function, a separately parameterized state-action critic,
pairwise observed counterfactual policy supervision, every-frame recurrent
ingestion, and response-timescale population batches. It likewise adds no
constrained-MDP mechanism and does not require the policy to change its tariff.

## State

The simulator can retain full ground truth, but Firm1 receives one 24-feature
public frame:

- 4 cyclical hour/day values (weather enters through demand intensity);
- 5 own tariff coefficients relative to the market anchors;
- 5 delayed own operating outcomes (completed share, profit/request,
  fulfillment, wait, utilization);
- 4 current demand descriptors;
- 4 public Firm2-minus-Firm1 quote gaps by trip segment; and
- 2 previous-action values (direction, time since intervention).

(Corrected 2026-10 to match `PlatformObservationModel.build_observation`.)

The default `--state_frame_stack 16` gives a 16-by-24 ordered sequence. Actor
and critic reshape the 384 input values back to that sequence. Each has its own
frame encoder, GRU, post-encoder, and output head:

```text
16 x 24 frames -> actor frame encoder -> actor GRU -> actor MLP -> 8 logits
16 x 24 frames -> critic frame encoder -> critic GRU -> critic MLP -> V(state)
```

Their parameters and Adam optimizers are disjoint. There is no shared trunk,
action-Q head, response predictor, opponent forecast, constraint/risk critic,
deployment reranker, learned magnitude, or direct actor-credit head.

In v24, the actor path is unchanged and still reads only public observations.
The critic additionally receives a fixed 64-field training-only centralized
summary: public state, exact bilateral tariffs, undelayed bilateral outcomes,
bilateral driver-supply state, and rival-controller execution state. Dedicated
modules fuse that summary into the critic latent; no privileged tensor is
reachable from actor parameters or deployment logits.

The v24 GRUs also retain hidden state across pricing decisions. PPO stores the
pre-action hidden state with each transition, so recomputed likelihoods use the
same recurrent context as rollout collection. Because adjacent frame stacks
overlap, a stateful step consumes only the newest frame after initialization.

V25 keeps the actor public-only, but maps the centralized state plus a
training-only one-hot identifier for the active opponent controller directly
to `V(s)` instead of fusing it with a second, partially observed recurrent
path. The identity is needed because the population contains distinct
transition kernels that can produce different TD targets from otherwise
identical market summaries. Its action-Q model has its own centralized-state encoder and an
explicit action-feature encoder, so `Q(s,a)` cannot collapse into a relabeled
state value. Every public market frame is consumed by the live actor and critic
recurrence, including frames between pricing decisions; PPO still stores the
exact pre-decision hidden state used to sample each action.

## Actions

The eight categorical actions are:

```text
0: hold the currently posted tariff
1..7: set every coefficient in the experiment's fixed managed scope to:
      0.70x, 0.80x, 0.90x, 1.00x, 1.10x, 1.20x, or 1.30x
```

`--manipulated_price_coefficients N` selects 1 through 5 coefficients in the
fixed nested order `base_fare`, `per_minute`, `per_mile`, `booking_fee`, and
`airport_fee`. Coefficients outside that prefix remain at their city-market
anchors. The selected count and exact keys are stored in the trained artifact;
validation and evaluation inherit them, and an explicitly conflicting runtime
count is rejected.

Each target changes every selected coefficient coherently. Repeating a target
leaves the posted tariff unchanged, so there is no incremental-action drift.
Every action is always available. The executor only clips a requested
coefficient to its physical bound; profit, share, price gaps, service, stages,
and previous actions never mask or replace a choice.

## Reward

Define four normalized economic outcomes for each market step:

```text
P  = Firm1 contribution profit/request / $2
DP = (Firm1 profit/request - Firm2 profit/request) / $1
S  = Firm1 completed-request share / 0.40
DS = (Firm1 completed share - Firm2 completed share) / 0.10
```

The profit policy uses exactly two active terms:

```text
R_profit = 0.80 P + 0.20 DP
```

The competitiveness policy uses exactly three active terms:

```text
R_competitiveness = 0.10 P + 0.20 S + 0.70 DS
```

The two actors are trained independently from initialization; one objective is
never fine-tuned from the other. The terms are linear and unclipped. The four
denominators are units, not success thresholds. Public quote gaps, service,
coverage, fulfillment, wait, margin, tariff movement, reversals, one-step
trends, action identity, episode boundaries, and validation results cannot add
reward. They remain report fields only.

For a held pricing action, Core computes the discounted sum of every market
reward observed before both firms may act again:

```text
held_reward = sum(gamma**k * r_k)
discount    = gamma ** duration
```

The transition therefore has the exact semi-MDP Bellman form
`R_t + discount_t * V(s_(t+1))`. V23/v24 compatibility mode retains the former
discounted-mean convention; v25 uses the exact sum and records the discounted
exposure separately for rate diagnostics. Accumulation uses float64, so the
credited return is independent of log formatting and avoids long-horizon
float32 cancellation.

## PPO credit and update

Defaults:

```text
gamma                    0.999 per market step
GAE lambda               0.99 per pricing decision
complete rollout         96 decisions
batch size               48
epochs                    8
clip epsilon             0.20
entropy target           30% of feasible categorical maximum
entropy coefficient      adaptive in [0.0005, 0.040]
behavior exploration     0.30 -> 0.05 cosine decay (v25 only)
critic learning rate     3x actor learning rate
```

For transition-specific `discount_t = gamma ** duration_t`:

```text
centered_reward_t = reward_t - mean(reward over this rollout)
delta_t  = centered_reward_t + discount_t * V(next_state) - V(state_t)
adv_t    = delta_t + discount_t * lambda * adv_(t+1)
return_t = adv_t + V(state_t)
```

The reward centering makes the critic a differential-value critic for this
continuing market. It removes the artificial value ramp caused only by a
transition's distance from the end of a rollout; it does not change the actor's
within-rollout action ranking or any logged business reward.

With v24, the rollout-local mean is replaced by a persistent exponential
reward-rate estimate (`beta=0.05` by default). An optional potential transform
uses exactly `r' = r + scale * (discount * Phi(next) - Phi(current))`; the
default v24 scale is `0.05`, and its telescoping identity is regression-tested.
The potential is a fixed function of public profit, completed-share, and
fulfillment observations, so privileged critic fields cannot reach actor credit.
Raw business reward remains separately logged.

V25 uses a lagged reward-rate baseline keyed by the terminal-delimited opponent
regime, then reconstructs each transition's baseline with that transition's
discounted exposure. A completed episode updates the cache only after all
advantages for that population have been fixed. At cold start, a segment uses
a leave-one-segment-out population rate (or zero when it is the only segment).
It never subtracts the episode's own realized mean: that action-dependent
centering erased the signal from a tariff that was consistently better for the
whole episode. The lagged center is action-independent and changes every fixed-
horizon regime return only by an action-independent constant; raw reward,
action ordering, and the named economic objective are unchanged.
The cached rate uses a 0.20 exponential update, avoiding a full target-
coordinate shift when one noisy population finishes.

True terminals stop bootstrapping. Training-stage, episode, rollout, and final
horizon cuts are continuing-task truncations and bootstrap the final observed
state; they are never assigned zero final value.  **Correction (2026-10 review):** in
the v25 code, opponent-episode and stage openings were stored with
`done=True` and therefore received zero final value; only rollout-tail cuts
bootstrapped.  v26 `--truncation_bootstrap` implements the behavior described
here. V23/v24 retain the standard
five-step trace multiplier of approximately `0.985`. V25 uses `lambda=1` over
complete opponent episodes, so its actor receives the full discounted episode
return while the separately trained critic remains TD(0).

The 96-decision minimum controls ordinary v23/v24 updates. V25 batches a whole
opponent population before each dynamic-stage update. Fast controllers use
three independently randomized 15-day terminal openings per population, while
PI price-gap and regional supply-demand retain one 300-day episode each so
their delayed response is observed rather than cut off. The default population
minimum is computed from that expanded cycle (732 pricing decisions with four
market steps/day and five-step actions). True episode
terminals partition reward-rate baselines and GAE recursion. Stage cuts remain
continuing-task truncations. A final incomplete population is quarantined
rather than applying a biased last update dominated by whichever opponent
happened to occur first; every complete population update is retained. Loss
weights first give every opponent controller class equal total mass and then
split that mass equally across its openings and transitions. Thus neither the
deliberately long PI/regional horizons nor replication of fast-controller
openings changes the intended opponent objective.

Advantages are standardized per rollout and clipped. The actor uses standard
clipped PPO plus categorical entropy. Its entropy coefficient rises when the
raw actor becomes too concentrated and falls when it is already diverse. V25's
training behavior distribution is
`mu=(1-epsilon)*pi + epsilon*Uniform(feasible)`: epsilon stays at 0.30 through
the first 5% of training, then cosine-decays to 0.05 at completion. The
sampled transition stores its epsilon and PPO recomputes the same `mu` for the
new log probability, preserving the on-policy likelihood ratio. Hold is part
of the uniform feasible set, and deterministic validation/evaluation uses the
raw actor, so exploration neither forces a move nor regulates deployment. V23
and v24 compatibility behavior is unchanged. In v23, approximate KL is
diagnostic only. V24/v25 stop remaining actor epochs when they exceed
`target_kl`.

V25 actor credit uses complete terminal-episode Monte-Carlo advantages
(`lambda=1`) while the critic retains its separate actual-next-state TD(0)
objective. Critic residual error therefore cannot truncate delayed actor
credit. Each population receives the configured eight critic passes but only
four actor passes, preventing a single nonstationary opponent population from
overwriting the actor while still allowing the weaker value head to catch up.
The critic outputs differential values and
minimizes a raw Huber error against the same bootstrapped returns. A three-times
actor learning rate prevents the separate critic from lagging the changing
policy.

At each v24/v25 real decision, up to three feasible actions (selected, actor top,
actor runner-up, then deterministic fill) are evaluated for one market step
under common random numbers. Every simulator, controller, observer, supply, and
RNG mutation is transactionally rolled back before the real action. Centered
branch rewards supervise a critic-only action-Q head. V25 does not use those
one-market-step branches as actor targets because their ordering need not match
the complete five-step semi-MDP return after supply and rival response. An
explicit ablation flag can still test the v24 selected-action blend and
pairwise ranking loss; ties have zero loss and unobserved actions receive zero
direct gradient. In the production v25 contract Q never reranks, forces, or
directly trains a deployment action. Clipped PPO over the complete held-action
return is the actor objective.

## Staged training

Default order and allocation:

1. market reference — 20%;
2. fixed static opponent — 20%;
3. dynamic opponents — 60%.

The v23/v24 final stage uses 600-day randomized openings and rotates without
omission through the configured Firm2 mode, `adaptive_best_response`,
`heuristic`, and `heuristic_random`. V25 instead rotates explicitly through
that population plus all five benchmark-controller classes, using the
response-timescale horizons above. Tariffs, observation history, controller
state, and episode metrics reset at an opening; actor, critic, entropy state,
and all Adam optimizers continue learning. Reward, action set, PPO
hyperparameters, and network architecture never change by stage.

Firm2 and Firm1 share one action interval in training, validation, and
evaluation. Firm2 posts first; Firm1 then observes its public quote and acts.
Explicit unequal frequencies fail immediately.

Convergence statistics and validation are diagnostic. They do not narrow the
action set, reduce learning, freeze an optimizer, or trigger early stopping.
The final durable dynamic-stage validation snapshot is the evaluation policy.
If the final snapshot fails the direct late economic-durability test, v25 falls
back to the highest-objective snapshot that previously passed that same test;
an uncertified high score cannot replace a durable policy. If no snapshot has
yet passed, evaluation uses the observed snapshot closest to certification
(largest feasible-opponent fraction, then smallest dominance shortfall, then
the economic objective score) and labels it explicitly as not yet durable,
rather than arbitrarily using a worse final update.

Policy diversity, switching, and state-action sensitivity are also diagnostic.
They do not contribute reward or checkpoint eligibility. A static policy is a
valid solution when it maximizes the objective; it is never prevented by a
forced move, action mask, movement quota, or minimum-switch gate.

## Validation and evaluation

V25 checkpoint validation uses independent openings against the configured
rival and all five benchmark controller classes; earlier versions retain
configured-rival validation. This matches v25's opponent-population objective
and prevents a checkpoint that improves one controller while catastrophically
forgetting another from being labeled durable. Final benchmark trajectories
use a disjoint random seed from checkpoint selection.
Its scalar is the same objective reward:

```text
checkpoint score = 0.40 * full-horizon mean reward
                 + 0.60 * final-quarter mean reward
```

This changes only temporal aggregation, not the economic objective. The fitted
trend of a few noisy validation snapshots is reported but is not a pass/fail
criterion. Durability requires the objective-specific Firm1 lead—profit lead
for the profit policy, completed-share lead for competitiveness—to be positive
in at least 90% of late rolling windows, together with retained late evaluation
reward. The five canonical benchmark controllers remain an evaluation suite,
not training reward terms. A benchmark passes only when its final-quarter lead
is positive and the late rolling dominance rate reaches 90%.

Evaluation uses the raw actor (`argmax_raw` by default). Post-batch stabilization
and evaluation guardrails are off and cannot modify the tariff.

## Artifacts and verification

Version-23 artifacts store the GRU architecture, eight-action whole-tariff
target mapping, 24-by-16 state schema, objective reward weights/scales, PPO
credit and entropy settings, shared clock/calendar, stage order, and
dynamic-opponent pool. Version-24 artifacts additionally store the complete
credit-assignment configuration and centralized-critic schema. Version-25
artifacts also store the exact transition, TD-target, direct-value,
action-feature-Q, recurrent-ingestion, population-batching, lagged-baseline,
and exact behavior-mixture contracts.
Loading rejects a v23/v24/v25 feature mismatch before network weights are
applied. Older artifacts are rejected as incompatible.

Run static and automated checks with:

```bash
python -m py_compile Core.py PPOAgent.py pricing_models.py platform_mdp.py
python -m unittest discover -v
```

For a wiring smoke, explicitly lower the long-run settings only for that smoke
(for example `--ppo_min_rollout_transitions 4 --ppo_batch_size 4
--ppo_update_epochs 1`). Substantive training should retain the v23 defaults.
