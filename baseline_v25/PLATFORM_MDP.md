# RideResponse economic-objective recurrent PPO (v23/v24/v25)

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

- 5 time and weather values;
- 1 own representative-tariff position;
- 8 delayed own operating outcomes;
- 4 current demand descriptors;
- 4 sampled public Firm2-minus-Firm1 quote gaps by trip segment; and
- 2 previous-action timing values.

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
state; they are never assigned zero final value. V23/v24 retain the standard
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
