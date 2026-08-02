# RideResponse policy-version analysis

## Executive conclusion

The experiments do not form a simple sequence in which each higher version is
better. They explore three different objectives:

1. **Profit/dominance:** maximize the RL firm's contribution profit and often
   suppress the rival (`v6`, `v7`, some short-horizon `v8` runs).
2. **Competitive durability:** remain profitable after a responsive rival has
   time to catch up (`v8 longgate`, `v9f`, `v13`, `v19`).
3. **Economic balance:** retain profit without sacrificing too much volume,
   price discipline, or policy responsiveness (`v8 economic`, `v18`, `v19`).

No supplied 1,000-day policy passes the current deployment durability test.
The strongest pure-profit results are not the strongest robust-competition
results, and vice versa.

The user's characterization that **v6 focuses on profit while v8 focuses on
competitiveness is directionally correct, but incomplete**:

- v6's stored reward puts weight `1.0` on own contribution profit and only
  `0.10` on profit advantage. Its 500-day result produces a very high late
  profit advantage (`+2.912/request`) and share (`71.7%`) at the cost of a
  large fare-gap error (`$4.62`).
- `v8 economic` increases the rival-relative term to `0.35`, while the v8
  generation also adds direct rival exposure, faster PPO updates,
  state/action specialization, longer checkpoint horizons, and durability
  screening.
- Other v8 variants still use `1.0/0.10`. Several 200-day v8 runs therefore
  remain aggressive profit/dominance policies, not balanced competitors.

## How results were normalized

For each primary run I used:

- the final quarter of evaluation as the late-performance window;
- completed share, contribution profit per incoming request, rival-relative
  profit, reward retention, price-gap error, and policy action diversity;
- stored configuration and backtest outcomes;
- 1,000-day audits as the main durability evidence, because a 200-day audit
  often ends before the rival completes its catch-up response.

The fare-gap threshold population is effectively constant across these runs
(mean about `$1.167`, median `$1.00`, standard deviation about `$0.427`).
Consequently, cross-version behavior is driven by the policy, curriculum,
opponent, horizon, and checkpoint selection—not by different customer
threshold samples.

## Main lineage comparison

| Version | Late evaluation result | Advantages | Disadvantages | Why it behaves this way |
|---|---:|---|---|---|
| v4 | 500d: share `52.0% vs 38.9%`; profit advantage `+1.015`; reward `0.550` | Reasonably balanced market/share and profit | Earlier reward is a multi-term proxy; weaker causal and durability machinery | Reward explicitly mixed profit, revenue, completed demand, and service, so it rewarded business balance rather than only contribution profit |
| long-profit transition | 500d: share near parity; profit advantage `+0.643`; reward `0.584` | Profitable without eliminating rival | Limited share advantage; weaker dynamics | First shift toward long-run profit (`0.9/0.1` own/relative weights) |
| v5 | 1,000d: share `24.3% vs 20.4%`; profit advantage `+0.294`; reward `0.165`; one late action | Long horizon exposes true steady state | Low reward, low volume, policy response collapses | Higher relative weight (`0.7/0.3`) plus sparse/slow PPO updates makes the agent conservative and poorly adaptive |
| v6 | 500d: share `71.7% vs 18.8%`; profit advantage `+2.912`; reward `0.811`; gap error `$4.62` | Best apparent profit and dominance among early versions; economically varied actions | Excessive undercutting; weak price discipline; only 500 days; not tested against today's durability suite | Own-profit-dominated reward (`1.0/0.10`), no robust multi-opponent checkpoint gate, and shorter evaluation favor an exploitative “win now” policy |
| v7 | 1,000d: share `59.9% vs 21.9%`; profit advantage `+1.904`; reward `0.579`; gap error `$1.73` | Keeps much of v6's profit while reducing gap error; 12 late action groups in the main run | Still highly asymmetric; top-two sampling adds evaluation variance; duplicate/mislabeled artifacts limit seed evidence | Longer audit and `top2_margin` action selection prevent immediate deterministic lock-in while preserving the v6 reward |
| v8 short gate/direct family | 200d: generally share `62–78%`; rival share near `0–10%`; profit advantage `+1.0 to +2.9`; gap error `$5.8–12.4` | Very strong short-run takeover | Rival collapse, huge price error, often only one late action; fails current durability test | Audit ends during the undercut/takeover phase, before rival catch-up. These are opponent-collapse policies, not evidence of durable competition |
| v8 longgate | 1,000d: share `40.3% vs 36.5%`; profit advantage `+0.225`; reward `0.351`; gap error `$0.906` | Best v8 balance of price discipline, share, and positive profit; stable long-run coexistence | Only `+3.8pp` share advantage; reward retention fails; two late actions | Long direct exposure lets the rival recover and forces convergence; the policy becomes stable but not decisively superior |
| v8 economic | 1,000d: share `38.7% vs 39.4%`; profit advantage `-0.043`; reward `0.314`; gap error `$1.153` | Dynamic (four late actions), price gap near threshold scale, healthy coexistence | Loses marginally on both late share and profit; reward retention only about `48%` | Larger relative-profit term (`0.35`) and direct competition encourage response to the rival, but the policy chases a moving opponent rather than protecting its own profit floor |
| v9f | 1,000d: share `37.7% vs 40.2%`; profit advantage `-0.141`; reward `0.302` | More realistic competitive balance than collapse runs | Late economic loss and only two actions | Dynamic curriculum is present, but checkpoint/policy quality is insufficient to turn adaptation into profitable differentiation |
| v13 | 1,000d: share `39.2% vs 38.6%`; profit advantage `+0.004`; reward `0.320`; five actions | Near parity, dynamic, lower gap error (`$1.068`) | Essentially no profit edge; validation trend and reward retention fail | Lower MI pressure avoids artificial action churn, but the learned policy converges to competitive parity rather than durable advantage |
| v18 | 1,000d: share `26.5% vs 37.4%`; profit advantage `+0.519`; reward `0.493`; retention `79.7%` | Strongest durable profit extraction among the recent long audits; passes every current criterion except share | Low volume/share, gap error `$2.785`; not a competitive-share winner | Selects a high-margin/low-volume position: it earns more per incoming request despite completing fewer rides |
| v19 | 1,000d: share `28.5% vs 27.5%`; profit advantage `+0.077`; reward `0.282`; gap error `$1.167`; eight raw late actions | Much better price discipline, near share parity, highly responsive | Gives up most of v18's profit edge; reward retention only `50.4%`; just `+1.0pp` late share advantage | Stronger responsiveness improves balance but creates frequent tariff movement and weakens own-profit retention; aggressive held-out rivals erase most of its early advantage |

## The most important diagnostic: 200 days versus 1,000 days

`dynamic_v8_seed9127_final` is the cleanest controlled warning:

| Audit | Late share | Rival share | Profit advantage | Late reward | Late actions |
|---|---:|---:|---:|---:|---:|
| 200 days | `77.8%` | `9.5%` | `+2.934` | `0.744` | 1 |
| 1,000 days, same final policy | `26.4%` | `26.9%` | `+0.237` | `0.214` | 1 |

The 200-day result records the temporary benefit of a large undercut. The
1,000-day result includes the rival's price response and shows that the frozen
one-action policy cannot defend the early lead. This is why all short gate,
direct, and symmetric runs should be treated as screening experiments, not
final version comparisons.

## Why the objective produces these behaviors

The present long-term reward is approximately:

`asinh(own profit / scale) + weight × own-profit-quality × tanh(profit advantage / scale)`

The rival-relative component is gated by positive own profit. This is a useful
anti-price-war design: beating the rival is not valuable if the RL firm is
destroying its own economics. However, the weight and the evaluation protocol
still change the learned strategy:

- **Large own-profit weight, weak durability selection:** exploit profitable
  undercuts and accept a large fare gap (v6).
- **Larger profit-advantage weight:** react more to the rival, potentially
  sacrificing absolute profit to preserve parity (v8 economic, v18/v19).
- **Long direct curriculum:** experience the complete undercut/catch-up/
  recovery cycle, which reduces false early dominance (longgate onward).
- **State/action MI regularization:** encourages different economic actions in
  different states. It prevents a frozen one-action policy, but excessive
  pressure can create action churn without a profit benefit.
- **Multi-opponent checkpoint validation:** rejects policies that win only
  against a single deterministic rival. In v19, the day-1200 checkpoint won all
  held-out rollouts but its worst aggressive-rival advantage was only `+0.01`;
  later checkpoints lost against at least one aggressive seed.

Price gap is a diagnostic rather than a scalar reward term. That is why a
policy can have high reward while maintaining a very large gap, and why lower
gap error should not by itself be called a better policy.

## Pros and cons within v8

| v8 variant | Interpretation |
|---|---|
| `gate`, `gate2` | Nearly identical 200-day opponent-collapse outcomes. Gate changes do not solve long-run behavior. |
| `gate3` | More action variation and historically reports a pass, but that pass used a short/older audit. It is not comparable to today's 1,000-day gate. |
| `direct`, `direct_symmetric` | Strong short-run takeover. Symmetric initialization does not fix the one-sided collapse because the policy rapidly drives to its lower tariff region. |
| `mi50` | Slightly more action diversity than the seed-matched gate run, but still a 200-day collapse result. MI changes policy motion, not the economic objective. |
| `economic` | Best test of a stronger competitive reward, but it overreacts enough to lose a small amount of late profit. |
| `longgate` | Most informative v8 experiment: long horizon, positive profit, moderate gap, no rival elimination. It is the best v8 base for further work despite failing certification. |
| `final` / `final_eval1000` | Demonstrates horizon overfitting. The short audit looks excellent; the long audit invalidates it. |

## v18 versus v19

These are not an ordinary “old versus improved” pair:

- **v18 is the better profit policy.** It retains `79.7%` of early evaluation
  reward and earns `+0.519/request` over the rival, but does so with an
  `-10.9pp` share disadvantage.
- **v19 is the better balance/dynamicity policy.** It narrows the share deficit
  to a `+1.0pp` advantage and lowers gap error from `$2.785` to `$1.167`, but
  profit advantage falls to `+0.077` and reward retention to `50.4%`.
- v19 was never certified or saved because its durability backtest failed.
  Consequently, the separate `vs_*` benchmark commands fail with “trained
  model not found”; the benchmark evidence that is valid is the held-out
  checkpoint validation embedded in the main terminal log.

## Data-quality and comparability cautions

- v6 and v6_1 have byte-identical evaluation CSVs. v6_2 differs at the file
  level but normalizes to the same reported evaluation metrics. They should not
  be counted as three independent replications.
- v7_1 and v7_2 evaluation CSVs are byte-identical. In addition, the
  `v7_1_seed4303` config records a trained model ID containing seed `4500`.
  Treat them as one policy evaluation until lineage is repaired.
- Most headline variants have one seed. The seed-9127/27183 gate comparison
  shows similar collapse behavior, but it does not establish confidence
  intervals for long-horizon policies.
- A historical `backtest_passed=true` is not automatically comparable with the
  current test. The current criteria require a 10-point late share advantage,
  positive late profit advantage, improving held-out validation, at least 75%
  reward retention, and material late action response.
- The `.pt` files contain deployable weights, but their raw tensors do not
  explain business behavior. The run configs, evaluation trajectories, and
  held-out validation outcomes are the relevant evidence.

## Recommended next experiment

Use `v8 longgate` as the behavioral base and the v18/v19 checkpoint suite as the
selection base:

1. Run a **1,000-day audit for every checkpoint**; never promote from a
   200-day evaluation.
2. Optimize a Pareto score with hard floors:
   positive own profit, positive worst-opponent profit advantage, minimum
   completed-share floor, reward retention at least `75%`, and at least two
   economically distinct late actions.
3. Do not require a 10-point share lead if the intended product objective is
   profitable coexistence. If market leadership is required, keep it—but label
   v18 explicitly as a profitable niche strategy rather than a failure.
4. Evaluate at least three independent training seeds and five aggressive-rival
   seeds; report median and worst-case late metrics.
5. Save a checkpoint hash, effective training seed, evaluation seed, and exact
   criterion version in every config to eliminate the v6/v7 lineage ambiguity.
6. Add action-churn cost or a minimum-profit-improvement test so v19-style
   responsiveness is rewarded only when it changes economic outcomes.

### Best current artifact by objective

- **Maximum apparent dominance:** v6, with a strong brittleness warning.
- **Best long-horizon profit:** v18.
- **Best v8 foundation for robust coexistence:** longgate.
- **Best price discipline and responsiveness:** v19.
- **Deployment-certified policy:** none of the supplied long-horizon runs.

## Local evidence

- Normalized run metrics: `analysis/generated_version_metrics.json`
- Reproducible extraction: `analysis/analyze_versions.py`
- Reward implementation: `platform_mdp.py`, especially the long-term reward
  computation around lines 174–185
- Current curriculum and validation design: `PLATFORM_MDP.md`
- Current durability criteria: `Core.py`, around lines 3520–3610

