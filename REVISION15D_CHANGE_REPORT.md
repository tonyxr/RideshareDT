# Revision 15d change report

Revision 15d preserves Revision 15c's simulator, four-stage curriculum,
opponent populations, reward schedules, benchmark evaluation, and Kaggle
reality-gap analysis. It changes the policy action and update contracts.

## Hybrid relative action contract

The policy selects one of:

- hold;
- whole-tariff `-0.20`, `-0.10`, `+0.10`, or `+0.20` relative steps; or
- one identifiable coefficient and one of those four signed relative steps.

Relative means additive city-anchor units. If a coefficient is currently at
`0.80 x anchor`, `+0.20` posts `1.00 x anchor`; repeating it posts
`1.20 x anchor`. It does not jump to a fixed absolute target.

NYC and Chicago expose five coefficients and therefore 25 actions. Boston's
public calibration does not identify a positive airport surcharge, so Boston
exposes four coefficients and 21 actions. An action that would have no effect
because every targeted coefficient is already at its relevant cap is masked
before behavior sampling and PPO likelihood calculation. A whole-tariff action
remains feasible if at least one coefficient can move.

The causal action-feature matrix receives the exact signed relative step and
computes bounded post-action fare effects. Counterfactual credit, action-impact
diagnostics, dataset evaluation, checkpoint metadata, and checkpoint reload all
use the same mapping.

## Policy update and credit contract

- Independently randomized trajectory length: 14 days.
- Complete trajectories per PPO update: 2.
- Minimum transitions per update: 28 (one pricing decision per day).
- Update readiness check: every 2 days; optimization occurs as soon as two
  complete 14-day trajectories are available.
- PPO minibatch size: 14; epochs: 12. A normal 28-transition update therefore
  makes 24 minibatch optimizer steps, instead of one full-batch step per epoch.
- Discount: `gamma=0.995`; GAE lambda: `0.90`. At the six-market-step pricing
  cadence the effective trace is approximately 7.9 pricing decisions/days.
- NMCC horizon: 12 days. This exceeds the memory rival's approximate complete
  10.8-day reaction window (1-day lag + 5.5-day retaliation hold + 4.3-day
  recovery), so paired branches observe the delayed cost of a move. Isolated
  branch simulations always replay all 12 days and are not truncated when the
  corresponding real PPO trajectory is close to its 14-day boundary.
- NMCC cadence: every third pricing decision.
- NMCC coverage: every hybrid action (25 branches in NYC/Chicago, 21 in
  Boston), with cap-infeasible actions masked.
- Branch-imitation coefficient: 0.05.

The update readiness check is intentionally more frequent than the update
itself. This prevents premature short rollouts while allowing an update
immediately after enough complete long-horizon evidence exists.

## Evaluation and exposure contract

The evaluation no longer labels all benchmark rivals as held out:

- the four catch-up, five profit-optimized, and three profit best-response
  controllers are explicitly labelled `training_exposed`;
- the static rival and five older rule-based controllers are labelled
  `genuinely_held_out`; and
- deterministic parameter perturbations of the five older rule-based
  controllers are labelled `genuinely_held_out_perturbed`. The static rival
  has no controller parameters to perturb.

Every case now runs the adaptive learned actor beside three control families
under common random numbers: the same actor acting once and then freezing its
posted differentiated tariff, the randomized opening tariff held fixed, and
the complete 13-point whole-tariff grid. The city-specific differentiated
tariff found by Revision 15b's structure search is also replayed directly from
its hash-recorded artifact. Reports retain every fixed-grid row. If a summary
selects the best fixed grid point on that same evaluation cell, it calls it an
**ex-post grid oracle**, never held-out policy selection.

The case registry's own `train+eval`, `eval`, `holdout`, and `null` roles are
also preserved in manifests rather than calling all 12 scenarios held out.

## Entry point

Use `scripts/run_revision15d_multicity_full.py`. It writes both action and
learning contracts into the root manifest and each city pipeline contract,
then retains the inherited standard evaluation and Kaggle price reality-gap
workflow. The Revision 15d full benchmark sweep expands that workflow to the
exposure-labelled opponent and matched comparator matrix above.
