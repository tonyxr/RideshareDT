# Revision 9: independent coefficient actions and share-aware profit reward

## Implemented contract

- The learned action is hierarchical: first choose `hold` or exactly one of
  `base_fare`, `per_minute`, `per_mile`, `booking_fee`, and `airport_fee`;
  conditional on a coefficient choice, choose one absolute multiplier from the
  configured tariff grid.
- The flat simulator action count is `1 + coefficient_count * grid_size`, but
  PPO represents its probabilities as
  `P(hold or coefficient) * P(multiplier | coefficient)`. Hold has no unused
  magnitude choice.
- Applying an action changes only its selected coefficient. Previously posted
  values of the other coefficients are preserved. Repeating a target does not
  create incremental drift.
- Driver incentives are not part of this action space. A non-unit
  `driver_incentive_levels` argument is rejected instead of silently creating
  a second control channel.
- Saved policies record and validate the coefficient targets, multiplier grid,
  and factorized action semantics. Earlier whole-tariff checkpoints are not
  silently loaded into this model.

## Reward contract

The profit objective remains linear and unclipped. No action penalty,
threshold gate, policy controller, or fallback was added.

- Environment stage: 95% own contribution-profit term and 5% completed-share
  lead.
- Opponent stages and evaluation: 95% of total weight remains on the configured
  own-profit/profit-lead trade-off, with 5% on completed-share lead. With the
  current 0.60 opponent profit-lead setting, the weights are 0.38 own profit,
  0.57 profit lead, and 0.05 completed-share lead.
- Completed-share lead is signed and linear:
  `(own completed share - rival completed share) / 0.10`. Losing share therefore
  reduces reward; exceeding the rival increases it.

## Verification

- Focused action/reward/critic contract suite: 188 passed, 2 intentionally
  deselected legacy source-snapshot checks.
- Broad supported suite: 624 passed, 1 skipped, 2 intentionally deselected.
- An earlier unfiltered run reached 610 passes before its 20-failure cutoff.
  Its failures were classified as: missing optional `pyarrow`; macOS/OpenMP
  shared-memory aborts in subprocess tests; obsolete whole-tariff/reward source
  snapshots; and several stale numerical assertions. The genuine numerical
  assertions and flat-PPO compatibility issue were corrected and retested.
- The legacy flat PPO network remains byte-structure compatible when action
  factorization is not requested; the new factor buffers are non-persistent.

No production training was started. Revision 9 is isolated from revision 8 so
existing runs and artifacts were not modified.
