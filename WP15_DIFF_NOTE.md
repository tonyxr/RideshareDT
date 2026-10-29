# WP15 - progressive objective, live own-fleet observation, and V17 preflight tooling

WP15 records the user-directed extensions that sit on top of the fourteen
committed V17 work packages.  It does not include the RevB F1-F4 corrections;
those are a separate WP16 so their effect can be reviewed and tested cleanly.

## Runtime contract

- `v17_progressive` keeps the manual's four curriculum stages and allocations
  (5/5/30/60 percent), but moves the profit-lead weight through
  `0.0, 0.5, 0.625, 0.75`.  The market stage is pure own-profit
  maximization.  Later stages progressively value direct competitiveness.
  Profit and profit lead use one common dollar scale and there is no
  market-share shaping term.  The fixed WP14 reward is rejected when the
  progressive curriculum is active.
- Both platform actors receive their exact current own-platform active-driver
  count.  The feature is invertibly encoded as `arcsinh(count/base_fleet)`, is
  current rather than delayed, and never exposes the opponent's fleet count.
  The existing smoothed own-supply block remains in the observation.
- The concave two-lever value model initializes the diagonal of its curvature
  factor to `0.10` instead of the zero-gradient point at `0.0`.  The initial
  maximum remains the card at `(1.0, 1.0)` while curvature can learn from the
  first optimizer step.

## Market and preflight corrections included

- V3 supply is wired without the incompatible persistent-stock switch.
- Realistic fares were refit by scaling only per-mile and per-minute rates;
  base, booking, and airport charges stay fixed under two-lever policies.
- Direct-post two-lever comparators use a no-action sentinel, preventing the
  legacy action `0` from overwriting posted schedules with `(0.8, 0.8)`.
- NYC's T1.1 fleet-size-only refit selects scale `1.2`; V3 elasticity remains
  fixed at `0.6`.  Boston and Chicago remain at `1.0` pending their declared
  checks.
- The production config remains `prefreeze`: T3 must choose the branch horizon
  and production length before any production launcher is allowed to run.

## Tooling and evidence

- Added a single validated production config, fare and fleet refit scripts,
  the resumable T1 harness/launcher, a T0 runner that saves status and logs,
  and demand/fare calibration inputs referenced by the config.
- Added regression tests for progressive weights, exact own-driver visibility,
  flags-off identity, value-model initialization, command wiring, fare/fleet
  refits, T1 pairing, and two-lever direct posting.
- The post-fleet-refit T1 run was still in progress when WP15 was prepared.
  Runtime outputs and caches are intentionally excluded from the commit.

## Deferred to WP16 (RevB)

WP16 will relabel stored branch components to the current stage at sample time
(F1/T0.12), correct held-out competitor groups (F2/T0.13), centralize the
final `own - 0.75 * competitor` ranking objective (F3), and add the flagged
six-level `two_lever_v2` action contract without changing V1 (F4/T0.14).
