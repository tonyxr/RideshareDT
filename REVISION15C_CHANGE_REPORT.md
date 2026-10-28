# CRISP-Ride Revision 15c

Revision 15c branches from Revision 15b without modifying any 15b code or run
directory. It restores coefficient-specific pricing, keeps the four-stage
curriculum, and makes Stage 4 explicitly market-share defensive.

## Actor action contract

The actor uses a factorized categorical policy:

1. Select `hold` or one of the tariff coefficients identified by the city's
   public calibration: `base_fare`, `per_minute`, `per_mile`, `booking_fee`,
   and, where identified, `airport_fee`.
2. Conditional on a coefficient, select an absolute multiplier from
   `0.6, 0.7, ..., 2.0` relative to that city's calibrated coefficient.

This produces 76 joint actions in NYC and Chicago: one hold plus 5 coefficients
times 15 magnitudes. Boston has 61 actions because its public quote data do not
identify an airport surcharge and the calibrated airport anchor is zero; the
15 guaranteed no-op airport targets are excluded. Only the selected coefficient
changes. The target and conditional magnitude heads are trained jointly by PPO.
The former Revision 15 global
level-step action contract is not used.

## Four-stage training contract

For every city the 4,000 training days are allocated as follows:

| Stage | Days | Opponent source |
|---|---:|---|
| Market | 200 | calibrated city-market reference |
| Static | 200 | fixed opponent |
| Catch-up | 1,500 | four public price-gap/catch-up rivals |
| Profit maximizers | 2,100 | five optimized benchmark objectives plus myopic, forward, and anticipating profit best responders |

Every stage owns distinct exploration, entropy, and learning-rate endpoints.
Each control follows a local cosine decay and resets at the next stage. No
legacy CLI override replaces the Stage 3 or Stage 4 endpoints.

## Reward scalarization

The reward remains a linear economic outcome objective. It does not use quote
gaps, action switches, or critic errors as reward.

| Stage | Own profit | Profit lead | Completed-share lead |
|---|---:|---:|---:|
| Market | 1.0000 | 0.0000 | 0.00 |
| Static | 0.6175 | 0.3325 | 0.05 |
| Catch-up | 0.4050 | 0.4950 | 0.10 |
| Profit maximizers | 0.2625 | 0.4875 | 0.25 |

Stage 4 therefore assigns 73.75% of its scalarization to competitive outcomes
(profit lead plus completed-share lead) while retaining 26.25% own-profit
credit. The component critic keeps four raw outputs in every stage; zero Stage
1 share weights do not change the critic width at a stage boundary.

## Persistence and evaluation wiring

Both stage schedules are saved in the policy artifact, restored by evaluation,
and checked for exact agreement on reload. The three-city launcher performs:

- full staged training;
- the standard benchmark suite;
- the wider held-out evaluation against static, canonical, catch-up, optimized,
  and profit-best-response rivals; and
- the 5,000-row Kaggle price reality-gap analysis for each city.

City calibration artifacts preserve the measured demand and supply operating
points while extending the action-grid declaration with the new 1.7x and 1.9x
coefficient targets. RL warm start remains disabled.

## Verification completed before launch

- Focused Revision 15c and reward/curriculum tests passed.
- Legacy competitiveness, PPO, and platform-reward compatibility tests passed.
- A 20-day four-stage NYC smoke crossed all four stages, saved and reloaded the
  policy, ran terminal multi-opponent validation, and completed held-out
  evaluation without numerical or contract errors.
- The smoke visibly exercised multiple individual coefficients and magnitudes,
  including downward and upward changes and hold actions.

The smoke's durability gate is not interpreted as a performance result because
the deliberately short 8-day held-out horizon is below the durability minimum.
