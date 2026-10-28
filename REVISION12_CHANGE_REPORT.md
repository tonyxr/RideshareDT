# Revision 12: coefficient-specific magnitude actions

Revision 12 is forked from Revision 11. It preserves the Revision 11 reward,
observation, shortage-signal, opponent, and four-stage training changes while
replacing the verification run's whole-tariff action contract.

## Action contract

- Managed coefficients: `base_fare`, `per_minute`, `per_mile`, `booking_fee`,
  and `airport_fee`.
- Magnitudes: 0.6x, 0.7x, 0.8x, 0.9x, 1.0x, 1.1x, 1.2x, 1.3x, 1.4x,
  1.5x, 1.6x, 1.8x, and 2.0x each coefficient's market anchor.
- Discrete actions: one hold plus five coefficient targets times thirteen
  magnitudes, for 66 actions.
- A decision changes exactly one coefficient. The other four posted
  coefficient values persist, allowing a differentiated tariff to be assembled
  across decisions without a 13^5 Cartesian action space.

The Revision 12 launchers reject `--whole_tariff_actions` before starting and
explicitly lock the manipulated coefficient count to five.

## Canonical entry points

- Smoke: `scripts/run_revision12_smoke_only.py`
- Balanced 8,000-day run: `scripts/run_revision12_long_only.py`
- Paired oracle/public verification: `scripts/run_revision12_verify.py`
- Background launcher: `scripts/start_revision12_balanced_8000.command`
