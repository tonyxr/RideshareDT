# Revision 10: Temporal Competitive Reward

## Implemented reward

Corrected opponent-stage and evaluation reward uses six linear outcome channels:

- 26% own contribution profit per incoming request
- 39% instantaneous own-minus-rival contribution-profit lead
- 20% five-day exponentially smoothed profit lead (durable dominance)
- 10% profit-lead innovation relative to the pre-step EMA forecast (adaptive residual)
- 5% completed-request-share lead

All four dollar channels use the same $2-per-request unit conversion. The
adaptive residual is an economic forecast error, not a critic TD residual.
Market pretraining remains 95% own profit plus 5% completed-share lead.

The component critic was expanded from four to six raw outcome values, and
each stored transition must reconstruct the scalar PPO reward exactly. Saved
policies record and validate the complete temporal reward contract.

## Transition-rich training

- Dynamic opponent episodes were shortened from 60 to 30 days.
- The surge/driver-incentive controller was added to the training population.
- The scripted curriculum now rotates natural calibration, severe shortage,
  forecasted transit storm, rival outage, promotion bait, and own-outage
  recovery cases across the three distinct opponent classes.
- Replicated controller episodes no longer delay scenario rotation.

## Verification

- Python compilation passed for the reward, critic, CLI, pipeline, and queue.
- Focused/regression gate: 70 passed.
- Broad suite: 651 passed, 1 skipped. The 20 failures were 18 subprocess
  OpenMP shared-memory aborts from the older test-runner interpreter, one
  missing optional `pyarrow` dependency, and one pre-existing action-index
  expectation unrelated to revision 10.
- End-to-end Chicago smoke: 192 training days and 12 evaluation steps passed,
  including six-component PPO updates and natural/shortage/storm rotation.
- Saved-policy corner reload passed at the calibrated 50-request volume against
  the surge/driver-incentive rival.

## Production rerun

The 3-seed × 3-city × 20,000-day training, standard evaluation, and full corner
matrix is queued behind the active revision-8 baseline. It will use three
workers after all nine baseline jobs complete successfully.

- Queue status: `artifacts/revision10_multiseed_temporal_20000_20261019.queue_status.json`
- Production output: `artifacts/revision10_multiseed_temporal_20000_20261019/`
