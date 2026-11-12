# CRISP-Ride V18 execution and result-generation plan

Status: code-preparation only. No V18 simulation, download, calibration, training, or evaluation job has been launched.

## Fixed execution sequence

1. **Code gate**
   - Run `python -m pytest -q tests/test_v18.py` and the existing V17 suites.
   - Run `python scripts/v18_preflight.py --step P1 --run-unit-tests`.
   - Complete the two 40-day V17/V18-default identity comparisons for NYC C0 and A1.
   - Stop on the first test or exact-identity failure.

2. **Preflight P2-P7**
   - P2: fit NYC driver pay, transfer it to Boston/Chicago, fit fleet and pickup, measure reference earnings, and build the nested mechanism-effect table.
   - P3: run the registered size grid and choose `N_star` once.
   - P4: time 20-day agent H5/H10, PPO, and MC-return jobs at `N_star`; set the two clock-matched day counts once.
   - P5: run the four threshold-source landscapes and record whether the optional rubric arm is required.
   - P6: probe H5/H10 credit on legacy agent/PPO states; choose `H_star` and whether `concave2d_residual` is required.
   - P7: compare proxy/full rival planning; choose 8-rider/1-day or 16-rider/2-day branches.
   - P5-P7 may overlap P2-P4 subject to the global 10-Core-process ceiling.
   - Every step writes `runs/v18/preflight/Pn/plan.json`, `report.json`, and `report.md`.

3. **Freeze exactly once**
   - Record each completed preflight report with `scripts/v18_preflight.py --step Pn --record-report <file>`.
   - Run `python scripts/v18_preflight.py --freeze`.
   - Commit the frozen `configs/v18_production.yaml` before production training.
   - Never revise `N_star`, `H_star`, threshold outcome, branch planner, clock-matched days, fitted supply values, or value-head choice inside a production run.

4. **Reference selection and Wave A**
   - Emit the three city selection plans with `scripts/v18_select.py --city <city> --emit-plan`.
   - Run all 36 cards, count rules, and reactive rules on the registered selection rivals/cases, then select from the saved landscapes.
   - Run `python scripts/v18_train.py --wave A --workers 9 --execute`.
   - Wave A contains nine main agents (three seeds per city); selection may occupy the tenth process slot.
   - Validate each policy artifact, 200-day snapshots, command hash, model switches, finite logs, and resume manifest before accepting it.

5. **Wave B**
   - Run `python scripts/v18_train.py --wave B --workers 10 --execute`.
   - Queue clock-matched PPO/MC arms first, then remaining main seeds, 1,200-day PPO, independent-noise, count-observation, own-profit, and alternate-horizon arms.
   - Run `B_optional` only when P5 requires the rubric-trained arm.

6. **Evaluation E1-E10**
   - First inspect each suite with `python scripts/v18_eval.py --suite E<n>`.
   - Launch with `--execute --workers 10` only after all required policies and selected references validate.
   - Scripted cases use `scenario_hazards=off`; training and E5 use `natural`.
   - E5 and E10 write gzip request logs. E4 writes evaluation-time credit probes. E6 overrides evaluation size to 500.
   - Accept a job only when its product is finite, complete, command-addressed, source-addressed, and carries the frozen V18 switches.

7. **Real-data validation V1-V7**
   - Commit `runs/v18/realdata/DECLARATION.json` before computing any metric.
   - Fetch and checksum the declared TLC, Chicago, Boston, and NOAA inputs.
   - Run the companion V1-V4 leakage and event-twin checks on V18 products.
   - Run V5-V7 with `scripts/v18_real_outcomes.py`; event twins use common random numbers and scripted hazards off.
   - Stop on a declaration, leakage, price-definition, request-log identity, or event-twin mismatch.

8. **Result generation**
   - Aggregate traces with `scripts/v18_aggregate.py`.
   - Compute seed-level paired intervals and Holm corrections with `scripts/v18_stats.py`.
   - Re-score rival-profit weights with `scripts/v18_rescore.py`.
   - Compute timing gains with `scripts/v18_value_of_timing.py`.
   - Generate T1-T9, F1-F8, and the release dataset from the immutable aggregate products.
   - Raw and intermediate products stay under `runs/v18/`; paper-facing tables and figures go to `Claude outputs/results_v18/`.

## Dependencies and safe overlap

```text
code/tests -> P1
          -> P2 -> P3 -> P4 -----------+
          -> P5 -----------------------+-> freeze -> selection + Wave A -> Wave B
          -> P6 -----------------------+                         |
          -> P7 -----------------------+                         +-> E1-E10
                                                                    |  |
                                              E5 request logs ------+  +-> V1-V7
                                              E10 request logs ---------> dataset
E1-E10 + V1-V7 -> aggregate -> statistics/rescoring/timing -> T1-T9/F1-F8
```

The overlap is computational, not scientific: no downstream job may consume a tentative decision or an unvalidated upstream product.

## Resource and recovery rules

- At most 10 `Core.py` processes total, with one numerical thread each.
- Runners are content-addressed and resume only completed jobs. Training also writes a daily resume state.
- A failed job may be repeated with the identical command up to two times; the third failure is a stop.
- Products are never deleted or overwritten to rescue a scientific result. A changed command creates a new content identity.
- Stop conditions are limited to the manual's test, identity, non-finite, stock-drift, repeated-crash, and logging-identity failures. Scientific outcomes are reported.

## Expected duration

The manual's planning basis is about 1.5 coding days plus 95 Mac compute hours: P1-P7 about 14 h, Wave A about 15 h, Wave B about 20 h, E1-E10 about 40 h, real-data simulations about 4 h, and final products about 2 h. The observed P4 timings replace these estimates after preflight; uncertainty is high until `N_star` and the branch planner are frozen.
