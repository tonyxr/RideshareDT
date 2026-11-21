# CRISP-Ride V18 final results audit

Generated: 2026-11-21T15:18:04.253301+00:00

## Completion

The Last Revision computation queue and V1-V7 validation program completed. Scientific tolerances are findings, not stops. The only remaining limitations are recorded below.

## Fixed transition and calibration findings

P2 is closed and was not rerun. Attempt 5 calm-stock drifts are report-only findings:

- nyc: 0.0268838325
- boston: 0.0463352890
- chicago: 0.0523553254

Production uses the fixed v17-corrected market specified by Last Revision Section 2.

## Real-data validation highlights

- V1 NYC card out-of-time: r=0.8894628818326258, rho=0.9130526509801052, R2=0.7369541710005401, MAE=7.03495072222222, bias=4.0144625000000005, slope=0.8728749103268216.
- V3 produced 1232 real/synthetic event comparisons.
- V5 published-card wait median ratio=1.565114789120367 and KS=0.4760692397028494.
- V7 common-month trips-per-vehicle-hour ratio=0.7105375652954153; trend agreement=False. January-February 2024 lack synthetic counterparts because E5 ends in December 2023.
- Chicago V1-V6 real-side comparisons were skipped after the Socrata source exhausted three operational attempts; Chicago simulation products remain retained.
- The raw NYC Kaggle August source was unavailable, so the exact August price-definition match is undefined. The conservative bank/test date guard passed.

## Statistical products

Primary agent-minus-published comparisons: 15. Bootstrap unit and correction are recorded in stats.json.

## Failed jobs

- E10/boston/A1/count_rule_1step
- E10/boston/A3/count_rule_1step
- E10/boston/C0/count_rule_1step
- E3/nyc/A1/best_fixed
- E3/nyc/A1/count_rule_0step
- E3/nyc/A1/count_rule_1step
- E3/nyc/A1/dated_two_lever
- E3/nyc/A1/published_card
- E3/nyc/A1/reactive_surge
- E3/nyc/C0/count_rule_1step
- E3/nyc/C0/reactive_surge
- E6/chicago/A1/chicago_s101
- E6/chicago/A1/chicago_s202
- E6/chicago/A1/chicago_s303
- E6/chicago/C0/chicago_s101
- E6/chicago/C0/chicago_s202
- E6/chicago/C0/chicago_s303
- E6/nyc/A1/nyc_s101
- E6/nyc/A1/nyc_s202
- E6/nyc/A1/nyc_s303
- E6/nyc/C0/nyc_s101
- E6/nyc/C0/nyc_s202
- E6/nyc/C0/nyc_s303
- S
- selection/chicago
- selection/nyc

## Skipped jobs

- None in the Section 5 queue.
- V1-V6 Chicago real-side validation: skipped because the D1 Chicago Socrata download failed operationally three times.

## Code-failure retries

- V1 attempt 1: JSON serialization defect; fixed and rerun.
- V1 attempt 2: undefined constant-baseline statistic defect; fixed and rerun.
- V2 attempt 1: superlinear Wasserstein implementation; replaced by exact linear-time empirical-CDF integration and rerun.

## Product inventory

Tables T1-T10, figures F1-F7, statistics, real-data paper numbers, and INDEX.json are colocated in this results directory or runs/v18/realdata.
