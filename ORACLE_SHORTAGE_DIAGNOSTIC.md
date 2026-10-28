# Oracle shortage diagnostic

This opt-in experiment adds one contemporaneous binary
`driver_shortage_active` feature to the Firm 1 actor observation. It is an
oracle diagnostic, not a production information claim. The default observation
schema, existing checkpoints, market mechanics, and benchmark policies remain
unchanged unless `--actor_true_shortage_flag` is explicitly supplied.

Run the preregistered NYC diagnostic with:

```bash
../.venv/bin/python scripts/run_oracle_shortage_diagnostic.py \
  --output runs/nyc_oracle_shortage_10000_20261019 \
  --days 10000
```

The run uses the existing four stages (10%, 10%, 30%, 50%), 30-day dynamic
episodes, actor/critic GAE lambda 0.90, the current reward, and the unchanged
benchmark pools. A positive diagnostic requires the deterministic raw actor to
select higher shortage tariffs conditional on the oracle flag; exploratory
price movement alone is not evidence of learning.
