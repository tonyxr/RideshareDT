from __future__ import annotations

import json
import math

from scripts.v18_report import finite_mean, parse_seed, summarize_raw


def test_parse_seed_and_finite_mean() -> None:
    assert parse_seed("nyc_s303") == 303
    assert parse_seed("published_card") is None
    assert finite_mean([1, 2, float("nan"), None]) == 1.5


def test_summarize_raw_uses_rollout_cells(tmp_path, monkeypatch) -> None:
    import scripts.v18_report as report

    root = tmp_path
    monkeypatch.setattr(report, "REPO", root)
    path = root / "runs/v18/eval/E1/nyc/A1/nyc_s101/raw.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "results": {"learned": {
            "rival__opening_1": {
                "objective": 1.0, "own_profit": 3.0, "rival_profit": 2.0,
                "profit_lead": 1.0, "own_share": .5, "rival_share": .4,
                "fulfillment": .9,
            },
            "rival__opening_2": {
                "objective": 3.0, "own_profit": 5.0, "rival_profit": 2.0,
                "profit_lead": 3.0, "own_share": .6, "rival_share": .3,
                "fulfillment": .8,
            },
        }}
    }), encoding="utf-8")
    row = summarize_raw(path)
    assert row["suite"] == "E1"
    assert row["city"] == "nyc"
    assert row["case"] == "A1"
    assert row["policy"] == "nyc_s101"
    assert row["seed"] == 101
    assert row["n_cells"] == 2
    assert math.isclose(row["objective"], 2.0)
