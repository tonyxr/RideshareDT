import pytest

from scripts.report_v30_profit_dominance import summarize_suite


@pytest.mark.parametrize("own,rival,tail,expected", [
    (2, 1, 0.5, True), (2, 1, -0.5, False),
    (1, 2, 0.5, False), (1, 1, 0.5, False), (2, 1, 0, False),
])
def test_both_full_and_late_profit_leads_are_required(own, rival, tail, expected):
    report = summarize_suite({"rollout_summaries": {"rival": {
        "own_profit_per_request": own, "rival_profit_per_request": rival,
        "late_objective_advantage": tail, "steps": 1000,
        "objective_reward_mean": 100, "passes_objective": True,
    }}})
    assert report["positive_full_and_tail_lead_on_every_observed_path"] is expected


def test_missing_results_cannot_claim_dominance():
    assert summarize_suite({})["positive_full_and_tail_lead_on_every_observed_path"] is None


def test_invalid_results_are_not_mislabeled_as_a_loss_or_win():
    with pytest.raises(ValueError, match="Nonfinite"):
        summarize_suite({"rollout_summaries": {"rival": {
            "own_profit_per_request": float("nan"), "rival_profit_per_request": 1,
            "late_objective_advantage": 1, "steps": 1000,
        }}})
