import pytest

from scripts.refit_v17_fleet import _replace_flag, select_candidate


def test_replace_flag_changes_only_requested_value():
    command = ["Core.py", "--fleet_scale", "1.0", "--policy_sweep", "x;y"]
    assert _replace_flag(command, "--policy_sweep", "x") == [
        "Core.py", "--fleet_scale", "1.0", "--policy_sweep", "x"
    ]


def test_select_candidate_prefers_closest_passing_scale():
    candidates = [
        {"fleet_scale": 1.1, "absolute_gap": 0.019, "within_two_points": True},
        {"fleet_scale": 1.2, "absolute_gap": 0.004, "within_two_points": True},
        {"fleet_scale": 1.3, "absolute_gap": 0.003, "within_two_points": False},
    ]
    assert select_candidate(candidates)["fleet_scale"] == pytest.approx(1.2)
