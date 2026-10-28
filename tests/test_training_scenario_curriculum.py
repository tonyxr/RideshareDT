import json
from types import SimpleNamespace

from Core import Core, load_training_scenario_manifest


def test_manifest_and_opponent_balanced_rotation(tmp_path):
    scenario = tmp_path / "shortage.json"
    scenario.write_text(json.dumps({"start_date": "2023-01-09", "events": [
        {"kind": "driver_shortage", "start_day": 4, "end_day": 7, "intensity": .4}
    ]}))
    manifest = tmp_path / "curriculum.json"
    manifest.write_text(json.dumps({
        "format": "ride-response-training-scenarios-v1",
        "cases": [{"name": "natural", "scenario": None},
                  {"name": "shortage", "scenario": "shortage.json"}],
    }))
    entries = load_training_scenario_manifest(manifest)
    core = Core.__new__(Core)
    core.v26_settings = {}
    core.market = SimpleNamespace(shocks=SimpleNamespace(scenario=None))
    core.training_scenario_curriculum = ()
    core.configure_training_scenario_curriculum(entries)
    observed = [core._set_training_episode_scenario(i, 2) for i in range(4)]
    assert observed == ["natural", "natural", "shortage", "shortage"]
    assert core._set_training_episode_scenario(4, 2) == "natural"
    assert core.market.shocks.scenario is None


def test_bad_or_empty_manifest_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"format": "ride-response-training-scenarios-v1", "cases": []}))
    try:
        load_training_scenario_manifest(path)
    except ValueError as error:
        assert "no cases" in str(error)
    else:
        raise AssertionError("empty curriculum was accepted")
