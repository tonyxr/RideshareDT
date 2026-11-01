import copy
import hashlib
import json

import pytest

from scripts import v17_freeze
from scripts.v17_config import canonical_hash, load_config


def _evidence(tmp_path):
    config = load_config()
    report = {
        "phase": "T3-short",
        "short_probe": True,
        "state": "passed",
        "pass": True,
        "config_hash": canonical_hash(config),
        "training_days": 60,
        "checkpoint_days": [30, 45, 60],
        "evaluation_days": 28,
        "evaluation_steps": 168,
        "validation_openings": 3,
        "horizons": {"5": {}, "7": {}},
        "selected_horizon": 5,
        "production_length": {"training_days": 1200},
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    status = {
        "phase": "T3-short",
        "state": "passed",
        "pass": True,
        "eligible_to_freeze_config": True,
        "config_hash": canonical_hash(config),
        "report_sha256": digest,
    }
    return config, status, report, report_path


def test_freeze_requires_and_accepts_hashed_passed_long_evaluation(tmp_path):
    config, status, report, report_path = _evidence(tmp_path)
    assert v17_freeze.validate_freeze_evidence(
        config, status, report, report_path=report_path
    ) == (5, 1200)


def test_freeze_rejects_stale_config_and_tampered_report(tmp_path):
    config, status, report, report_path = _evidence(tmp_path)
    stale = copy.deepcopy(status)
    stale["config_hash"] = "stale"
    with pytest.raises(ValueError, match="stale"):
        v17_freeze.validate_freeze_evidence(
            config, stale, report, report_path=report_path
        )

    report_path.write_text(report_path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        v17_freeze.validate_freeze_evidence(
            config, status, report, report_path=report_path
        )


def test_freeze_rejects_non_freeze_diagnostic(tmp_path):
    config, status, report, report_path = _evidence(tmp_path)
    status["eligible_to_freeze_config"] = False
    with pytest.raises(ValueError, match="not eligible"):
        v17_freeze.validate_freeze_evidence(
            config, status, report, report_path=report_path
        )


def test_freeze_rejects_two_day_evaluation_even_when_report_is_hashed(tmp_path):
    config, status, report, report_path = _evidence(tmp_path)
    report["evaluation_days"] = 2
    report["evaluation_steps"] = 12
    report_path.write_text(json.dumps(report))
    status["report_sha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="too short"):
        v17_freeze.validate_freeze_evidence(
            config, status, report, report_path=report_path
        )
