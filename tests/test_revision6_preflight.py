"""The approved experiment must never run on failed or stale test evidence."""
import json
from types import SimpleNamespace
import pytest
from scripts import run_revision6_preflight as preflight


@pytest.mark.parametrize('failures,errors,skipped', [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
def test_failed_or_missing_tests_are_not_a_pass(tmp_path, failures, errors, skipped):
    path = tmp_path/'tests.xml'
    path.write_text(f'<testsuites><testsuite tests="5" failures="{failures}" '
                   f'errors="{errors}" skipped="{skipped}"/></testsuites>')
    with pytest.raises(RuntimeError):
        preflight.require_passing_report(path)
    path.write_text('<testsuites/>')
    with pytest.raises(RuntimeError):
        preflight.require_passing_report(path)


@pytest.mark.parametrize('failure', ['test_exit', 'source_change', 'none', 'smoke_exit'])
def test_only_passing_unchanged_source_launches_exact_command(tmp_path, monkeypatch, failure):
    out, smoke = tmp_path/'preflight', tmp_path/'smoke'
    out.mkdir()
    calls = []
    hashes = iter([{'source': 'a'}, {'source': 'b' if failure == 'source_change' else 'a'}])
    monkeypatch.setattr(preflight, 'source_hashes', lambda: next(hashes))
    def run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            (out/'tests.xml').write_text('<testsuites><testsuite tests="10" failures="0" errors="0" skipped="0"/></testsuites>')
            return SimpleNamespace(returncode=int(failure == 'test_exit'))
        assert cmd == preflight.APPROVED_COMMAND
        return SimpleNamespace(returncode=int(failure == 'smoke_exit'))
    monkeypatch.setattr(preflight.subprocess, 'run', run)
    if failure == 'none':
        preflight.run_tests_then_smoke(out, smoke)
        assert json.loads((smoke/'verified_sources.json').read_text()) == {'source': 'a'}
    else:
        with pytest.raises(RuntimeError):
            preflight.run_tests_then_smoke(out, smoke)
    assert len(calls) == (1 if failure in ('test_exit', 'source_change') else 2)
    assert smoke.exists() == (failure in ('none', 'smoke_exit'))
