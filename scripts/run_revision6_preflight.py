"""One-shot tests followed by the approved smoke command; never retries."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from diagnostic_logging import atomic_json
from scripts.run_v26r_staged import environment, source_hashes

OUT = BASE.parent / 'artifacts/revision6_preflight_20261018'
SMOKE = BASE.parent / 'artifacts/revision6_smoke_20261018'
APPROVED_COMMAND = ['/bin/zsh', str(BASE/'scripts/start_revision6_smoke.command')]


def require_passing_report(path):
    suites = ET.parse(path).getroot()
    rows = list(suites.iter('testsuite'))
    totals = {key: sum(int(row.attrib.get(key, 0)) for row in rows)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    if not totals['tests'] or any(totals[key] for key in ('failures', 'errors', 'skipped')):
        raise RuntimeError(f'Required tests did not all pass: {totals}')
    return totals


def run_tests_then_smoke(out, smoke):
    before = source_hashes()
    atomic_json(out/'tested_sources.json', before)
    env = environment()
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        MPLCONFIGDIR=str(out/'mplconfig'),
        V26R_IDENTITY_OUTPUT=str(out/'identity/boston'),
        V26R_EXPANDED_IDENTITY_OUTPUT=str(out/'identity/expanded'))
    args = ['-q', 'tests', '--basetemp='+str(out/'fixtures'),
            '--junitxml='+str(out/'tests.xml')]
    bootstrap = ('import sys; '
        'sys.path.append("/Users/huali/opt/anaconda3/lib/python3.9/site-packages"); '
        'import pytest; raise SystemExit(pytest.main('+repr(args)+'))')
    cmd = [str(BASE.parent/'.venv/bin/python'), '-u', '-c', bootstrap]
    atomic_json(out/'test_command.json', cmd)
    (out/'status.txt').write_text('testing: complete regression, integration and flags-off identity suite\n')
    with (out/'tests.log').open('x') as log:
        result = subprocess.run(cmd, cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT)
    (out/'test_exit_status.txt').write_text(str(result.returncode)+'\n')
    if result.returncode:
        raise RuntimeError(f'Tests exited {result.returncode}; smoke NOT launched')
    totals = require_passing_report(out/'tests.xml')
    if before != source_hashes():
        raise RuntimeError('Source changed during tests; smoke NOT launched')
    smoke.mkdir(exist_ok=False)
    atomic_json(smoke/'verified_sources.json', before)
    atomic_json(out/'test_result.json', totals)
    atomic_json(out/'smoke_command.json', APPROVED_COMMAND)
    (out/'status.txt').write_text('tests passed; running approved three-city smoke\n')
    result = subprocess.run(APPROVED_COMMAND, cwd=BASE, env=env)
    if result.returncode:
        raise RuntimeError(f'Smoke exited {result.returncode}; no retry')
    (out/'status.txt').write_text('completed: tests and smoke\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'started.json', dict(pid=os.getpid(), started=datetime.datetime.now().isoformat(),
        scope='One sequential test suite, then previously approved smoke; no performance gates or retries'))
    try:
        run_tests_then_smoke(out, SMOKE)
        (out/'exit_status.txt').write_text('0\n')
    except BaseException as error:
        traceback.print_exc()
        (out/'status.txt').write_text('failed: '+str(error)+'\n')
        (out/'exit_status.txt').write_text('1\n')
        raise


if __name__ == '__main__':
    main()
