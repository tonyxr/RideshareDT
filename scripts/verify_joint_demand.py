"""One independent verification run; no training promotion or automatic retry."""
import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'pid.txt').write_text(str(os.getpid())+'\n')
    before = source_hashes()
    atomic_json(out/'sources.json', before)
    env = environment()
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', MPLCONFIGDIR=str(out/'mpl'),
        V26R_IDENTITY_OUTPUT=str(out/'identity/boston'),
        V26R_EXPANDED_IDENTITY_OUTPUT=str(out/'identity/expanded'))
    try:
        (out/'status.txt').write_text('regression and flags-off identity tests\n')
        args = ['-q', 'tests', '--tb=short', '--basetemp='+str(out/'fixtures'), '--junitxml='+str(out/'tests.xml')]
        bootstrap = ('import sys; sys.path.append("/Users/huali/opt/anaconda3/lib/python3.9/site-packages"); '
                     'import pytest; raise SystemExit(pytest.main('+repr(args)+'))')
        cmd = [str(BASE.parent/'.venv/bin/python'), '-u', '-c', bootstrap]
        atomic_json(out/'test_command.json', cmd)
        with (out/'tests.log').open('x') as log:
            result = subprocess.run(cmd, cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT)
        (out/'test_exit_status.txt').write_text(str(result.returncode)+'\n')
        if result.returncode:
            raise RuntimeError('Required tests failed; calibration not started')
        suites = ET.parse(out/'tests.xml').getroot()
        totals = {k: sum(int(s.attrib.get(k, 0)) for s in suites.iter('testsuite'))
                  for k in ('tests', 'failures', 'errors', 'skipped')}
        atomic_json(out/'test_result.json', totals)
        if not totals['tests'] or any(totals[k] for k in ('failures', 'errors', 'skipped')):
            raise RuntimeError('Missing or incomplete passing test evidence')
        if source_hashes() != before:
            raise RuntimeError('Source changed during verification')
        (out/'status.txt').write_text('tests passed; three-city demand-only prior sensitivity audit\n')
        cmd = [str(BASE.parent/'.venv/bin/python'), '-u', str(BASE/'scripts/audit_joint_demand.py'),
               '--elasticities', '-0.9', '-1.15', '-1.4', '--records', '2000', '--output', str(out/'audit')]
        atomic_json(out/'audit_command.json', cmd)
        with (out/'audit.log').open('x') as log:
            result = subprocess.run(cmd, cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Demand audit failed; no retry')
        if source_hashes() != before:
            raise RuntimeError('Source changed during the audit')
        (out/'status.txt').write_text('completed; no RL training or capacity refitting performed\n')
        (out/'exit_status.txt').write_text('0\n')
    except BaseException as error:
        traceback.print_exc()
        (out/'status.txt').write_text('failed: '+str(error)+'\n')
        (out/'exit_status.txt').write_text('1\n')
        raise


if __name__ == '__main__':
    main()
