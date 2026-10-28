"""Explicitly approved, sequential three-city smoke; no retry or promotion."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from diagnostic_logging import atomic_json
from scripts.audit_city_calibration import CITIES, audit, load_core
from scripts.run_v26r_staged import command, environment, source_hashes, summarize
from city_supply_calibration import calibrate_capacity


def training_command(out, slug, calibration):
    return command(out, days=2400, city=CITIES[slug], slug=slug,
        update_days=1, rollout_transitions=30) + [
        '--learning_corrections', '--learner_revision', '4',
        '--ppo_rollout_episodes', '2', '--no-live_diagnostics',
        '--city_calibration_artifact', str(calibration)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    verified = json.loads((out/'verified_sources.json').read_text())
    if source_hashes() != verified:
        raise RuntimeError('Source differs from the tested source; experiment not started')
    with (out/'started.json').open('x') as f:
        json.dump(dict(pid=os.getpid(), started=datetime.datetime.now().isoformat()), f)
    def status(value):
        (out/'status.txt').write_text(value+'\n')
        print(value, flush=True)
    try:
        reports = {}
        for slug in CITIES:
            status(f'{slug}: real 50-request capacity calibration, 30 days / 15 burn-in')
            core = load_core(slug, out/f'calibration_setup_{slug}')
            checks = {}
            for volume in (50, 300):
                checks[str(volume)] = calibrate_capacity(core, sample_size=volume,
                    days=45, burnin_days=15, measure_reference=False, verify_only=True)
            atomic_json(out/f'{slug}_operating_point_checks.json', checks)
            status(f'{slug}: validation/held-out constants and elasticity (3 openings, 40 days, 15 burn-in)')
            audit(slug, out/'calibration', openings=3, horizon_days=40, burnin_days=15, core=core)
            del core
            city_out = out/slug
            city_out.mkdir()
            cmd = training_command(city_out, slug, out/'calibration'/slug/'city_calibration.json')
            atomic_json(city_out/'command.json', cmd)
            status(f'{slug}: staged 2400-day smoke, then final-policy standard benchmarks')
            with (city_out/'run.log').open('x') as log:
                proc = subprocess.Popen(cmd, cwd=BASE, env=environment(), stdout=log, stderr=subprocess.STDOUT)
                (city_out/'pid.txt').write_text(str(proc.pid)+'\n')
                code = proc.wait()
            (city_out/'exit_status.txt').write_text(str(code)+'\n')
            if code:
                raise RuntimeError(f'{slug} exited {code}; no automatic retry')
            reports[slug] = summarize(city_out, expected_days=2400, slug=slug)
            atomic_json(out/'results.json', reports)
        if source_hashes() != verified:
            raise RuntimeError('Source changed during the smoke; provenance requires review')
        status('completed; results include every losing case, no performance-based promotion')
        (out/'exit_status.txt').write_text('0\n')
    except BaseException:
        traceback.print_exc()
        status('failed; evidence preserved, no retries or policy replacement')
        (out/'exit_status.txt').write_text('1\n')
        raise


if __name__ == '__main__':
    main()
