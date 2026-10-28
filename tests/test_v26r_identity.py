"""Exact flags-off comparison to untouched v26 on the same local runtime.

Scope since revision 10: evaluation-only runs with a static Firm 1. Flags-off
RL *training* is no longer the v26 curriculum: Core splits the old dynamic
stage 40/60 into gap catch-up and profit-best-response opponents (four-stage
curriculum, see ``Core.__init__``). Training identity is therefore checked
revision to revision instead (``scripts/r14_identity_check.py``,
``scripts/r15_identity_check.py``).
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import pytest

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent


def expanded_output():
    return Path(os.environ.get('V26R_EXPANDED_IDENTITY_OUTPUT',
                               str(BASE/'runs/revision5_tests/identity_expanded')))


def run_identity(output, python=sys.executable, *, city='Boston', slug='boston', training_days=0,
                 validation_interval=None, extra_args=()):
    if training_days:
        raise ValueError('flags-off RL training is checked revision to revision, '
                         'not against the v26 snapshot (four-stage curriculum since revision 10)')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='paired-', dir=output))
    reference = work / 'reference'
    reference.mkdir()
    archive_path = ROOT / 'artifacts/v26_confirmatory/source_snapshot_scheduler_20261006.tar.gz'
    with tarfile.open(archive_path) as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if (not member.isfile() or relative.is_absolute() or '..' in relative.parts
                    or relative.name.startswith(('._', 'test_')) or relative.suffix != '.py'):
                continue
            if len(relative.parts) != 1 and relative.parts[0] != 'calibration_presets':
                continue
            target = reference.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('wb') as destination:
                shutil.copyfileobj(source, destination)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    common = ['--market',city,'--choice_mode','cognitive','--firm1_mode','RL' if training_days else 'static',
              '--firm2_mode','adaptive_best_response_aggressive','--run_experiment',
              '--credit_assignment_v25','--v26','--deterministic_experiment_seed',
              '--seed','2601','--experiment_seed','2601','--policy_objective','profit_maximization',
              '--pool','300','--threshold_profile_source','cached','--threshold_cache_path',
              str(ROOT/f'artifacts/v26_confirmatory/thresholds/{slug}/llm_numeric_primary.jsonl'),
              '--calibration_json',str(ROOT/f'data/processed/{slug}/calibration.json'),
              '--gpt_threshold_mode','llm_numeric','--train_timesteps',str(training_days),'--train_customers','20',
              '--train_steps_per_day','6','--eval_timesteps','36','--eval_customers','20',
              '--no-truncation_bootstrap']
    if validation_interval is not None:
        common += ['--checkpoint_validation_interval_days', str(validation_interval)]
    common += list(extra_args)
    results = {}
    training_hashes = {}
    for label, code in [('reference', reference), ('flags_off', BASE)]:
        destination = work / label / 'outputs'
        destination.mkdir(parents=True, exist_ok=True)
        # The CLI preserves relative paths but relocates absolute prefixes.
        prefix = os.path.relpath(destination/'run', code)
        with (output/(label+'.log')).open('w') as log:
            subprocess.run([python, str(code/'Core.py'), *common, '--report_prefix',prefix],
                           cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=900, check=True)
        diagnostics = destination/'run_evaluation_diagnostics.csv'
        if not diagnostics.is_file():
            raise AssertionError('Missing identity diagnostics: '+str(diagnostics))
        saved = output/(label+'_evaluation_diagnostics.csv')
        shutil.copy2(diagnostics, saved)
        results[label] = hashlib.sha256(saved.read_bytes()).hexdigest()
        if training_days:
            training_hashes[label] = hashlib.sha256((destination/'run_training_diagnostics.csv').read_bytes()).hexdigest()
    historical = BASE/'tests/reference/identity_boston_static_eval.csv'
    report = dict(passed=results['reference']==results['flags_off'] and (
                      not training_days or training_hashes['reference']==training_hashes['flags_off']), sha256=results,
                  training_sha256=training_hashes,
                  python=python, archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                  historical_fixture_exact_match=(historical.read_bytes()==
                      (output/'flags_off_evaluation_diagnostics.csv').read_bytes()),
                  scope=f'{city}: {training_days} training days and 36 evaluation blocks; same runtime, no tolerance or filtering',
                  validation_interval_days=validation_interval,
                  work=str(work))
    (output/'identity.json').write_text(json.dumps(report,indent=2))
    assert report['passed'], 'Flags-off v26r differs from untouched v26 in the same runtime'
    return report


def test_flags_off_bit_identical_evaluation():
    run_identity(os.environ.get('V26R_IDENTITY_OUTPUT', str(BASE/'runs/identity_check')))


@pytest.mark.parametrize('city,slug', [('New York City', 'nyc'), ('Chicago', 'chicago')])
def test_flags_off_other_city_evaluation(city, slug):
    run_identity(expanded_output()/slug, city=city, slug=slug)


def test_run_identity_refuses_training_comparisons():
    with pytest.raises(ValueError, match='revision to revision'):
        run_identity(Path(tempfile.mkdtemp()), training_days=24)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args()
    print(json.dumps(run_identity(args.output, args.python),indent=2))
