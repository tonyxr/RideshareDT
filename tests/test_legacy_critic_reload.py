"""Reject an older policy when its reward/action contract is incompatible."""
import hashlib
import subprocess

from scripts.run_v26r_staged import BASE, command, environment


def test_pre_repair_saved_policy_is_not_reinterpreted_under_new_reward(tmp_path):
    source = BASE.parent / 'rideshare_v26r_corrected/runs/competitive_retraining_tests_20261016/weighted_final_fixtures/test_corrected_stages_learning0/corrected/policy.final_training.pt'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    out = tmp_path / 'legacy_reload'
    out.mkdir()
    cmd = command(out, days=0, integration=True, lead_weight=.6) + [
        '--trained_model_in', str(source), '--eval_only']
    # Evaluation uses the saved grid; deliberately requesting today's wider
    # grid is a different action space, not backward-compatible inference.
    index = cmd.index('--tariff_targets')
    del cmd[index:index + 2]
    # The CLI must inherit the saved learner contract; an explicit
    # --learner_revision is rejected before the reward check is reached.
    cmd.remove('--learning_corrections')
    index = cmd.index('--learner_revision')
    del cmd[index:index + 2]
    result = subprocess.run(
        cmd, cwd=BASE, env=environment(), capture_output=True, text=True,
        timeout=180,
    )
    assert result.returncode != 0
    assert 'reward mechanism does not match runtime' in result.stderr
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
