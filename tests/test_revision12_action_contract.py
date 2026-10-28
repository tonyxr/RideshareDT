from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pricing_models import FirmRLPricer  # noqa: E402
from run_revision12_verify import (  # noqa: E402
    GRID,
    assert_independent_action_contract,
    train_cmd,
)


def test_revision12_launcher_uses_individual_coefficient_magnitude_actions(tmp_path):
    command = train_cmd(tmp_path, days=192, oracle=True, smoke=True)
    assert_independent_action_contract(command)
    assert "--whole_tariff_actions" not in command
    assert command[command.index("--manipulated_price_coefficients") + 1] == "5"


def test_revision12_action_space_has_five_targets_and_thirteen_magnitudes():
    targets = tuple(float(value) for value in GRID.split(","))
    pricer = FirmRLPricer(
        seed=7,
        opt_keys=[
            "base_fare",
            "per_minute",
            "per_mile",
            "booking_fee",
            "airport_fee",
        ],
        tariff_targets=targets,
        independent_coefficient_actions=True,
    )

    assert pricer.agent.action_dim == 1 + 5 * 13
    assert pricer.action_label(0) == "hold"
    assert {
        pricer.action_to_target_key[action]
        for action in range(1, pricer.agent.action_dim)
    } == {
        "base_fare",
        "per_minute",
        "per_mile",
        "booking_fee",
        "airport_fee",
    }
    assert set(pricer.action_to_multiplier.values()) == set(targets)
