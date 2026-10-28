"""Policy-independent batch and rider streams; legacy flags-off path is untouched."""
from functools import wraps
import numpy as np


def isolated_batch_noise(function):
    @wraps(function)
    def wrapped(core, *args, **kwargs):
        tape_enabled = bool(
            getattr(core, "v16_crn_tape", False)
            and getattr(core, "_v16_tape_position", None) is not None
        )
        if not tape_enabled and not core.driver_supply.supply_v2:
            return function(core, *args, **kwargs)
        owners = (core, core.choice_model, core.driver_supply)
        streams = [owner.rng for owner in owners]
        if tape_enabled:
            root = core._v16_tape_rng("endo")
            keys = [int(value) for value in root.integers(0, 2**63 - 1, size=3)]
        else:
            keys = [int(rng.integers(0, 2**63-1)) for rng in streams]
        previous = getattr(core, '_batch_noise_keys', None)
        core._batch_noise_keys = keys
        try:
            for owner, key in zip(owners, keys):
                owner.rng = np.random.default_rng([key, 0])
            return function(core, *args, **kwargs)
        finally:
            for owner, rng in zip(owners, streams):
                owner.rng = rng
            core._batch_noise_keys = previous
    return wrapped


def rider_streams(core, index):
    keys = getattr(core, '_batch_noise_keys', None)
    if keys is not None:
        for owner, key in zip((core, core.choice_model, core.driver_supply), keys):
            owner.rng = np.random.default_rng([key, int(index)+1])
