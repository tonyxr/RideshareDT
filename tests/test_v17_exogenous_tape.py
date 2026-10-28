"""V17 common-random-number tape and exogenous trace contract."""

from Core import Core
from v17_tape import ExogenousTape, TAPE_PURPOSES


def _core(*, tape: bool = True) -> Core:
    core = Core(
        "Chicago",
        seed=1708,
        firm1_mode="RL",
        firm2_mode="static",
        total_customers_pool=24,
        simulation_sample_cap=8,
        enable_driver_supply=False,
        openai_api_key="",
        action_contract="two_lever_v1",
        v16_crn_tape=tape,
    )
    core._refresh_profile_pool(rides_per_timestep=8)
    return core


def _action(core: Core, distance: float, time: float) -> int:
    return next(
        index
        for index, cell in core.firm1.action_to_two_lever_cell.items()
        if cell == (distance, time)
    )


def _exogenous_trace(rows):
    keys = (
        "DayOfWeek",
        "Weather",
        "Hour",
        "Airport",
        "Service",
        "TravelDistance",
        "Age",
        "Gender",
        "IncomeBracket",
        "Region",
        "PriceThreshold",
    )
    return [{key: row.get(key) for key in keys} for row in rows]


def _branch(core: Core, snapshot, action: int):
    core._restore_counterfactual_state(snapshot)
    core._set_v16_tape_position(3, 2)
    profiles = core._sample_profiles_from_pool(8, generalization_fraction=0.0)
    core.firm1.apply_action(action, core.market)
    result = core.simulate_batch(
        day_of_week=2,
        weather="clear",
        hour=12,
        customers_per_step=8,
        sampled_profiles=profiles,
        collect_rows=True,
        supply_reference_customers=8,
    )
    return result


def test_tape_addresses_are_repeatable_purpose_isolated_and_serializable():
    tape = ExogenousTape(root_seed=91, tape_id=2)
    first = tape.gen(4, 5, "trip").integers(0, 2**31, size=20).tolist()
    assert first == tape.gen(4, 5, "trip").integers(
        0, 2**31, size=20
    ).tolist()
    assert first != tape.gen(4, 5, "profiles").integers(
        0, 2**31, size=20
    ).tolist()
    assert first != tape.gen(4, 5, "trip", tape_id=3).integers(
        0, 2**31, size=20
    ).tolist()
    restored = ExogenousTape.from_state_dict(tape.state_dict())
    assert restored.fingerprint() == tape.fingerprint()
    assert set(TAPE_PURPOSES) == {
        "calendar",
        "hours",
        "weather",
        "requests",
        "profiles",
        "trip",
        "shock",
        "endo",
    }


def test_same_cell_replays_identical_full_trace():
    core = _core()
    core._set_v16_tape_position(3, 2)
    snapshot = core._snapshot_counterfactual_state()
    action = _action(core, 1.0, 1.0)
    first = _branch(core, snapshot, action)
    second = _branch(core, snapshot, action)
    assert first == second


def test_day_request_and_profile_tapes_replay_from_restored_state():
    core = _core()
    core._set_v16_tape_position(1, 0)
    snapshot = core._snapshot_counterfactual_state()

    def draw():
        core._restore_counterfactual_state(snapshot)
        day, hours = core._sample_v16_day_and_hours(6, 6)
        core._set_v16_tape_position(6, 2)
        requests = core._step_customers(
            8,
            hour=hours[2],
            day_of_week=day.day_of_week,
            weather=day.weather,
        )
        profiles = core._sample_profiles_from_pool(
            8, generalization_fraction=0.0
        )
        profile_trace = [
            (
                profile.get("Age"),
                profile.get("Gender"),
                profile.get("IncomeBracket"),
                profile.get("Region"),
                profile.get("PriceThreshold"),
            )
            for profile in profiles
        ]
        return day, hours, requests, profile_trace

    assert draw() == draw()


def test_different_cells_share_requests_profiles_and_trip_attributes():
    core = _core()
    core._set_v16_tape_position(3, 2)
    snapshot = core._snapshot_counterfactual_state()
    low = _branch(core, snapshot, _action(core, 0.8, 0.8))
    high = _branch(core, snapshot, _action(core, 1.5, 1.5))
    assert _exogenous_trace(low[0]) == _exogenous_trace(high[0])
    assert [row["Price_Firm1"] for row in low[0]] != [
        row["Price_Firm1"] for row in high[0]
    ]


def test_endogenous_stream_realigns_at_next_step_after_conditional_draws():
    core = _core()
    core._set_v16_tape_position(7, 0)
    first = core._v16_tape_rng("endo")
    first.random(500)
    core._set_v16_tape_position(7, 1)
    after_long_branch = core._v16_tape_rng("endo").random(20).tolist()
    core._set_v16_tape_position(7, 0)
    second = core._v16_tape_rng("endo")
    second.random(1)
    core._set_v16_tape_position(7, 1)
    after_short_branch = core._v16_tape_rng("endo").random(20).tolist()
    assert after_long_branch == after_short_branch


def test_counterfactual_restore_includes_tape_position():
    core = _core()
    core._set_v16_tape_position(8, 4, tape_id=1)
    snapshot = core._snapshot_counterfactual_state()
    core._set_v16_tape_position(9, 5, tape_id=2)
    core._restore_counterfactual_state(snapshot)
    assert core._v16_tape_position == (8, 4, 1)


def test_flag_defaults_off_without_allocating_a_tape():
    core = _core(tape=False)
    assert not core.v16_crn_tape
    assert core.exogenous_tape is None
    assert core._v16_tape_position is None

