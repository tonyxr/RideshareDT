"""Capacity calibration under the installed v2/v3 model, never under a scored policy."""
import copy
import numpy as np
from coeff_utils import CoefficientOverrides


def calibrate_capacity(core, *, sample_size=50, days=30, burnin_days=15, iterations=9,
                       measure_reference=True, verify_only=False):
    target = float(core.market_dynamics.capacity_target_fulfillment)
    if not (core.market_dynamics.enabled and core.enable_driver_supply and target > 0):
        return {"enabled": False, "base_active_drivers": core.driver_supply.config.base_active_drivers}
    if not (core.driver_supply.supply_v2 or core.driver_supply.supply_v3):
        raise ValueError("capacity calibration requires the installed v2 or v3 supply model")
    if days <= burnin_days:
        raise ValueError("calibration must include post-burn-in days")
    if not core.training_profile_pool:
        core._refresh_profile_pool(rides_per_timestep=sample_size)
    snapshot = core._snapshot_counterfactual_state()
    old_regime = core.regime_market
    history = []
    steps = int(core.shared_steps_per_day)
    hours = [int(24 * (block + .5) / steps) for block in range(steps)]
    reference = float(core.driver_supply.config.reference_earnings_per_hour)
    if reference <= 0:
        raise ValueError("capacity calibration requires a fixed positive reference earnings rate")

    def evaluate(drivers):
        core._restore_counterfactual_state(snapshot)
        core.regime_market = False
        core.market.install_shocks(None)
        core.driver_supply.apply_shock()
        core.driver_supply.config.base_active_drivers = int(drivers)
        if not verify_only:
            core.driver_supply.config.initial_online_stock = None
        core.driver_supply.reset_stock()
        core.firm1.overrides = CoefficientOverrides()
        core.firm2.overrides = CoefficientOverrides()
        core._reset_rider_memory()
        if core.loyalty_v2:
            core._build_active_rider_pool(np.random.default_rng(260051))
        completed = offers = online_minutes = pay = 0.
        calibration_days = []
        stocks = []
        for day in range(days):
            calibration_pay = calibration_online = 0.
            for block, hour in enumerate(hours):
                # Re-key every block: conditional dispatch draws cannot shift
                # later candidates' rider/trip tapes during capacity search.
                seed = 260052 + day * steps + block
                core.rng = np.random.default_rng([seed, 0])
                core.choice_model.rng = np.random.default_rng([seed, 1])
                core.driver_supply.rng = np.random.default_rng([seed, 2])
                n = core._step_customers(sample_size, hour=hour, day_of_week=day % 7, weather='clear')
                profiles = core._sample_profiles_from_pool(n, generalization_fraction=0.)
                _, a, b, _, _, _ = core.simulate_batch(day_of_week=day % 7, hour=hour,
                    weather='clear', customers_per_step=n, sampled_profiles=profiles,
                    collect_rows=False, supply_reference_customers=sample_size)
                if day < burnin_days:
                    for state in core.driver_supply.last_states.values():
                        calibration_pay += state.total_driver_pay
                        calibration_online += state.total_online_minutes
                if day >= burnin_days:
                    completed += a.completed + b.completed
                    offers += n * (1 - core.last_crowd_response_stats['no_ride_rate'])
                    for state in core.driver_supply.last_states.values():
                        pay += state.total_driver_pay
                        online_minutes += state.total_online_minutes
            core.driver_supply.advance_day()
            if day < burnin_days and measure_reference:
                # Reference is measured only on this designated calm 1x
                # calibration arm, never learned during training/evaluation.
                calibration_days.append((calibration_pay, calibration_online))
                recent = calibration_days[-7:]
                measured = 60. * sum(p for p, _ in recent) / max(1., sum(n for _, n in recent))
                if measured <= 0:
                    raise ValueError('calm calibration generated no positive driver earnings')
                core.driver_supply.reference_earnings = {f: measured for f in ('Firm1', 'Firm2')}
            elif day >= burnin_days:
                stocks.append(dict(core.driver_supply.online_stock))
        row = dict(drivers=int(drivers), fulfillment=completed / max(1., offers),
                   gross_earnings_per_online_hour=60 * pay / max(1., online_minutes),
                   reference_earnings=core.driver_supply.reference_earnings['Firm1'],
                   opening_stock={f: float(np.mean([s[f] for s in stocks[-min(7, len(stocks)):]]))
                                  for f in ('Firm1', 'Firm2')},
                   final_stock=dict(core.driver_supply.online_stock))
        history.append(row)
        print(f'[capacity-{core.driver_supply.config.supply_model}] {core.market_name}: {row}', flush=True)
        return row

    try:
        original = int(core.driver_supply.config.base_active_drivers)
        original_row = evaluate(original)
        if verify_only:
            return dict(original_row, sample_size=sample_size, days=days, burnin_days=burnin_days,
                        installed_reference=reference, measurement_only=True)
        low, high = 20, 4000
        maximum = evaluate(high)
        if maximum['fulfillment'] < target:
            raise RuntimeError(f"{core.driver_supply.config.supply_model} capacity target unreachable: {maximum}; target={target}. "
                               "No substitute fleet or lower target was installed.")
        for _ in range(iterations):
            middle = round(np.sqrt(low * high))
            if evaluate(middle)['fulfillment'] >= target:
                high = middle
            else:
                low = middle
        selected = evaluate(high)
    finally:
        core._restore_counterfactual_state(snapshot)
        core.regime_market = old_regime
    core.driver_supply.config.base_active_drivers = high
    core.driver_supply.config.reference_earnings_per_hour = selected['reference_earnings']
    core.driver_supply.config.initial_online_stock = selected['opening_stock']
    core.driver_supply.reset_stock()
    return dict(enabled=True, supply_model=core.driver_supply.config.supply_model,
        reference_earnings=selected['reference_earnings'],
        reference_contract=('measured_calm_1x_training_calibration_then_fixed' if measure_reference
                            else 'explicit_reference'), target_fulfillment=target,
        initial_online_stock=selected['opening_stock'], final_online_stock=selected['final_stock'],
        prior_reference_earnings=reference,
        base_active_drivers=high, anchor_fulfillment=selected['fulfillment'],
        anchor_gross_earnings_per_online_hour=selected['gross_earnings_per_online_hour'],
        previous_fleet_measurement=original_row, sample_size=sample_size,
        days=days, burnin_days=burnin_days, candidates=history,
        anchor_policy='both firms at 1x, calm, training rider partition',
        reference_customers=core.driver_supply.config.reference_customers)
