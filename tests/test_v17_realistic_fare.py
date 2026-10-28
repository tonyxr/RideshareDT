import numpy as np
import pytest

from MarketInteraction import MarketInteraction, RideContext
from Market_models import CoefficientOverrides


def _legacy_formula(market, distance, duration, ctx, override, extra):
    card = market.curr_market
    base = override.base_fare + override.booking_fee
    variable = override.per_mile * distance + override.per_minute * duration
    context = (
        market._time_multiplier(ctx.hour)
        * override.day_multiplier.get(ctx.day_of_week, 1.0)
        * override.weather_multiplier.get(ctx.weather, 1.0)
        * override.service_multiplier.get(ctx.service, 1.0)
    )
    airport = override.airport_fee if ctx.airport else 0.0
    assert card is not None
    return round(max((base + variable) * context + airport + extra, 0.0), 2)


@pytest.fixture
def override():
    return CoefficientOverrides(
        base_fare=3.0,
        per_mile=1.7,
        per_minute=0.31,
        booking_fee=2.2,
        airport_fee=4.75,
        day_multiplier={day: 1.0 + 0.02 * day for day in range(7)},
        weather_multiplier={"clear": 1.0, "rain": 1.15, "snow": 1.3},
        service_multiplier={"economy": 1.0, "premium": 1.4},
    )


def test_legacy_fare_is_unchanged_for_1000_trips(override):
    market = MarketInteraction("New York City", seed=31)
    rng = np.random.default_rng(991)
    for _ in range(1000):
        ctx = RideContext(
            day_of_week=int(rng.integers(0, 7)),
            weather=str(rng.choice(["clear", "rain", "snow"])),
            hour=int(rng.integers(0, 24)),
            airport=bool(rng.integers(0, 2)),
            service=str(rng.choice(["economy", "premium"])),
        )
        distance = float(rng.uniform(0.1, 30.0))
        duration = float(rng.uniform(1.0, 120.0))
        extra = float(rng.choice([0.0, 1.25, 3.50]))
        expected = _legacy_formula(market, distance, duration, ctx, override, extra)
        quote = market.quote_components(distance, duration, ctx, override, extra)
        assert quote.rider_fare == expected
        assert quote.platform_fare == expected
        assert quote.pass_through == 0.0
        assert market.quote_price(distance, duration, ctx, override, extra) == expected


def test_realistic_fare_and_pass_through_for_1000_trips(override):
    market = MarketInteraction("New York City", seed=31)
    market.configure_fare_model("realistic")
    rng = np.random.default_rng(1771)
    for _ in range(1000):
        ctx = RideContext(
            day_of_week=int(rng.integers(0, 7)),
            weather=str(rng.choice(["clear", "rain", "snow"])),
            hour=int(rng.integers(0, 24)),
            airport=bool(rng.integers(0, 2)),
            service=str(rng.choice(["economy", "premium"])),
        )
        distance = float(rng.uniform(0.1, 30.0))
        duration = float(rng.uniform(1.0, 120.0))
        extra = float(rng.choice([0.0, 1.25, 3.50]))
        context = (
            market._time_multiplier(ctx.hour)
            * override.day_multiplier[ctx.day_of_week]
            * override.weather_multiplier[ctx.weather]
            * override.service_multiplier[ctx.service]
        )
        platform_unrounded = (
            override.base_fare
            + override.booking_fee
            + (override.per_mile * distance + override.per_minute * duration)
            * context
        )
        pass_through = (override.airport_fee if ctx.airport else 0.0) + extra
        rider = round(platform_unrounded + pass_through, 2)
        expected_pass = round(pass_through, 2)
        quote = market.quote_components(distance, duration, ctx, override, extra)
        assert quote.rider_fare == rider
        assert quote.pass_through == expected_pass
        assert quote.platform_fare == round(rider - expected_pass, 2)
        assert quote.rider_fare == pytest.approx(
            quote.platform_fare + quote.pass_through
        )


def test_bad_fare_model_is_rejected():
    market = MarketInteraction("Boston", seed=1)
    with pytest.raises(ValueError, match="fare_model"):
        market.configure_fare_model("real-ish")


def test_core_flag_wires_realistic_model_without_changing_default():
    from Core import Core

    legacy = Core(
        "Boston",
        seed=7,
        firm1_mode="static",
        firm2_mode="static",
        total_customers_pool=8,
        simulation_sample_cap=8,
        enable_driver_supply=False,
        openai_api_key="",
    )
    realistic = Core(
        "Boston",
        seed=7,
        firm1_mode="static",
        firm2_mode="static",
        total_customers_pool=8,
        simulation_sample_cap=8,
        enable_driver_supply=False,
        openai_api_key="",
        fare_model="realistic",
    )
    assert legacy.fare_model == "legacy"
    assert legacy.market.fare_model == "legacy"
    assert realistic.fare_model == "realistic"
    assert realistic.market.fare_model == "realistic"
