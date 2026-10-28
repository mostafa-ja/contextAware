import pytest
from src.context_aware.context_builder import ContextBuilder
from src.context_aware.weather_service import create_simulated_weather
from src.context_aware.config import FEATURE_NAMES


def test_context_builder_dimensions_and_bounds():
    builder = ContextBuilder()
    weather_data, sr, ss = create_simulated_weather(temperature=22.0, humidity=50.0, weather_code=0)

    day_info = {
        "day_type": "day_workday",
        "persian_month": 2,
        "persian_day": 10,
        "is_holiday": False,
        "events": [],
    }

    ctx = builder.build(weather_data, sr, ss, day_info)

    # Must contain all 65 features
    for feat in FEATURE_NAMES:
        assert feat in ctx, f"Missing feature: {feat}"
        assert 0.0 <= ctx[feat] <= 1.0, f"Out of bounds: {feat}={ctx[feat]}"

    assert len(ctx) == len(FEATURE_NAMES)


def test_context_builder_yalda_enrichment():
    builder = ContextBuilder()
    weather_data, sr, ss = create_simulated_weather(temperature=5.0, weather_code=3)

    day_info = {
        "day_type": "day_workday",
        "persian_month": 9,
        "persian_day": 30,
        "is_holiday": False,
        "events": ["یلدا"],
    }

    ctx = builder.build(weather_data, sr, ss, day_info)

    # Yalda should boost cultural tradition, family social, and nostalgic mood
    assert ctx["cultural_tradition"] == 1.0
    assert ctx["social_family"] > 0.40
    assert ctx["mood_nostalgic"] > 0.40
    assert ctx["location_home"] > 0.30


def test_context_builder_bad_weather_location():
    builder = ContextBuilder()
    # Heavy rain (code 65) with high wind
    weather_data, sr, ss = create_simulated_weather(temperature=8.0, weather_code=65, wind_speed=35.0)

    ctx = builder.build(weather_data, sr, ss)

    # Indoor and home should dominate outdoor
    assert ctx["location_indoor"] > ctx["location_outdoor"]
    assert ctx["location_home"] > ctx["location_outdoor"]
