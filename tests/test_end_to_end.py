import os
import pytest
from src.context_aware.engine import SuggestionEngine
from src.context_aware.selector import SmartSelector
from src.context_aware.context_builder import ContextBuilder
from src.context_aware.weather_service import create_simulated_weather
from src.context_aware.evaluation import (
    evaluate_bundle_diversity,
    benchmark_inference_latency,
    simulate_freshness_decay_curve,
)


@pytest.fixture(scope="module")
def loaded_engine():
    engine = SuggestionEngine(state_file="scratch_e2e_freshness.json")
    dataset_dir = "dataset"
    count = engine.load_data(dataset_dir)
    assert count > 1000, f"Expected >1000 items in dataset, got {count}"
    yield engine
    if os.path.exists("scratch_e2e_freshness.json"):
        os.remove("scratch_e2e_freshness.json")


def test_end_to_end_real_dataset_bundle_generation(loaded_engine):
    builder = ContextBuilder()
    selector = SmartSelector(loaded_engine)

    # Rainy autumn evening in Tehran (Mehr 20)
    w_data, sr, ss = create_simulated_weather(
        temperature=14.0,
        humidity=75.0,
        weather_code=61,  # Rain
        wind_speed=12.0,
    )

    day_info = {
        "day_type": "day_workday",
        "persian_month": 7,
        "persian_day": 20,
        "is_holiday": False,
        "events": [],
    }

    ctx = builder.build(w_data, sr, ss, day_info)
    assert ctx["weather_rain"] > 0.50
    assert ctx["season_autumn"] > 0.50

    bundle = selector.generate_bundle(ctx, day_info, custom_hour=19.50)
    assert bundle is not None
    assert "main" in bundle
    assert bundle["main"]["score"] > 0.0
    assert "sidekicks" in bundle

    diversity = evaluate_bundle_diversity(bundle)
    assert 0.0 <= diversity <= 1.0


def test_end_to_end_latency_benchmark(loaded_engine):
    builder = ContextBuilder()
    w_data, sr, ss = create_simulated_weather(temperature=20.0, weather_code=0)
    day_info = {"day_type": "day_workday", "persian_month": 1, "persian_day": 1, "is_holiday": False, "events": []}
    ctx = builder.build(w_data, sr, ss, day_info)

    perf = benchmark_inference_latency(loaded_engine, ctx, day_info, iterations=25)
    assert perf["mean_ms"] < 25.0, f"Mean latency too high: {perf['mean_ms']} ms"
    assert perf["candidate_pool_size"] > 1000
