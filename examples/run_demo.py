"""
ContextAware: Interactive Demonstration Script
==============================================
Runs live weather queries or pre-configured cultural/environmental
scenarios to showcase the fuzzy context builder and smart bundle selector.
"""

import os
import sys
import argparse
import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.context_aware.engine import SuggestionEngine
from src.context_aware.context_builder import ContextBuilder
from src.context_aware.selector import SmartSelector
from src.context_aware.weather_service import fetch_weather, create_simulated_weather
from src.context_aware.evaluation import evaluate_bundle_diversity


SCENARIOS = {
    "1": {
        "title": "❄️ Winter Solstice (Shab-e Yalda) - Tehran, Freezing Cold Night",
        "weather": {"temperature": 1.5, "humidity": 70.0, "weather_code": 3, "wind_speed": 4.0},
        "day_info": {
            "day_type": "day_holiday_eve",
            "persian_month": 9,
            "persian_day": 30,
            "is_holiday": False,
            "events": ["یلدا"],
        },
        "hour": 20.5,
    },
    "2": {
        "title": "🌸 Norooz & Sizdah Bedar - Sunny Spring Picnic in Nature",
        "weather": {"temperature": 21.0, "humidity": 35.0, "weather_code": 0, "wind_speed": 7.0},
        "day_info": {
            "day_type": "day_holiday",
            "persian_month": 1,
            "persian_day": 13,
            "is_holiday": True,
            "events": ["سیزدهبدر", "نوروز"],
        },
        "hour": 13.0,
    },
    "3": {
        "title": "🌧️ Autumn Rain & Cozy Evening - Workday Wind-down",
        "weather": {"temperature": 12.0, "humidity": 85.0, "weather_code": 61, "wind_speed": 15.0},
        "day_info": {
            "day_type": "day_workday",
            "persian_month": 8,
            "persian_day": 18,
            "is_holiday": False,
            "events": [],
        },
        "hour": 19.0,
    },
    "4": {
        "title": "🔥 Scorching Summer Afternoon - Midday Heatwave",
        "weather": {"temperature": 39.5, "humidity": 18.0, "weather_code": 0, "wind_speed": 10.0},
        "day_info": {
            "day_type": "day_workday",
            "persian_month": 5,
            "persian_day": 12,
            "is_holiday": False,
            "events": [],
        },
        "hour": 14.5,
    },
}


def run_scenario(scenario_key: str, dataset_dir: str = "dataset"):
    sc = SCENARIOS.get(scenario_key)
    if not sc:
        print(f"Unknown scenario key: {scenario_key}")
        return

    print("=" * 80)
    print(f"🎬 SCENARIO: {sc['title']}")
    print("=" * 80)

    engine = SuggestionEngine(state_file="freshness_state.json")
    loaded_count = engine.load_data(dataset_dir)
    print(f"📦 Loaded {loaded_count} suggestions from '{dataset_dir}'.")

    builder = ContextBuilder()
    selector = SmartSelector(engine)

    w_data, sr, ss = create_simulated_weather(**sc["weather"])
    ctx = builder.build(w_data, sr, ss, sc["day_info"])

    print("\n📊 Top Active Context Features:")
    sorted_features = sorted([(k, v) for k, v in ctx.items() if v > 0.30], key=lambda x: x[1], reverse=True)
    for feat, val in sorted_features[:8]:
        bar = "█" * int(val * 20)
        print(f"  {feat:25} : {val:4.2f} | {bar}")

    bundle = selector.generate_bundle(ctx, sc["day_info"], custom_hour=sc["hour"])
    if not bundle:
        print("\n⚠️ No recommendation could be synthesized.")
        return

    diversity = evaluate_bundle_diversity(bundle)

    print("\n🎁 SYNTHESIZED AMBIENT RECOMMENDATION BUNDLE:")
    print("-" * 80)
    hero = bundle["main"]
    print(f"🌟 HERO [{hero['score']:4.2f}] ({hero['item'].get('subcategory')}): {hero['item'].get('text')}")

    for i, sk in enumerate(bundle.get("sidekicks", []), 1):
        print(f"  ↳ Sidekick {i} [{sk['score']:4.2f}] ({sk['item'].get('subcategory')}): {sk['item'].get('text')}")

    print(f"\n✨ Intra-Bundle Diversity Index: {diversity:.3f}")
    print("=" * 80)


def run_live(latitude: float, longitude: float, dataset_dir: str = "dataset"):
    print("=" * 80)
    print(f"🌍 FETCHING LIVE OPEN-METEO WEATHER (Lat: {latitude}, Lon: {longitude})")
    print("=" * 80)

    weather_data, sunrise_h, sunset_h = fetch_weather(latitude, longitude)
    curr = weather_data.get("current", {})
    temp = curr.get("temperature_2m", 20.0)
    code = curr.get("weather_code", 0)
    humidity = curr.get("relative_humidity_2m", 50.0)

    print(f"🌡️ Temperature: {temp}°C | Humidity: {humidity}% | WMO Code: {code}")
    print(f"☀️ Sunrise: {sunrise_h:.2f}h | Sunset: {sunset_h:.2f}h")

    engine = SuggestionEngine(state_file="freshness_state.json")
    loaded_count = engine.load_data(dataset_dir)
    print(f"📦 Loaded {loaded_count} suggestions from '{dataset_dir}'.")

    # Iranian calendar fallback inference
    now = datetime.datetime.now()
    # Simple approximate Persian month
    p_month = ((now.month - 3) % 12) + 1
    p_day = now.day
    day_info = {
        "day_type": "day_weekend" if now.weekday() == 4 else "day_workday",
        "persian_month": p_month,
        "persian_day": p_day,
        "is_holiday": False,
        "events": [],
    }

    builder = ContextBuilder()
    ctx = builder.build(weather_data, sunrise_h, sunset_h, day_info)

    selector = SmartSelector(engine)
    bundle = selector.generate_bundle(ctx, day_info)

    if not bundle:
        print("\n⚠️ No recommendation could be synthesized.")
        return

    diversity = evaluate_bundle_diversity(bundle)

    print("\n🎁 LIVE REAL-TIME RECOMMENDATION BUNDLE:")
    print("-" * 80)
    hero = bundle["main"]
    print(f"🌟 HERO [{hero['score']:4.2f}] ({hero['item'].get('subcategory')}): {hero['item'].get('text')}")

    for i, sk in enumerate(bundle.get("sidekicks", []), 1):
        print(f"  ↳ Sidekick {i} [{sk['score']:4.2f}] ({sk['item'].get('subcategory')}): {sk['item'].get('text')}")

    print(f"\n✨ Intra-Bundle Diversity Index: {diversity:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ContextAware Recommendation Engine Demo")
    parser.add_argument("--scenario", choices=["1", "2", "3", "4"], default=None, help="Run predefined scenario")
    parser.add_argument("--live", action="store_true", help="Run live weather recommendation")
    parser.add_argument("--lat", type=float, default=35.6892, help="Latitude (default: Tehran 35.6892)")
    parser.add_argument("--lon", type=float, default=51.3890, help="Longitude (default: Tehran 51.3890)")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Path to suggestions dataset folder")

    args = parser.parse_args()

    if args.scenario:
        run_scenario(args.scenario, args.data_dir)
    elif args.live:
        run_live(args.lat, args.lon, args.data_dir)
    else:
        # Default: run Scenario 1 (Yalda) and then Scenario 2 (Norooz)
        run_scenario("1", args.data_dir)
