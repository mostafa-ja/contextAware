import pytest
from src.context_aware.engine import SuggestionEngine
from src.context_aware.selector import SmartSelector
from src.context_aware.context_builder import ContextBuilder
from src.context_aware.weather_service import create_simulated_weather


def test_smart_selector_bundle_structure():
    engine = SuggestionEngine(state_file="scratch_test_selector.json")

    # Add mock suggestions spanning different categories
    engine.suggestions = [
        {
            "text": "دمنوش بابونه و بهارنارنج",
            "category": "food_drink",
            "subcategory": "warming_drinks",
            "preferencesJson": {"temp_cold": 0.9, "time_night": 0.8, "location_home": 0.8},
        },
        {
            "text": "خواندن رمان کلیدر",
            "category": "media",
            "subcategory": "books",
            "preferencesJson": {"time_night": 0.8, "location_home": 0.8, "mood_calm": 0.8},
        },
        {
            "text": "پلی‌لیست موسیقی آرامش‌بخش سنتی",
            "category": "media",
            "subcategory": "music",
            "preferencesJson": {"time_night": 0.8, "mood_calm": 0.9},
        },
        {
            "text": "طراحی ماندالا با مداد رنگی",
            "category": "media",
            "subcategory": "creative",
            "preferencesJson": {"time_night": 0.8, "location_home": 0.8, "mood_calm": 0.8},
        },
        {
            "text": "جوراب پشمی دستبافت",
            "category": "clothing",
            "subcategory": "home_comfort",
            "preferencesJson": {"temp_cold": 0.9, "location_home": 0.9},
        },
    ]

    for it in engine.suggestions:
        engine._ensure_item_state(it)

    selector = SmartSelector(engine)
    builder = ContextBuilder()

    # Freezing cold winter night at home
    weather_data, sr, ss = create_simulated_weather(temperature=0.0, weather_code=0)
    day_info = {"day_type": "day_workday", "persian_month": 10, "persian_day": 15, "is_holiday": False, "events": []}

    ctx = builder.build(weather_data, sr, ss, day_info)
    bundle = selector.generate_bundle(ctx, day_info, custom_hour=21.0)

    assert bundle is not None
    assert "main" in bundle
    assert "item" in bundle["main"]
    assert "score" in bundle["main"]
    assert "sidekicks" in bundle
    assert len(bundle["sidekicks"]) > 0

    # Sidekicks should belong to mapped categories (books, music, home_comfort, etc.)
    hero_sub = bundle["main"]["item"]["subcategory"]
    for sk in bundle["sidekicks"]:
        sk_sub = sk["item"]["subcategory"]
        assert sk_sub in selector.BUNDLE_MAP.get(hero_sub, [])


def test_smart_selector_time_gating():
    engine = SuggestionEngine(state_file="scratch_test_selector.json")
    selector = SmartSelector(engine)

    item_lunch = {"subcategory": "comfort_meal"}

    # 13:00 (1:00 PM) is within lunch window
    assert selector.is_time_appropriate(item_lunch, hour=13.0) is True

    # 07:00 (7:00 AM) is outside comfort_meal gate (12:00 to 21:30)
    assert selector.is_time_appropriate(item_lunch, hour=7.0) is False
