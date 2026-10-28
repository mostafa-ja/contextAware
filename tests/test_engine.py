import os
import pytest
from src.context_aware.engine import SuggestionEngine, validate_suggestion
from src.context_aware.context_builder import ContextBuilder
from src.context_aware.weather_service import create_simulated_weather


def test_schema_validator():
    # Valid item
    valid_item = {
        "text": "آش گرم در هوای برفی",
        "category": "food_drink",
        "subcategory": "comfort_meal",
        "preferencesJson": {"temp_cold": 0.8, "weather_snow": 1.0},
    }
    assert validate_suggestion(valid_item) == []

    # Invalid item (missing text)
    invalid_item = {
        "category": "food_drink",
        "subcategory": "comfort_meal",
        "preferencesJson": {},
    }
    assert len(validate_suggestion(invalid_item)) > 0


def test_veto_logic():
    engine = SuggestionEngine(state_file="scratch_freshness_test.json")

    # Item with thunderstorm veto (preference = -10.0)
    item_picnic = {
        "text": "پیک‌نیک در فضای باز",
        "category": "activity",
        "subcategory": "outdoor_active",
        "preferencesJson": {"weather_thunderstorm": -10.0, "weather_clear": 0.9},
    }

    # Context WITH thunderstorm active
    ctx_storm = {"weather_thunderstorm": 0.80, "weather_clear": 0.0}
    assert engine.is_veto(ctx_storm, item_picnic["preferencesJson"]) is True
    assert engine.calculate_item_score(item_picnic, ctx_storm) == -9999.0

    # Context WITHOUT thunderstorm
    ctx_sunny = {"weather_thunderstorm": 0.0, "weather_clear": 0.90}
    assert engine.is_veto(ctx_sunny, item_picnic["preferencesJson"]) is False
    assert engine.calculate_item_score(item_picnic, ctx_sunny) > 0.0

    if os.path.exists("scratch_freshness_test.json"):
        os.remove("scratch_freshness_test.json")


def test_freshness_decay_and_recovery():
    engine = SuggestionEngine(state_file="scratch_freshness_test.json")
    item = {
        "text": "کتاب‌خوانی عصرانه",
        "category": "media",
        "subcategory": "books",
        "preferencesJson": {"time_evening": 0.7},
    }

    engine.suggestions.append(item)
    engine._ensure_item_state(item)
    assert engine.get_item_freshness(item) == 1.0

    # Apply decay
    engine.apply_decay_for_items([item])
    decayed_w = engine.get_item_freshness(item)
    assert decayed_w < 1.0
    assert decayed_w == pytest.approx(0.90)

    # Recover
    engine.refresh_all_before_run()
    recovered_w = engine.get_item_freshness(item)
    assert recovered_w > decayed_w

    if os.path.exists("scratch_freshness_test.json"):
        os.remove("scratch_freshness_test.json")
