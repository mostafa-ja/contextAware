import pytest
from src.context_aware.vectorizers import (
    WeatherVectorizer,
    TimeVectorizer,
    vectorize_season,
)


def test_weather_vectorizer_code():
    # Clear sky
    res_clear = WeatherVectorizer.vectorize_code(0)
    assert res_clear["weather_clear"] == 1.0
    assert res_clear["weather_rain"] == 0.0

    # Rain
    res_rain = WeatherVectorizer.vectorize_code(65)
    assert res_rain["weather_rain"] == 1.0
    assert res_rain["weather_clear"] == 0.0

    # Thunderstorm
    res_storm = WeatherVectorizer.vectorize_code(95)
    assert res_storm["weather_thunderstorm"] == 1.0


def test_weather_vectorizer_temp():
    # Freezing cold (-5°C)
    freezing = WeatherVectorizer.vectorize_temp(-5.0, -8.0)
    assert freezing["temp_extreme_cold"] > freezing["temp_warm"]
    assert freezing["temp_hot"] == pytest.approx(0.0, abs=1e-4)

    # Scorching summer (+38°C)
    hot = WeatherVectorizer.vectorize_temp(38.0, 42.0)
    assert hot["temp_hot"] > 0.5
    assert hot["temp_extreme_cold"] == pytest.approx(0.0, abs=1e-4)


def test_time_vectorizer_circadian():
    # Morning (9:00 AM)
    morning = TimeVectorizer.vectorize(hour=9.0, sunrise_hour=6.0, sunset_hour=18.0, weekday=1)
    assert morning["time_morning"] > morning["time_late_night"]

    # Late Night (2:00 AM)
    late_night = TimeVectorizer.vectorize(hour=2.0, sunrise_hour=6.0, sunset_hour=18.0, weekday=1)
    assert late_night["time_late_night"] > late_night["time_afternoon"]


def test_vectorize_season_persian():
    # Farvardin 15 (Spring)
    spring = vectorize_season(1, 15)
    assert spring["season_spring"] > spring["season_winter"]
    assert spring["season_spring"] > spring["season_summer"]

    # Mordad 15 (Summer)
    summer = vectorize_season(5, 15)
    assert summer["season_summer"] > summer["season_winter"]

    # Day 1 (Winter)
    winter = vectorize_season(10, 1)
    assert winter["season_winter"] > winter["season_summer"]
