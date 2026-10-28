"""
ContextAware: Environmental & Temporal Vectorizers
=================================================
Transforms meteorological observables (WMO weather codes, temperature,
humidity, wind) and temporal signals (solar circadian hours, sunrise/sunset,
and Solar Hijri calendar seasons) into normalized fuzzy vectors.
"""

import math
from typing import Dict
from .config import FEATURE_NAMES
from .fuzzy import apply_fuzzy


class WeatherVectorizer:
    """Transforms raw meteorological readings into continuous fuzzy feature vectors."""

    @staticmethod
    def vectorize_code(code: int) -> Dict[str, float]:
        """Maps WMO Weather Interpretation Codes to fuzzy condition activations."""
        f = {k: 0.0 for k in FEATURE_NAMES if k.startswith("weather_")}

        if code == 0:
            f["weather_clear"] = 1.0
        elif code == 1:
            f["weather_clear"] = 0.8
            f["weather_partly_cloudy"] = 0.2
        elif code == 2:
            f["weather_partly_cloudy"] = 1.0
        elif code == 3:
            f["weather_cloudy"] = 1.0
        elif code in [45, 48]:
            f["weather_fog"] = 1.0
        elif code in [51, 56]:
            f["weather_drizzle"] = 0.6
        elif code == 53:
            f["weather_drizzle"] = 0.8
        elif code in [55, 57]:
            f["weather_drizzle"] = 1.0
        elif code in [61, 66]:
            f["weather_rain"] = 0.6
        elif code == 63:
            f["weather_rain"] = 0.8
        elif code in [65, 67]:
            f["weather_rain"] = 1.0
        elif code == 71:
            f["weather_snow"] = 0.6
        elif code == 73:
            f["weather_snow"] = 0.8
        elif code in [75, 77]:
            f["weather_snow"] = 1.0
        elif code == 80:
            f["weather_rain_shower"] = 0.6
        elif code == 81:
            f["weather_rain_shower"] = 0.8
        elif code == 82:
            f["weather_rain_shower"] = 1.0
        elif code == 85:
            f["weather_snow_shower"] = 0.7
        elif code == 86:
            f["weather_snow_shower"] = 1.0
        elif code in [95, 96, 99]:
            f["weather_thunderstorm"] = 1.0
        else:
            f["weather_clear"] = 0.5

        return f

    @staticmethod
    def vectorize_temp(temp_c: float, feels_like_c: float) -> Dict[str, float]:
        """Calculates perceptual temperature membership using weighted apparent temperature."""
        effective = 0.70 * feels_like_c + 0.30 * temp_c
        norm = max(0.0, min(1.0, (effective + 10.0) / 50.0))
        return {
            "temp_extreme_cold": apply_fuzzy("temp_extreme_cold", norm),
            "temp_cold":         apply_fuzzy("temp_cold", norm),
            "temp_cool":         apply_fuzzy("temp_cool", norm),
            "temp_warm":         apply_fuzzy("temp_warm", norm),
            "temp_hot":          apply_fuzzy("temp_hot", norm),
        }

    @staticmethod
    def vectorize_humidity(humidity_percent: float) -> Dict[str, float]:
        """Maps relative humidity percentage [0, 100] to fuzzy comfort levels."""
        norm = max(0.0, min(1.0, humidity_percent / 100.0))
        return {
            "humidity_very_dry":    apply_fuzzy("humidity_very_dry", norm),
            "humidity_dry":         apply_fuzzy("humidity_dry", norm),
            "humidity_comfortable": apply_fuzzy("humidity_comfortable", norm),
            "humidity_humid":       apply_fuzzy("humidity_humid", norm),
            "humidity_very_humid":  apply_fuzzy("humidity_very_humid", norm),
        }

    @staticmethod
    def vectorize_wind(speed_kmh: float) -> Dict[str, float]:
        """Maps wind velocity (km/h) to Beaufort-scale-like fuzzy descriptors."""
        norm = max(0.0, min(1.0, speed_kmh / 60.0))
        return {
            "wind_calm":   apply_fuzzy("wind_calm", norm),
            "wind_breeze": apply_fuzzy("wind_breeze", norm),
            "wind_windy":  apply_fuzzy("wind_windy", norm),
            "wind_strong": apply_fuzzy("wind_strong", norm),
            "wind_storm":  apply_fuzzy("wind_storm", norm),
        }


class TimeVectorizer:
    """Circadian time vectorizer with astronomical solar diurnal adjustment."""

    @staticmethod
    def vectorize(hour: float, sunrise_hour: float, sunset_hour: float, weekday: int) -> Dict[str, float]:
        """
        Computes continuous fuzzy time memberships.
        Adjusts evening and night weights based on solar daylight boundaries
        and Iranian weekend-eve dynamics (Thursday night).
        """
        is_day = 1 if sunrise_hour <= hour <= sunset_hour else 0

        result = {
            name: apply_fuzzy(name, hour)
            for name in [
                "time_late_night",
                "time_early_morning",
                "time_morning",
                "time_afternoon",
                "time_evening",
                "time_night",
            ]
        }

        # Solar adjustment
        if not is_day:
            result["time_evening"] *= 1.15
            result["time_night"] *= 1.25
            result["time_late_night"] *= 1.10
            result["time_morning"] *= 0.70

        # Iranian weekend eve (Thursday night = weekday 3)
        if weekday == 3:
            result["time_evening"] *= 1.15
            result["time_night"] *= 1.20

        # Clamp all values to [0.0, 1.0]
        for k in result:
            result[k] = max(0.0, min(1.0, result[k]))
        return result


def vectorize_season(persian_month: int, persian_day: int) -> Dict[str, float]:
    """
    Computes smooth fuzzy season memberships based on the Solar Hijri (Jalali) calendar.
    Seasons:
      Months 1-3  (Farvardin - Khordad): Spring
      Months 4-6  (Tir - Shahrivar):     Summer
      Months 7-9  (Mehr - Azar):         Autumn
      Months 10-12 (Dey - Esfand):       Winter
    """
    try:
        m = int(persian_month)
        d = int(persian_day)
    except (TypeError, ValueError):
        m, d = 1, 1

    m = max(1, min(12, m))
    d = max(1, min(30, d))

    # Continuous coordinate along the 12-month annual cycle
    x = m + (d - 15) / 30.0
    x = max(1.0, min(12.0, x))

    def gaussian_circular(x_val: float, center: float, width: float) -> float:
        diff = abs(x_val - center)
        diff = min(diff, 12.0 - diff)
        return math.exp(-0.5 * (diff / width) ** 2)

    # Season peak anchors: Farvardin/Ordibehesht (2), Mordad (5), Aban (8), Bahman (11)
    spring = gaussian_circular(x, 2.0, 1.3)
    summer = gaussian_circular(x, 5.0, 1.3)
    autumn = gaussian_circular(x, 8.0, 1.3)
    winter = gaussian_circular(x, 11.0, 1.3)

    total = spring + summer + autumn + winter
    if total > 0:
        spring /= total
        summer /= total
        autumn /= total
        winter /= total

    return {
        "season_spring": spring,
        "season_summer": summer,
        "season_autumn": autumn,
        "season_winter": winter,
    }
