"""
ContextAware: Meteorological Data Service
=========================================
Communicates with Open-Meteo REST API without requiring external SDKs,
extracts current conditions, astronomical sunrise/sunset, and provides
resilient offline simulation fallbacks for benchmarking and testing.
"""

import json
import urllib.request
from typing import Tuple, Dict, Any, Optional
from .config import DEFAULT_LATITUDE, DEFAULT_LONGITUDE


def decimal_hour_from_iso(iso_str: str) -> float:
    """Extracts decimal hours (e.g. 06:30 -> 6.50) from an ISO time string 'YYYY-MM-DDTHH:MM'."""
    try:
        time_part = iso_str.split("T")[1]
        hh, mm = time_part.split(":")[:2]
        return int(hh) + int(mm) / 60.0
    except Exception:
        return 6.0


def fetch_weather(
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    timeout_sec: float = 6.0,
) -> Tuple[Dict[str, Any], float, float]:
    """
    Fetches real-time weather from Open-Meteo API.
    Returns: (weather_data_dict, sunrise_hour, sunset_hour).
    Falls back gracefully to realistic baseline conditions if offline.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        "&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
        "precipitation,weather_code,cloud_cover,wind_speed_10m"
        "&daily=sunrise,sunset"
        "&timezone=auto"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ContextAware/2.0"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            data = json.loads(response.read().decode("utf-8"))

        sunrise_list = data.get("daily", {}).get("sunrise", [])
        sunset_list = data.get("daily", {}).get("sunset", [])

        sunrise_h = decimal_hour_from_iso(sunrise_list[0]) if sunrise_list else 6.0
        sunset_h = decimal_hour_from_iso(sunset_list[0]) if sunset_list else 18.0

        return data, sunrise_h, sunset_h

    except Exception as e:
        # Realistic offline fallback (e.g. clear autumn day in Tehran)
        fallback_data = {
            "current": {
                "temperature_2m": 18.5,
                "apparent_temperature": 18.0,
                "relative_humidity_2m": 42.0,
                "weather_code": 1,
                "wind_speed_10m": 6.5,
                "precipitation": 0.0,
                "cloud_cover": 15.0,
            },
            "is_fallback": True,
            "error_msg": str(e),
        }
        return fallback_data, 6.0, 18.0


def create_simulated_weather(
    temperature: float = 20.0,
    apparent_temp: Optional[float] = None,
    humidity: float = 45.0,
    weather_code: int = 0,
    wind_speed: float = 5.0,
    sunrise_hour: float = 6.0,
    sunset_hour: float = 18.0,
) -> Tuple[Dict[str, Any], float, float]:
    """Generates synthetic meteorological conditions for controlled testing & scenario simulations."""
    if apparent_temp is None:
        apparent_temp = temperature

    data = {
        "current": {
            "temperature_2m": float(temperature),
            "apparent_temperature": float(apparent_temp),
            "relative_humidity_2m": float(humidity),
            "weather_code": int(weather_code),
            "wind_speed_10m": float(wind_speed),
            "precipitation": 1.0 if weather_code in [61, 63, 65, 80, 81, 82] else 0.0,
            "cloud_cover": 0.0 if weather_code == 0 else 50.0,
        },
        "is_simulated": True,
    }
    return data, sunrise_hour, sunset_hour
