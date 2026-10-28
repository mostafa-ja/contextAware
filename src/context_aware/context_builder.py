r"""
ContextAware: Master Context Vector Builder
===========================================
Fuses multi-modal environmental, temporal, and cultural signals into the
65-dimensional Master Context Vector $\mathbf{c} \in [0, 1]^{65}$.
Infuses cognitive baselines (mood, social dynamics, location viability, energy)
and applies contextual cultural priors (Persian events, national holidays).
"""

import datetime
from typing import Dict, Any, Optional
from .config import (
    FEATURE_NAMES,
    WEEKEND_DAYS,
    EVENT_CANONICAL_MAP,
    normalize_event_key,
)
from .vectorizers import (
    WeatherVectorizer,
    TimeVectorizer,
    vectorize_season,
)


def clamp01(x: float) -> float:
    """Clamps a floating point number to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(x)))


class ContextBuilder:
    """
    Constructs the unified 65-dimensional context representation
    from weather data, solar astronomical timing, and Persian calendar metadata.
    """

    def build(
        self,
        weather_data: Dict[str, Any],
        sunrise_hour: float = 6.0,
        sunset_hour: float = 18.0,
        day_info: Optional[Dict[str, Any]] = None,
        custom_datetime: Optional[datetime.datetime] = None,
    ) -> Dict[str, float]:
        """
        Builds the 65-D context vector.

        :param weather_data: Dictionary returned by Open-Meteo or fallback simulation.
        :param sunrise_hour: Astronomical sunrise in local decimal hours (e.g. 6.25).
        :param sunset_hour: Astronomical sunset in local decimal hours (e.g. 18.50).
        :param day_info: Metadata containing Persian month, day, holiday status, and events.
        :param custom_datetime: Optional datetime override for simulation/backtesting.
        :return: Dict mapping each feature in FEATURE_NAMES to a float in [0.0, 1.0].
        """
        if day_info is None:
            day_info = {
                "day_type": "day_workday",
                "persian_month": 1,
                "persian_day": 1,
                "is_holiday": False,
                "events": [],
            }

        current = weather_data.get("current", {})

        # 1. Weather Signals
        temp_f = WeatherVectorizer.vectorize_temp(
            current.get("temperature_2m", 20.0),
            current.get("apparent_temperature", 20.0),
        )
        weather_f = WeatherVectorizer.vectorize_code(int(current.get("weather_code", 0)))
        humidity_f = WeatherVectorizer.vectorize_humidity(current.get("relative_humidity_2m", 50.0))
        wind_f = WeatherVectorizer.vectorize_wind(current.get("wind_speed_10m", 5.0))

        # 2. Circadian & Astronomical Time Signals
        now = custom_datetime if custom_datetime else datetime.datetime.now()
        hour = now.hour + now.minute / 60.0 + now.second / 3600.0
        weekday = now.weekday()
        time_f = TimeVectorizer.vectorize(hour, sunrise_hour, sunset_hour, weekday)

        # 3. Persian Date & Season
        persian_month = day_info.get("persian_month", 1)
        persian_day = day_info.get("persian_day", 1)
        is_holiday = bool(day_info.get("is_holiday", False))
        day_type = day_info.get("day_type", "day_workday")

        season_f = vectorize_season(persian_month, persian_day)

        # 4. Day-Type Features
        day_f = {
            "day_weekend": 0.0,
            "day_holiday": 0.0,
            "day_holiday_eve": 0.0,
            "day_workday": 0.0,
        }

        if day_type in day_f:
            day_f[day_type] = 1.0
        else:
            is_weekend_py = 1.0 if weekday in WEEKEND_DAYS else 0.0
            day_f["day_weekend"] = is_weekend_py
            day_f["day_workday"] = 1.0 - is_weekend_py

        if is_holiday:
            day_f["day_holiday"] = max(day_f["day_holiday"], 1.0)
            day_f["day_workday"] *= 0.40

        # 5. Cultural & National Event Features
        event_f = {
            "romantic_event": 0.0,
            "national_festival": 0.0,
            "national_mourning": 0.0,
            "cultural_tradition": 0.0,
        }

        # 6. Cognitive & Behavioral Inferences
        mood_f = self.infer_mood(weather_f, time_f)

        is_weekend_like = (
            day_f["day_weekend"] > 0.50
            or day_f["day_holiday"] > 0.50
            or day_type == "day_holiday_eve"
        )
        social_f = self.infer_social(time_f, 1.0 if is_weekend_like else 0.0)
        location_f = self.infer_location(weather_f, wind_f, temp_f, time_f)
        energy_f = self.infer_energy(time_f)

        # 7. Enrich with Persian Cultural Nuances & Holiday Priors
        self.enrich_with_persian_day_info(
            day_info=day_info,
            season_f=season_f,
            day_f=day_f,
            event_f=event_f,
            mood_f=mood_f,
            social_f=social_f,
            location_f=location_f,
            energy_f=energy_f,
        )

        # 8. Unify into complete 65-dimensional Context Vector
        ctx: Dict[str, float] = {}
        for group in (
            temp_f, weather_f, humidity_f, wind_f, time_f, day_f,
            season_f, event_f, mood_f, social_f, location_f, energy_f
        ):
            ctx.update(group)

        # Ensure all defined features exist and are clamped
        for feat in FEATURE_NAMES:
            ctx[feat] = clamp01(ctx.get(feat, 0.0))

        return ctx

    # -------------------------------------------------------------
    # Cognitive & Environmental Inference Modules
    # -------------------------------------------------------------

    def infer_mood(self, w: Dict[str, float], t: Dict[str, float]) -> Dict[str, float]:
        """Infers subjective emotional state from environmental atmosphere."""
        mood = {k: 0.10 for k in FEATURE_NAMES if k.startswith("mood_")}
        mood["mood_calm"] = 0.20
        mood["mood_happy"] = 0.30

        # Atmospheric precipitation impact
        if w.get("weather_rain", 0.0) > 0.50:
            mood["mood_calm"] += 0.30
            mood["mood_thoughtful"] += 0.30

        # High solar radiance impact
        if w.get("weather_clear", 0.0) > 0.70:
            mood["mood_happy"] += 0.40
            mood["mood_energetic"] += 0.30

        # Nocturnal introspective shift
        if t.get("time_late_night", 0.0) > 0.50:
            mood["mood_calm"] += 0.40
            mood["mood_thoughtful"] += 0.30

        return {k: clamp01(v) for k, v in mood.items()}

    def infer_social(self, t: Dict[str, float], is_weekend_like: float) -> Dict[str, float]:
        """Infers social group preference from temporal opportunity."""
        social = {k: 0.10 for k in FEATURE_NAMES if k.startswith("social_")}
        social["social_solo"] = 0.30

        if is_weekend_like > 0.50:
            social["social_family"] += 0.40
            social["social_friends"] += 0.30

        if t.get("time_evening", 0.0) > 0.50:
            social["social_family"] += 0.20

        return {k: clamp01(v) for k, v in social.items()}

    def infer_location(
        self,
        w: Dict[str, float],
        wind: Dict[str, float],
        temp: Dict[str, float],
        time_f: Dict[str, float],
    ) -> Dict[str, float]:
        """Infers indoor vs outdoor vs home viability based on physical comfort constraints."""
        loc = {
            "location_indoor": 0.40,
            "location_outdoor": 0.40,
            "location_home": 0.30,
        }

        # Adverse precipitation
        rain_strength = (
            w.get("weather_rain", 0.0)
            + 0.70 * w.get("weather_rain_shower", 0.0)
            + 0.40 * w.get("weather_drizzle", 0.0)
        )
        snow_strength = (
            w.get("weather_snow", 0.0)
            + 0.60 * w.get("weather_snow_shower", 0.0)
        )
        thunder_strength = w.get("weather_thunderstorm", 0.0)
        bad_weather = max(rain_strength, snow_strength, thunder_strength)

        if bad_weather > 0.20:
            boost = min(0.60, bad_weather)
            loc["location_indoor"] += 0.40 * boost
            loc["location_outdoor"] -= 0.50 * boost
            loc["location_home"] += 0.30 * boost

        # High winds
        wind_bad = (
            0.40 * wind.get("wind_windy", 0.0)
            + 0.60 * wind.get("wind_strong", 0.0)
            + 1.00 * wind.get("wind_storm", 0.0)
        )
        if wind_bad > 0.25:
            loc["location_outdoor"] -= 0.40 * wind_bad
            loc["location_indoor"] += 0.20 * wind_bad

        # Thermal extremes
        too_cold = temp.get("temp_extreme_cold", 0.0) + 0.70 * temp.get("temp_cold", 0.0)
        too_hot = temp.get("temp_hot", 0.0)

        if too_cold > 0.20:
            loc["location_indoor"] += 0.30 * too_cold
            loc["location_outdoor"] -= 0.40 * too_cold

        if too_hot > 0.20:
            loc["location_indoor"] += 0.30 * too_hot
            loc["location_outdoor"] -= 0.30 * too_hot

        # Ideal weather boost
        perfect_weather = w.get("weather_clear", 0.0) * wind.get("wind_calm", 0.0)
        if perfect_weather > 0.50:
            loc["location_outdoor"] += 0.35 * perfect_weather
            loc["location_indoor"] -= 0.20 * perfect_weather

        # Night shifts towards home
        night_val = time_f.get("time_night", 0.0)
        if night_val > 0.40:
            loc["location_outdoor"] -= 0.25 * night_val
            loc["location_home"] += 0.20 * night_val

        late_night_val = time_f.get("time_late_night", 0.0)
        if late_night_val > 0.30:
            loc["location_outdoor"] -= 0.35 * late_night_val
            loc["location_home"] += 0.30 * late_night_val

        # Softmax-like normalization of probabilities
        for k in loc:
            loc[k] = clamp01(loc[k])
        total = sum(loc.values()) or 1.0
        for k in loc:
            loc[k] /= total

        return loc

    def infer_energy(self, t: Dict[str, float]) -> Dict[str, float]:
        """Infers energy and vitality from circadian phase."""
        e = {k: 0.20 for k in FEATURE_NAMES if k.startswith("energy_")}
        if t.get("time_morning", 0.0) > 0.50:
            e["energy_high"] += 0.40
        if t.get("time_late_night", 0.0) > 0.50:
            e["energy_very_low"] += 0.60
        return {k: clamp01(v) for k, v in e.items()}

    # -------------------------------------------------------------
    # Persian Cultural & Holiday Enrichment
    # -------------------------------------------------------------

    def enrich_with_persian_day_info(
        self,
        day_info: Dict[str, Any],
        season_f: Dict[str, float],
        day_f: Dict[str, float],
        event_f: Dict[str, float],
        mood_f: Dict[str, float],
        social_f: Dict[str, float],
        location_f: Dict[str, float],
        energy_f: Dict[str, float],
    ) -> None:
        """Applies domain-specific behavioral modifiers for Persian events and holidays."""
        # Holiday baseline
        if day_info.get("is_holiday"):
            day_f["day_holiday"] = clamp01(max(day_f.get("day_holiday", 0.0), 1.0))
            mood_f["mood_happy"] = clamp01(mood_f.get("mood_happy", 0.0) + 0.20)
            mood_f["mood_relaxed"] = clamp01(mood_f.get("mood_relaxed", 0.0) + 0.20)
            social_f["social_family"] = clamp01(social_f.get("social_family", 0.0) + 0.40)
            social_f["social_group"] = clamp01(social_f.get("social_group", 0.0) + 0.30)
            location_f["location_home"] = clamp01(location_f.get("location_home", 0.0) + 0.20)

        # Normalize event tags
        events_raw = day_info.get("events") or []
        normalized_events = set()
        for e in events_raw:
            key = normalize_event_key(str(e))
            canon = EVENT_CANONICAL_MAP.get(key)
            if canon:
                normalized_events.add(canon)

        def add_m(k: str, d: float): mood_f[k] = clamp01(mood_f.get(k, 0.0) + d)
        def add_s(k: str, d: float): social_f[k] = clamp01(social_f.get(k, 0.0) + d)
        def add_l(k: str, d: float): location_f[k] = clamp01(location_f.get(k, 0.0) + d)
        def add_e(k: str, d: float): energy_f[k] = clamp01(energy_f.get(k, 0.0) + d)

        # 1. Yalda Night (Shab-e Yalda)
        if "yalda" in normalized_events:
            event_f["cultural_tradition"] = 1.0
            add_m("mood_nostalgic", 0.50)
            add_m("mood_romantic", 0.30)
            add_m("mood_relaxed", 0.30)
            add_s("social_family", 0.50)
            add_s("social_friends", 0.30)
            add_l("location_home", 0.50)
            add_l("location_indoor", 0.30)
            season_f["season_winter"] = max(season_f.get("season_winter", 0.0), 0.70)

        # 2. Norooz (Spring Equinox New Year)
        if "norooz" in normalized_events:
            event_f["national_festival"] = 1.0
            event_f["cultural_tradition"] = max(event_f["cultural_tradition"], 0.90)
            add_m("mood_happy", 0.40)
            add_m("mood_energetic", 0.30)
            add_s("social_family", 0.50)
            add_s("social_friends", 0.40)
            add_s("social_group", 0.30)
            add_l("location_outdoor", 0.30)
            add_l("location_home", 0.20)
            add_e("energy_high", 0.30)

        # 3. Sizdah Bedar (Nature Day)
        if "sizdah_bedar" in normalized_events:
            event_f["national_festival"] = max(event_f["national_festival"], 0.90)
            event_f["cultural_tradition"] = max(event_f["cultural_tradition"], 0.90)
            add_s("social_family", 0.40)
            add_s("social_group", 0.40)
            add_l("location_outdoor", 0.60)
            location_f["location_home"] = clamp01(location_f.get("location_home", 0.0) - 0.20)
            add_m("mood_happy", 0.40)
            add_m("mood_energetic", 0.40)
            add_e("energy_high", 0.30)

        # 4. Mourning Observances
        if any(ev in normalized_events for ev in ["ashura", "tasua", "arbaeen"]):
            event_f["national_mourning"] = 1.0
            add_m("mood_sad", 0.60)
            add_m("mood_thoughtful", 0.50)
            mood_f["mood_happy"] = clamp01(mood_f.get("mood_happy", 0.0) * 0.60)
            mood_f["mood_energetic"] = clamp01(mood_f.get("mood_energetic", 0.0) * 0.70)
            add_s("social_group", 0.40)
            add_s("social_family", 0.30)
            add_l("location_outdoor", 0.30)
            add_l("location_home", 0.10)
            add_e("energy_low", 0.30)

        # 5. Religious Eids
        if any(ev in normalized_events for ev in ["eid_fitr", "eid_adha", "eid_mabath", "mid_shaban"]):
            event_f["national_festival"] = max(event_f["national_festival"], 0.90)
            event_f["cultural_tradition"] = max(event_f["cultural_tradition"], 0.70)
            add_m("mood_happy", 0.40)
            add_m("mood_relaxed", 0.20)
            add_s("social_family", 0.40)
            add_s("social_friends", 0.30)
            add_s("social_group", 0.30)
            add_l("location_home", 0.20)
            add_l("location_outdoor", 0.20)
            add_e("energy_medium", 0.20)

        # 6. Family & Honorific Days
        if any(ev in normalized_events for ev in ["mother_day", "father_day", "women_day", "men_day"]):
            add_m("mood_nostalgic", 0.30)
            add_m("mood_happy", 0.30)
            add_s("social_family", 0.50)
            add_l("location_home", 0.30)
            event_f["cultural_tradition"] = max(event_f["cultural_tradition"], 0.40)

        # 7. Valentine Day
        if "valentine" in normalized_events:
            event_f["romantic_event"] = 1.0
            add_m("mood_romantic", 0.60)
            add_m("mood_happy", 0.30)
            add_s("social_couple", 0.70)
            add_s("social_friends", 0.20)
            add_l("location_home", 0.30)
            add_l("location_outdoor", 0.20)
            add_e("energy_medium", 0.20)
