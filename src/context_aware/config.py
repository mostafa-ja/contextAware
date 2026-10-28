"""
ContextAware: Configuration & Feature Space Definitions
=======================================================
Defines the 65-dimensional feature taxonomy, group weights for the
weighted dot-product scoring engine, freshness decay hyperparameters,
and Persian calendar canonical event mappings.
"""

# Default Geo-coordinates (Tehran, Iran)
DEFAULT_LATITUDE = 35.6892
DEFAULT_LONGITUDE = 51.3890

# Dataset and State Defaults
DEFAULT_DATA_DIR = "dataset"
DEFAULT_FRESHNESS_STATE_FILE = "freshness_state.json"

# Freshness System Hyperparameters
FRESHNESS_DECAY_FACTOR = 0.90      # Applied to selected items per turn
FRESHNESS_MIN_WEIGHT = 0.05        # Floor weight to prevent complete starvation
FRESHNESS_MAX_WEIGHT = 1.00        # Baseline ceiling weight
DEFAULT_RECOVERY_RATE = 0.010      # Passive recovery rate per execution cycle

# Subcategory-specific recovery modifiers (recovery per interaction cycle)
RECOVERY_RATE_BY_SUBCATEGORY = {
    # Food & Drink (slower recovery for meals, faster for basic hydration)
    "warming_drinks": 0.005,
    "cooling_drinks": 0.005,
    "comfort_meal": 0.005,
    "light_meal": 0.005,
    "quick_snack": 0.005,
    "seasonal_special": 0.005,

    # Activities
    "outdoor_active": 0.010,
    "outdoor_shade": 0.010,
    "indoor_cozy": 0.010,
    "home_micro": 0.010,
    "fitness_wellness": 0.010,
    "travel_errand": 0.010,
    "social_community": 0.010,

    # Clothing & Accessories
    "layering_core": 0.008,
    "accessories_weather": 0.008,
    "footwear": 0.008,
    "protection_items": 0.008,
    "comfort_recovery": 0.010,
    "home_comfort": 0.010,

    # Media & Creative
    "movies": 0.003,
    "books": 0.003,
    "music": 0.002,
    "creative": 0.005,

    # Mood & Wellness
    "boosters": 0.010,
    "relaxation": 0.010,
    "energizing": 0.010,
    "mental_health_support": 0.010,
    "productivity_focus": 0.010,
    "sensory": 0.010,
    "social_connection": 0.010,
}

# Iranian calendar weekend: Friday (weekday 4 in Python datetime: Monday=0, Sunday=6)
WEEKEND_DAYS = {4}

# Minimum candidate threshold for scoring inclusion
MIN_SCORE_THRESHOLD = -1.0

# Veto threshold constant (hard feasibility veto)
VETO_THRESHOLD = -9.0
VETO_CONTEXT_MIN = 0.10

# Multi-objective feature group weights (must sum or be balanced according to cognitive priority)
GROUP_WEIGHTS = {
    "temp": 1.0,        # Critical physical thermal comfort
    "weather": 0.9,     # Feasibility constraint (rain, snow, storm)
    "social": 0.6,      # Contextual social dynamics
    "mood": 0.6,        # Emotional alignment
    "time": 1.0,        # Circadian / temporal constraint
    "location": 1.0,    # Physical location viability (indoor vs outdoor)
    "events": 0.8,      # Cultural and festive relevance
    "humidity": 0.6,    # Secondary comfort
    "wind": 0.6,        # Secondary comfort & outdoor feasibility
    "season": 0.5,      # Broad seasonal context
    "energy": 0.5,      # Personal fatigue/energy constraint
}

# 65-Dimensional Master Feature Space
FEATURE_NAMES = [
    # Temperature (5 features)
    "temp_extreme_cold", "temp_cold", "temp_cool", "temp_warm", "temp_hot",

    # Weather Condition (10 features)
    "weather_clear", "weather_partly_cloudy", "weather_cloudy", "weather_fog",
    "weather_drizzle", "weather_rain", "weather_rain_shower", "weather_snow",
    "weather_snow_shower", "weather_thunderstorm",

    # Humidity (5 features)
    "humidity_very_dry", "humidity_dry", "humidity_comfortable",
    "humidity_humid", "humidity_very_humid",

    # Wind (5 features)
    "wind_calm", "wind_breeze", "wind_windy", "wind_strong", "wind_storm",

    # Time of Day (6 features)
    "time_late_night", "time_early_morning", "time_morning",
    "time_afternoon", "time_evening", "time_night",

    # Day Type (4 features)
    "day_weekend", "day_holiday", "day_holiday_eve", "day_workday",

    # Season (4 features)
    "season_spring", "season_summer", "season_autumn", "season_winter",

    # Cultural & National Events (4 features)
    "romantic_event", "national_festival", "national_mourning", "cultural_tradition",

    # Social Dynamics (5 features)
    "social_solo", "social_couple", "social_family", "social_friends", "social_group",

    # Inferred Emotional State / Mood (9 features)
    "mood_calm", "mood_energetic", "mood_happy", "mood_sad", "mood_thoughtful",
    "mood_romantic", "mood_nostalgic", "mood_stressed", "mood_relaxed",

    # Location / Environment (3 features)
    "location_indoor", "location_outdoor", "location_home",

    # Human Energy Levels (5 features)
    "energy_very_low", "energy_low", "energy_medium", "energy_high", "energy_very_high",
]

# Fast reverse mapping from feature name to group key
FEATURE_GROUP = {}
for feat in FEATURE_NAMES:
    for group in GROUP_WEIGHTS.keys():
        if feat.startswith(group + "_"):
            FEATURE_GROUP[feat] = group
            break

# Canonical Persian event dictionary
EVENT_CANONICAL_MAP = {
    # Norooz (Persian New Year)
    "نوروز": "norooz", "nowruz": "norooz", "norooz": "norooz",

    # Chaharshanbe Suri (Festival of Fire)
    "چهارشنبهسوری": "chaharshanbe_suri", "chaharshanbesuri": "chaharshanbe_suri",

    # Yalda Night (Winter Solstice Celebration)
    "یلدا": "yalda", "yalda": "yalda",

    # Sizdah Bedar (Nature Day)
    "سیزدهبدر": "sizdah_bedar", "sizdahbedar": "sizdah_bedar",

    # Traditional Mourning Days
    "عاشورا": "ashura", "ashura": "ashura",
    "تاسوعا": "tasua", "tasua": "tasua",
    "اربعین": "arbaeen", "arbaeen": "arbaeen",

    # Eids & Religious Festivals
    "عیدفطر": "eid_fitr", "eidfitr": "eid_fitr",
    "عیدقربان": "eid_adha", "eidadha": "eid_adha",
    "عیدمبعث": "eid_mabath", "eidmabath": "eid_mabath",
    "نیمهشعبان": "mid_shaban", "midshaban": "mid_shaban",

    # Family & Honorific Days
    "روزمادر": "mother_day", "roozemadar": "mother_day",
    "روزپدر": "father_day", "roozpedar": "father_day",
    "روززن": "women_day", "roozezan": "women_day",
    "روزمرد": "men_day", "roozemard": "men_day",

    # Romance
    "ولنتاین": "valentine", "valentine": "valentine",
}


def normalize_event_key(name: str) -> str:
    """Normalizes an event string by stripping Persian zero-width spaces and hyphens."""
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    for ch in [" ", "‌", "_", "-", "ـ"]:
        s = s.replace(ch, "")
    return s
