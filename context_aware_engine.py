import json
import urllib.request
import math
import os
import datetime
import time

# ==================================================================================
# 🌤️ AI Weather Suggestion Engine - Standalone Python Implementation
# ==================================================================================
# This script replicates the logic of the Android app's recommendation engine.
# It fetches weather data, builds a fuzzy context vector, loads suggestions,
# and scores them to find the best matches for the current moment.
# ==================================================================================

# Configuration
# Default location: Tehran (can be changed)
LATITUDE = 35.6892
LONGITUDE = 51.3890
DATA_DIR = os.path.join("app", "src", "main", "assets", "recom_data")

# ==================================================================================
# 1. Feature Definitions & Weights
# ==================================================================================

# Weights for different feature groups (must match Android app logic)
GROUP_WEIGHTS = {
    "temp": 1.0,        # Critical for physical comfort
    "weather": 0.9,     # Critical for feasibility
    "social": 0.9,      # Strong constraint
    "mood": 0.9,        # High emotional relevance
    "time": 0.8,        # Strong temporal constraint
    "location": 0.8,    # Physical constraint
    "events": 0.7,      # Cultural relevance
    "humidity": 0.6,    # Secondary comfort
    "wind": 0.6,        # Secondary comfort
    "season": 0.5,      # Broad context
    "energy": 0.5       # Personal state
}

# List of all feature names in the vector space
FEATURE_NAMES = [
    # Temperature
    "temp_extreme_cold", "temp_cold", "temp_cool", "temp_warm", "temp_hot",
    # Weather
    "weather_clear", "weather_partly_cloudy", "weather_cloudy", "weather_fog",
    "weather_drizzle", "weather_rain", "weather_rain_shower", "weather_snow",
    "weather_snow_shower", "weather_thunderstorm",
    # Humidity
    "humidity_very_dry", "humidity_dry", "humidity_comfortable",
    "humidity_humid", "humidity_very_humid",
    # Wind
    "wind_calm", "wind_breeze", "wind_windy", "wind_strong", "wind_storm",
    # Time
    "time_late_night", "time_early_morning", "time_morning",
    "time_afternoon", "time_evening", "time_night",
    # Day Type
    "day_weekend", "day_holiday", "day_holiday_eve", "day_workday",
    # Season
    "season_spring", "season_summer", "season_autumn", "season_winter",
    # Events
    "romantic_event", "national_festival", "national_mourning", "cultural_tradition",
    # Social
    "social_solo", "social_couple", "social_family", "social_friends", "social_group",
    # Mood
    "mood_calm", "mood_energetic", "mood_happy", "mood_sad", "mood_thoughtful",
    "mood_romantic", "mood_nostalgic", "mood_stressed", "mood_relaxed",
    # Location
    "location_indoor", "location_outdoor", "location_home",
    # Energy
    "energy_very_low", "energy_low", "energy_medium", "energy_high", "energy_very_high"
]

# ==================================================================================
# 2. Helper Functions (Fuzzy Logic)
# ==================================================================================

def fuzzy_membership(value, center, width):
    """
    Calculates the fuzzy membership score (0.0 to 1.0) for a value.
    A triangular function: 1.0 at center, dropping to 0.0 at center +/- width.
    """
    distance = abs(value - center)
    return max(0.0, min(1.0, 1.0 - distance / width))

def add_with_diminishing(current, increment, factor=0.7):
    """
    Adds a value with diminishing returns to prevent saturation (max 1.0).
    """
    current = max(0.0, min(1.0, current))
    headroom = 1.0 - current
    return current + (increment * headroom * factor)

def subtract_with_diminishing(current, decrement, factor=0.7):
    """
    Subtracts a value with diminishing returns (min 0.0).
    """
    current = max(0.0, min(1.0, current))
    return current - (decrement * current * factor)

# ==================================================================================
# 3. Vectorizers (Raw Data -> Fuzzy Features)
# ==================================================================================

class WeatherVectorizer:
    """Converts WMO weather codes and raw values into fuzzy features."""
    
    @staticmethod
    def vectorize_code(code):
        features = {k: 0.0 for k in FEATURE_NAMES if k.startswith("weather_")}
        
        # Mapping logic from WeatherVectorizer.kt
        if code == 0: features["weather_clear"] = 1.0
        elif code == 1: 
            features["weather_clear"] = 0.8
            features["weather_partly_cloudy"] = 0.2
        elif code == 2: features["weather_partly_cloudy"] = 1.0
        elif code == 3: features["weather_cloudy"] = 1.0
        elif code in [45, 48]: features["weather_fog"] = 1.0
        elif code in [51, 56]: features["weather_drizzle"] = 0.6
        elif code == 53: features["weather_drizzle"] = 0.8
        elif code in [55, 57]: features["weather_drizzle"] = 1.0
        elif code in [61, 66]: features["weather_rain"] = 0.6
        elif code == 63: features["weather_rain"] = 0.8
        elif code in [65, 67]: features["weather_rain"] = 1.0
        elif code == 71: features["weather_snow"] = 0.6
        elif code == 73: features["weather_snow"] = 0.8
        elif code in [75, 77]: features["weather_snow"] = 1.0
        elif code == 80: features["weather_rain_shower"] = 0.6
        elif code == 81: features["weather_rain_shower"] = 0.8
        elif code == 82: features["weather_rain_shower"] = 1.0
        elif code == 85: features["weather_snow_shower"] = 0.7
        elif code == 86: features["weather_snow_shower"] = 1.0
        elif code in [95, 96, 99]: features["weather_thunderstorm"] = 1.0
        else: features["weather_clear"] = 0.5 # Default
        
        return features

    @staticmethod
    def vectorize_temp(temp_c, feels_like_c):
        # Effective temp leans towards feels_like
        effective = (0.7 * feels_like_c) + (0.3 * temp_c)
        # Normalize -10 to 40 -> 0.0 to 1.0
        norm = (effective + 10) / 50.0
        norm = max(0.0, min(1.0, norm))
        
        return {
            "temp_extreme_cold": fuzzy_membership(norm, 0.0, 0.2), # -10
            "temp_cold": fuzzy_membership(norm, 0.2, 0.2),         # 0
            "temp_cool": fuzzy_membership(norm, 0.45, 0.2),        # 12.5
            "temp_warm": fuzzy_membership(norm, 0.7, 0.2),         # 25
            "temp_hot": fuzzy_membership(norm, 0.9, 0.2)           # 35+
        }

    @staticmethod
    def vectorize_humidity(humidity_percent):
        norm = humidity_percent / 100.0
        return {
            "humidity_very_dry": fuzzy_membership(norm, 0.10, 0.25),
            "humidity_dry": fuzzy_membership(norm, 0.25, 0.25),
            "humidity_comfortable": fuzzy_membership(norm, 0.45, 0.25),
            "humidity_humid": fuzzy_membership(norm, 0.65, 0.25),
            "humidity_very_humid": fuzzy_membership(norm, 0.85, 0.25)
        }

    @staticmethod
    def vectorize_wind(speed_kmh):
        norm = min(1.0, speed_kmh / 60.0)
        return {
            "wind_calm": fuzzy_membership(norm, 0.00, 0.25),
            "wind_breeze": fuzzy_membership(norm, 0.20, 0.25),
            "wind_windy": fuzzy_membership(norm, 0.45, 0.25),
            "wind_strong": fuzzy_membership(norm, 0.70, 0.25),
            "wind_storm": fuzzy_membership(norm, 0.90, 0.25)
        }

class TimeVectorizer:
    """Converts time into fuzzy features."""
    @staticmethod
    def vectorize(hour):
        # Circular distance logic handled by specific centers
        # 0=24, so late night (0) is close to 23 and 1
        
        def circular_dist(h1, h2):
            d = abs(h1 - h2)
            return min(d, 24 - d) / 12.0 # Normalize 0-1 (12 hours is max dist)
            
        def time_membership(h, center):
            dist = circular_dist(h, center)
            # Width approx 3-4 hours (0.15 normalized)
            return max(0.0, min(1.0, 1.0 - dist / 0.15))

        return {
            "time_late_night": time_membership(hour, 0),    # 00:00
            "time_early_morning": time_membership(hour, 5), # 05:00
            "time_morning": time_membership(hour, 9),       # 09:00
            "time_afternoon": time_membership(hour, 14),    # 14:00
            "time_evening": time_membership(hour, 19),      # 19:00
            "time_night": time_membership(hour, 22)         # 22:00
        }

# ==================================================================================
# 4. Context Builder (The Brain)
# ==================================================================================

class ContextBuilder:
    """
    Integrates all data sources to build the Master Context Vector.
    Infers missing context like Mood and Social state.
    """
    
    def build(self, weather_data, current_hour):
        # 1. Base Features
        temp_f = WeatherVectorizer.vectorize_temp(
            weather_data['current']['temperature_2m'],
            weather_data['current']['apparent_temperature']
        )
        weather_f = WeatherVectorizer.vectorize_code(
            weather_data['current']['weather_code']
        )
        humidity_f = WeatherVectorizer.vectorize_humidity(
            weather_data['current']['relative_humidity_2m']
        )
        wind_f = WeatherVectorizer.vectorize_wind(
            weather_data['current']['wind_speed_10m']
        )
        time_f = TimeVectorizer.vectorize(current_hour)
        
        # 2. Season (Simplified based on month)
        month = datetime.datetime.now().month
        season_f = {
            "season_spring": 1.0 if 3 <= month <= 5 else 0.0,
            "season_summer": 1.0 if 6 <= month <= 8 else 0.0,
            "season_autumn": 1.0 if 9 <= month <= 11 else 0.0,
            "season_winter": 1.0 if month == 12 or month <= 2 else 0.0
        }
        
        # 3. Day Type (Simplified)
        weekday = datetime.datetime.now().weekday() # 0=Mon, 6=Sun
        is_weekend = 1.0 if weekday == 4 or weekday == 5 else 0.0 # Fri/Sat in Iran? Or Thu/Fri? Assuming Thu/Fri for now or Sat/Sun. Let's stick to standard Sat/Sun for simplicity or Fri.
        # Actually, let's assume standard international weekend (Sat/Sun) for this script unless specified.
        # But user is Iranian. Let's assume Friday is weekend.
        is_weekend = 1.0 if weekday == 4 else 0.0 # Friday
        day_f = {
            "day_weekend": is_weekend,
            "day_holiday": 0.0, # Placeholder
            "day_holiday_eve": 0.0,
            "day_workday": 1.0 if not is_weekend else 0.0
        }
        
        # 4. Events (Placeholder)
        event_f = {
            "romantic_event": 0.0,
            "national_festival": 0.0,
            "national_mourning": 0.0,
            "cultural_tradition": 0.0
        }
        
        # 5. Infer Mood (Simplified Logic from Kotlin)
        mood_f = self.infer_mood(weather_f, time_f)
        
        # 6. Infer Social (Simplified)
        social_f = self.infer_social(time_f, is_weekend)
        
        # 7. Infer Location
        location_f = self.infer_location(weather_f, wind_f)
        
        # 8. Infer Energy
        energy_f = self.infer_energy(time_f, weather_f)
        
        # Combine all
        context = {}
        context.update(temp_f)
        context.update(weather_f)
        context.update(humidity_f)
        context.update(wind_f)
        context.update(time_f)
        context.update(day_f)
        context.update(season_f)
        context.update(event_f)
        context.update(mood_f)
        context.update(social_f)
        context.update(location_f)
        context.update(energy_f)
        
        return context

    def infer_mood(self, w, t):
        # Baseline
        mood = {k: 0.1 for k in FEATURE_NAMES if k.startswith("mood_")}
        mood["mood_calm"] = 0.2
        mood["mood_happy"] = 0.3
        
        # Weather Impact
        if w["weather_rain"] > 0.5:
            mood["mood_calm"] = add_with_diminishing(mood["mood_calm"], 0.3)
            mood["mood_thoughtful"] = add_with_diminishing(mood["mood_thoughtful"], 0.3)
        
        if w["weather_clear"] > 0.7:
            mood["mood_happy"] = add_with_diminishing(mood["mood_happy"], 0.4)
            mood["mood_energetic"] = add_with_diminishing(mood["mood_energetic"], 0.3)
            
        # Time Impact
        if t["time_late_night"] > 0.5:
            mood["mood_calm"] = add_with_diminishing(mood["mood_calm"], 0.4)
            mood["mood_thoughtful"] = add_with_diminishing(mood["mood_thoughtful"], 0.3)
            
        return mood

    def infer_social(self, t, is_weekend):
        social = {k: 0.1 for k in FEATURE_NAMES if k.startswith("social_")}
        social["social_solo"] = 0.3
        
        if is_weekend:
            social["social_family"] = add_with_diminishing(social["social_family"], 0.4)
            social["social_friends"] = add_with_diminishing(social["social_friends"], 0.3)
            
        if t["time_evening"] > 0.5:
            social["social_family"] = add_with_diminishing(social["social_family"], 0.2)
            
        return social

    def infer_location(self, w, wind):
        loc = {"location_indoor": 0.5, "location_outdoor": 0.5, "location_home": 0.3}
        
        # Bad weather -> Indoor
        bad_weather = max(w["weather_rain"], w["weather_snow"], w["weather_thunderstorm"])
        if bad_weather > 0.5:
            loc["location_indoor"] = add_with_diminishing(loc["location_indoor"], 0.5)
            loc["location_outdoor"] = subtract_with_diminishing(loc["location_outdoor"], 0.5)
            loc["location_home"] = add_with_diminishing(loc["location_home"], 0.4)
            
        # Good weather -> Outdoor
        if w["weather_clear"] > 0.7 and wind["wind_calm"] > 0.5:
            loc["location_outdoor"] = add_with_diminishing(loc["location_outdoor"], 0.4)
            
        return loc

    def infer_energy(self, t, w):
        energy = {k: 0.2 for k in FEATURE_NAMES if k.startswith("energy_")}
        
        if t["time_morning"] > 0.5:
            energy["energy_high"] = add_with_diminishing(energy["energy_high"], 0.4)
            
        if t["time_late_night"] > 0.5:
            energy["energy_very_low"] = add_with_diminishing(energy["energy_very_low"], 0.6)
            
        return energy

# ==================================================================================
# 5. Data Loader & Scorer
# ==================================================================================

class SuggestionEngine:
    def __init__(self):
        self.suggestions = []
        
    def load_data(self, root_dir):
        """Recursively loads all JSON files from the directory."""
        print(f"📂 Loading data from {root_dir}...")
        count = 0
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".json"):
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                self.suggestions.extend(data)
                                count += len(data)
                    except Exception as e:
                        print(f"⚠️ Error loading {file}: {e}")
        print(f"✅ Loaded {count} suggestions.")

    def score(self, context_vector):
        """
        Scores all suggestions against the context vector.
        Returns sorted list of (score, suggestion).
        """
        scored_items = []
        
        for item in self.suggestions:
            prefs = item.get("preferencesJson", {})
            
            # 1. Veto Check
            # If any feature has a score < -9.0, it's a hard veto
            veto = False
            for key, val in prefs.items():
                if val <= -9.0 and context_vector.get(key, 0.0) > 0.1:
                    veto = True
                    break
            if veto: continue
            
            # 2. Weighted Dot Product
            total_score = 0.0
            
            # We iterate by groups to apply group weights
            # But for simplicity in this script, we can iterate features and look up group weights
            # Optimization: Pre-calculate group membership if needed. 
            # Here we'll do it per feature for clarity.
            
            for feat_name, feat_val in prefs.items():
                if feat_name not in context_vector: continue
                
                ctx_val = context_vector[feat_name]
                if ctx_val <= 0.0: continue
                
                # Find group weight
                weight = 0.5 # Default
                for group, w in GROUP_WEIGHTS.items():
                    if feat_name.startswith(group + "_"):
                        weight = w
                        break
                
                # Dot product component
                total_score += weight * (feat_val * ctx_val)
                
            scored_items.append((total_score, item))
            
        # Sort descending
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return scored_items

    def get_top_by_subcategory(self, context_vector, top_n=3):
        """
        Scores items and returns top N suggestions for EACH subcategory.
        Returns: dict { "Category > Subcategory": [(score, item), ...] }
        """
        scored_items = self.score(context_vector)
        
        # Group by subcategory
        grouped = {}
        for score, item in scored_items:
            cat = item.get('category', 'Unknown')
            sub = item.get('subcategory', 'Unknown')
            key = f"{cat} > {sub}"
            
            if key not in grouped:
                grouped[key] = []
            grouped[key].append((score, item))
            
        # Sort each group and take top N
        results = {}
        for key, items in grouped.items():
            # Sort by score desc
            items.sort(key=lambda x: x[0], reverse=True)
            results[key] = items[:top_n]
            
        return results

# ==================================================================================
# 6. Main Execution
# ==================================================================================

def fetch_weather():
    """Fetches real-time weather from Open-Meteo API."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&current=temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,rain,showers,snowfall,weather_code,wind_speed_10m"
    print(f"🌍 Fetching weather for Lat:{LATITUDE}, Lon:{LONGITUDE}...")
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"❌ Weather API failed: {e}")
        return None

def main():
    # 1. Get Environment Data
    weather_data = fetch_weather()
    if not weather_data:
        print("⚠️ Using dummy weather data due to API failure.")
        weather_data = {
            "current": {
                "temperature_2m": 20, "apparent_temperature": 20,
                "relative_humidity_2m": 50, "weather_code": 0,
                "wind_speed_10m": 5
            }
        }
        
    current_hour = datetime.datetime.now().hour + (datetime.datetime.now().minute / 60.0)
    print(f"🕒 Current Time: {datetime.datetime.now().strftime('%H:%M')}")
    print(f"🌡️  Temperature: {weather_data['current']['temperature_2m']}°C")
    
    # 2. Build Context
    builder = ContextBuilder()
    context = builder.build(weather_data, current_hour)
    
    # Debug: Print top active context features
    print("\n📊 Active Context Features:")
    active_ctx = sorted([(k, v) for k, v in context.items() if v > 0.5], key=lambda x: x[1], reverse=True)
    for k, v in active_ctx[:5]:
        print(f"   - {k}: {v:.2f}")
        
    # 3. Load & Score
    engine = SuggestionEngine()
    engine.load_data(DATA_DIR)
    
    # Get top 3 per subcategory
    grouped_suggestions = engine.get_top_by_subcategory(context, top_n=3)
    
    # 4. Output Results
    output_lines = []
    output_lines.append("\n🏆 Top Suggestions by Subcategory:")
    output_lines.append("=" * 60)
    
    # Sort categories alphabetically for consistent output
    sorted_keys = sorted(grouped_suggestions.keys())
    
    for key in sorted_keys:
        items = grouped_suggestions[key]
        if not items: continue
        
        output_lines.append(f"\n📂 {key}")
        output_lines.append("-" * 60)
        
        for i, (score, item) in enumerate(items):
            output_lines.append(f"  {i+1}. [Score: {score:.2f}] {item.get('text', 'Unknown')}")
            
    output_text = "\n".join(output_lines)
    print(output_text)
    
    # Save to file
    with open("suggestion_output.txt", "w", encoding="utf-8") as f:
        f.write(f"🕒 Current Time: {datetime.datetime.now().strftime('%H:%M')}\n")
        f.write(f"🌡️  Temperature: {weather_data['current']['temperature_2m']}°C\n")
        f.write(output_text)
    print("\n✅ Output saved to suggestion_output.txt")

if __name__ == "__main__":
    main()
