"""
ContextAware: Multi-Objective Bundle Synthesizer (SmartSelector)
==============================================================
Synthesizes coherent ambient lifestyle bundles consisting of a primary
"Hero" suggestion paired with complementary "Sidekick" recommendations
selected via a directed subcategory affinity graph and Boltzmann exploration.
"""

import math
import random
import datetime
from typing import Dict, List, Any, Optional, Set
from .config import EVENT_CANONICAL_MAP, normalize_event_key
from .engine import SuggestionEngine


class SmartSelector:
    """
    Intelligent bundle generator that harmonizes cross-domain recommendations
    (e.g., Food + Activity + Media) aligned with the active context.
    """

    def __init__(self, suggestion_engine: SuggestionEngine):
        self.engine = suggestion_engine

        # ---------------------------------------------------------
        # Directed Subcategory Bundle Affinity Graph
        # Maps Hero Subcategory -> Priority Candidate Subcategories
        # ---------------------------------------------------------
        self.BUNDLE_MAP: Dict[str, List[str]] = {
            # --- FOOD & DRINK ---
            "warming_drinks":     ["books", "music", "home_comfort", "relaxation"],
            "cooling_drinks":     ["outdoor_shade", "quick_snack", "accessories_weather"],
            "light_meal":         ["productivity_focus", "creative", "travel_errand"],
            "comfort_meal":       ["movies", "social_community", "home_comfort", "cooling_drinks"],
            "quick_snack":        ["movies", "music", "boosters", "travel_errand"],
            "seasonal_special":   ["social_community", "books", "sensory", "warming_drinks"],

            # --- ACTIVITIES ---
            "outdoor_active":     ["cooling_drinks", "footwear", "protection_items", "accessories_weather"],
            "outdoor_shade":      ["cooling_drinks", "books", "sensory", "accessories_weather"],
            "indoor_cozy":        ["warming_drinks", "home_comfort", "music", "creative"],
            "home_micro":         ["music", "warming_drinks", "sensory", "boosters"],
            "fitness_wellness":   ["cooling_drinks", "comfort_recovery", "energizing"],
            "social_community":   ["quick_snack", "movies", "music"],
            "travel_errand":      ["footwear", "accessories_weather", "music", "quick_snack"],

            # --- MEDIA ---
            "movies":             ["quick_snack", "home_comfort", "comfort_recovery", "warming_drinks"],
            "books":              ["warming_drinks", "sensory", "home_comfort"],
            "music":              ["boosters", "sensory", "creative", "travel_errand"],
            "creative":           ["music", "warming_drinks", "productivity_focus"],

            # --- CLOTHING / COMFORT ---
            "layering_core":      ["travel_errand", "outdoor_active"],
            "footwear":           ["travel_errand", "outdoor_active"],
            "comfort_recovery":   ["relaxation", "movies", "warming_drinks"],
            "home_comfort":       ["movies", "books", "relaxation"],

            # --- MOOD & WELLNESS ---
            "boosters":           ["quick_snack", "music"],
            "relaxation":         ["home_comfort", "sensory", "warming_drinks"],
            "energizing":         ["outdoor_active", "music", "cooling_drinks"],
            "mental_health_support": ["books", "sensory", "indoor_cozy"],
            "productivity_focus": ["music", "home_micro", "light_meal"],
        }

        # ---------------------------------------------------------
        # Hard Circadian Time Windows (Allowed Local Hours)
        # ---------------------------------------------------------
        self.TIME_GATES: Dict[str, tuple] = {
            # Food & Drink
            "light_meal":        (6.0, 21.0),
            "comfort_meal":      (12.0, 21.5),
            "quick_snack":       (10.0, 25.0),
            "warming_drinks":    (5.0, 26.0),
            "cooling_drinks":    (10.0, 20.0),

            # Activities
            "outdoor_active":    (6.0, 19.0),
            "outdoor_shade":     (10.0, 17.0),
            "travel_errand":     (9.0, 21.0),
            "social_community":  (10.0, 22.0),

            # Media & Psychological focus
            "movies":            (13.0, 25.0),
            "energizing":        (6.0, 17.0),
            "productivity_focus":(7.0, 19.0),
        }

    # -------------------------------------------------------------
    # Contextual Filtering & Temporal Permission
    # -------------------------------------------------------------

    def get_allowed_subcategories(
        self,
        time_context: Dict[str, float],
        is_weekend_like: bool,
        is_holiday: bool,
        hour: Optional[float] = None,
    ) -> Set[str]:
        """Filters eligible subcategories based on circadian phase and workday/holiday status."""
        allowed: Set[str] = set()

        is_morning = time_context.get("time_morning", 0.0) > 0.40 or time_context.get("time_early_morning", 0.0) > 0.40
        is_afternoon = time_context.get("time_afternoon", 0.0) > 0.40
        is_evening = time_context.get("time_evening", 0.0) > 0.40
        is_night = time_context.get("time_night", 0.0) > 0.40 or time_context.get("time_late_night", 0.0) > 0.40

        h = hour if hour is not None else datetime.datetime.now().hour
        is_lunch_window = 12.0 <= h <= 15.0
        is_dinner_window = 19.0 <= h <= 22.0

        # Meal phase gating
        if is_morning:
            allowed.update(["warming_drinks", "light_meal", "seasonal_special"])

        if is_lunch_window:
            allowed.update(["comfort_meal", "cooling_drinks", "seasonal_special"])
            allowed.discard("light_meal")

        if is_afternoon and not is_lunch_window:
            allowed.update(["warming_drinks", "cooling_drinks", "quick_snack", "boosters"])

        if is_dinner_window:
            allowed.update(["comfort_meal", "light_meal", "seasonal_special"])

        if is_night and not is_dinner_window:
            allowed.update(["quick_snack", "warming_drinks"])

        # Always-permissible ambient categories
        allowed.update([
            "home_micro", "home_comfort", "music", "sensory",
            "mental_health_support", "layering_core",
            "accessories_weather", "protection_items", "footwear",
        ])

        # Workday vs Weekend/Holiday scheduling
        if is_weekend_like:
            allowed.update(["social_community", "creative", "movies", "books", "travel_errand"])
            if not is_night:
                allowed.update(["outdoor_active", "outdoor_shade", "fitness_wellness"])
            else:
                allowed.update(["relaxation", "comfort_recovery"])
        else:
            if is_morning or is_afternoon:
                allowed.update(["productivity_focus", "travel_errand", "fitness_wellness"])
            if is_evening or is_night:
                allowed.update(["movies", "books", "relaxation", "creative", "comfort_recovery"])

        if is_holiday:
            allowed.update([
                "social_community", "movies", "comfort_meal",
                "home_comfort", "books", "creative", "relaxation",
            ])
            allowed.discard("productivity_focus")

        return allowed

    def is_time_appropriate(self, item: Dict[str, Any], hour: float) -> bool:
        """Enforces hard time bounds from TIME_GATES."""
        sub = item.get("subcategory", "")
        if sub not in self.TIME_GATES:
            return True

        start, end = self.TIME_GATES[sub]
        check_hour = hour + 24.0 if hour < 4.0 else hour
        return start <= check_hour <= end

    # -------------------------------------------------------------
    # Softmax / Boltzmann Stochastic Selection
    # -------------------------------------------------------------

    def weighted_random_choice(
        self,
        scored_candidates: List[Dict[str, Any]],
        top_n: int = 1,
        temperature: float = 1.0,
    ) -> List[Dict[str, Any]]:
        r"""
        Samples candidates without replacement using Boltzmann distribution:
        $P(i) \propto \exp\left(\frac{S_i - \max S}{T}\right)$
        Temperature $T$ balances exploitation (low $T$) and serendipitous discovery (high $T$).
        """
        if not scored_candidates:
            return []

        max_score = max(x["score"] for x in scored_candidates)
        weights = [math.exp((x["score"] - max_score) / max(temperature, 0.01)) for x in scored_candidates]

        chosen = []
        available = list(zip(scored_candidates, weights))

        for _ in range(min(top_n, len(scored_candidates))):
            if not available:
                break
            candidates, w = zip(*available)
            selected = random.choices(list(candidates), weights=w, k=1)[0]
            chosen.append(selected)
            available = [pair for pair in available if pair[0] is not selected]

        return chosen

    # -------------------------------------------------------------
    # Event-Aware Re-ranking & Biases
    # -------------------------------------------------------------

    def apply_event_bundle_biases(
        self,
        candidates: List[Dict[str, Any]],
        context: Dict[str, float],
        day_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Applies culturally authentic multipliers to candidate scores."""
        events_raw = day_info.get("events") or []
        normalized_events = set()
        for e in events_raw:
            key = normalize_event_key(str(e))
            canon = EVENT_CANONICAL_MAP.get(key)
            if canon:
                normalized_events.add(canon)

        def adjust_score(subcats: List[str], multiplier: float):
            for c in candidates:
                if c["item"].get("subcategory") in subcats:
                    c["score"] *= multiplier

        if "yalda" in normalized_events:
            adjust_score(
                ["indoor_cozy", "home_comfort", "warming_drinks", "books",
                 "comfort_meal", "seasonal_special", "relaxation"],
                1.30,
            )
            adjust_score(["outdoor_active", "outdoor_shade"], 0.75)

        if "norooz" in normalized_events:
            adjust_score(
                ["outdoor_active", "travel_errand", "seasonal_special",
                 "social_community", "light_meal", "energizing"],
                1.25,
            )
            adjust_score(["indoor_cozy", "home_comfort"], 0.90)

        if "sizdah_bedar" in normalized_events:
            adjust_score(
                ["outdoor_active", "travel_errand", "cooling_drinks",
                 "quick_snack", "social_community"],
                1.40,
            )
            adjust_score(["indoor_cozy", "home_comfort"], 0.70)

        if "valentine" in normalized_events:
            adjust_score(
                ["warming_drinks", "indoor_cozy", "home_comfort",
                 "books", "music", "creative", "light_meal", "relaxation"],
                1.35,
            )
            adjust_score(["outdoor_active"], 0.85)

        if any(e in normalized_events for e in ["eid_fitr", "eid_adha", "eid_mabath", "mid_shaban"]):
            adjust_score(
                ["comfort_meal", "seasonal_special", "social_community",
                 "home_comfort", "warming_drinks"],
                1.20,
            )

        if any(e in normalized_events for e in ["ashura", "tasua", "arbaeen"]):
            adjust_score(
                ["mental_health_support", "books", "indoor_cozy",
                 "home_comfort", "light_meal"],
                1.25,
            )
            adjust_score(["movies", "social_community", "boosters", "energizing"], 0.65)

        if any(e in normalized_events for e in ["mother_day", "father_day", "women_day", "men_day"]):
            adjust_score(
                ["home_comfort", "indoor_cozy", "warming_drinks",
                 "books", "movies", "light_meal"],
                1.22,
            )

        return candidates

    # -------------------------------------------------------------
    # End-to-End Bundle Generation Pipeline
    # -------------------------------------------------------------

    def generate_bundle(
        self,
        context: Dict[str, float],
        day_info: Dict[str, Any],
        custom_hour: Optional[float] = None,
        hero_temperature: float = 0.80,
        sidekick_temperature: float = 1.00,
    ) -> Optional[Dict[str, Any]]:
        """
        Executes multi-stage bundle generation:
        1. Identifies allowed subcategories given context & temporal gates.
        2. Filters, scores, and samples the primary Hero suggestion.
        3. Traverses the directed bundle affinity graph to retrieve complementary sidekicks.
        4. Applies deduplication and returns the unified bundle structure.
        """
        now = datetime.datetime.now()
        hour = custom_hour if custom_hour is not None else (now.hour + now.minute / 60.0)

        is_weekend_like = (
            context.get("day_weekend", 0.0) > 0.50
            or context.get("day_holiday", 0.0) > 0.50
            or context.get("day_holiday_eve", 0.0) > 0.50
        )
        is_holiday = bool(day_info.get("is_holiday", False))

        # 1. Allowed subcategories
        time_ctx = {k: v for k, v in context.items() if k.startswith("time_")}
        allowed = self.get_allowed_subcategories(time_ctx, is_weekend_like, is_holiday, hour)

        # 2. Hero Candidate Selection
        hero_candidates = []
        for item in self.engine.suggestions:
            sub = item.get("subcategory", "")
            if sub not in allowed:
                continue
            if not self.is_time_appropriate(item, hour):
                continue

            score = self.engine.calculate_item_score(item, context)
            if score > -0.50:
                hero_candidates.append({"item": item, "score": score})

        hero_candidates = self.apply_event_bundle_biases(hero_candidates, context, day_info)

        heroes = self.weighted_random_choice(hero_candidates, top_n=1, temperature=hero_temperature)
        if not heroes:
            return None

        hero = heroes[0]
        result = {"main": hero, "sidekicks": []}

        # 3. Complementary Sidekicks Selection
        hero_sub = hero["item"].get("subcategory", "")
        target_subs = self.BUNDLE_MAP.get(hero_sub, [])

        side_candidates = []
        for item in self.engine.suggestions:
            if item is hero["item"]:
                continue
            if item.get("subcategory") not in target_subs:
                continue
            if not self.is_time_appropriate(item, hour):
                continue

            s_score = self.engine.calculate_item_score(item, context)
            if s_score > -0.50:
                side_candidates.append({"item": item, "score": s_score})

        side_candidates = self.apply_event_bundle_biases(side_candidates, context, day_info)

        if side_candidates:
            # Deduplicate by text content
            unique: Dict[str, Dict[str, Any]] = {}
            for sc in side_candidates:
                txt = sc["item"].get("text", "")
                unique[txt] = sc

            clean = list(unique.values())
            sidekicks = self.weighted_random_choice(clean, top_n=2, temperature=sidekick_temperature)
            result["sidekicks"] = sidekicks

        return result
