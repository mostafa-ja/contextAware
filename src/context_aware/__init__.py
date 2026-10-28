"""
ContextAware: AI Contextual Suggestion Engine
============================================
A lightweight, zero-dependency, ambient intelligence recommendation engine
combining fuzzy logic, circadian & astronomical time vectorization,
Persian cultural priors, and multi-objective bundle synthesis.
"""

from .config import (
    FEATURE_NAMES,
    GROUP_WEIGHTS,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    DEFAULT_DATA_DIR,
    DEFAULT_FRESHNESS_STATE_FILE,
)
from .fuzzy import (
    fuzzy_triangular,
    fuzzy_gaussian,
    fuzzy_trapezoidal,
    apply_fuzzy,
)
from .vectorizers import (
    WeatherVectorizer,
    TimeVectorizer,
    vectorize_season,
)
from .context_builder import ContextBuilder
from .engine import SuggestionEngine, validate_suggestion
from .selector import SmartSelector
from .weather_service import fetch_weather, create_simulated_weather
from .evaluation import (
    compute_jaccard_distance,
    evaluate_bundle_diversity,
    benchmark_inference_latency,
    simulate_freshness_decay_curve,
    run_scenario_benchmark,
)

__version__ = "2.1.0"
__author__ = "Mostafa Jaafari"
__all__ = [
    "FEATURE_NAMES",
    "GROUP_WEIGHTS",
    "DEFAULT_LATITUDE",
    "DEFAULT_LONGITUDE",
    "DEFAULT_DATA_DIR",
    "DEFAULT_FRESHNESS_STATE_FILE",
    "fuzzy_triangular",
    "fuzzy_gaussian",
    "fuzzy_trapezoidal",
    "apply_fuzzy",
    "WeatherVectorizer",
    "TimeVectorizer",
    "vectorize_season",
    "ContextBuilder",
    "SuggestionEngine",
    "validate_suggestion",
    "SmartSelector",
    "fetch_weather",
    "create_simulated_weather",
    "compute_jaccard_distance",
    "evaluate_bundle_diversity",
    "benchmark_inference_latency",
    "simulate_freshness_decay_curve",
    "run_scenario_benchmark",
]
