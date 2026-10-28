r"""
ContextAware: Fuzzy Logic Core
==============================
Implements triangular, gaussian, and trapezoidal fuzzy membership functions
with parameterized curves for human perception modeling (temperature,
humidity, wind, and circadian time).
"""

import math
from typing import Dict, Any


def fuzzy_triangular(value: float, center: float, width: float) -> float:
    r"""
    Computes triangular fuzzy membership degree in [0.0, 1.0].
    $\mu(x) = \max(0, 1 - \frac{|x - c|}{w})$
    """
    distance = abs(value - center)
    return max(0.0, min(1.0, 1.0 - distance / width))


def fuzzy_gaussian(value: float, center: float, sigma: float) -> float:
    r"""
    Computes Gaussian fuzzy membership degree in [0.0, 1.0].
    $\mu(x) = \exp\left(-\frac{1}{2} \left(\frac{x - c}{\sigma}\right)^2\right)$
    """
    if sigma <= 0:
        return 1.0 if value == center else 0.0
    val = math.exp(-0.5 * ((value - center) / sigma) ** 2)
    return 0.0 if val < 1e-5 else val


def fuzzy_trapezoidal(value: float, a: float, b: float, c: float, d: float) -> float:
    """
    Computes trapezoidal fuzzy membership degree in [0.0, 1.0].
    Returns 1.0 in [b, c], linearly ramps up from a to b, and ramps down from c to d.
    """
    if value <= a or value >= d:
        return 0.0
    if b <= value <= c:
        return 1.0
    if a < value < b:
        return (value - a) / (b - a) if b > a else 1.0
    if c < value < d:
        return (d - value) / (d - c) if d > c else 1.0
    return 0.0


# Parameterized fuzzy membership configurations for continuous domains
FUZZY_CONFIG: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------
    # Temperature Normalized: norm = (T_effective + 10°C) / 50°C
    # Span: -10°C (norm=0.0) to +40°C (norm=1.0)
    # -------------------------------------------------------------
    "temp_extreme_cold": {"type": "gaussian",   "center": 0.05, "sigma": 0.055},
    "temp_cold":         {"type": "triangular", "center": 0.20, "width": 0.20},
    "temp_cool":         {"type": "trapezoidal","a": 0.28, "b": 0.38, "c": 0.52, "d": 0.64},
    "temp_warm":         {"type": "triangular", "center": 0.70, "width": 0.22},
    "temp_hot":          {"type": "gaussian",   "center": 0.90, "sigma": 0.085},

    # -------------------------------------------------------------
    # Humidity Normalized: norm = RH% / 100
    # Span: 0% to 100%
    # -------------------------------------------------------------
    "humidity_very_dry":    {"type": "triangular", "center": 0.10, "width": 0.22},
    "humidity_dry":         {"type": "triangular", "center": 0.25, "width": 0.22},
    "humidity_comfortable": {"type": "trapezoidal","a": 0.37, "b": 0.43, "c": 0.55, "d": 0.63},
    "humidity_humid":       {"type": "triangular", "center": 0.70, "width": 0.22},
    "humidity_very_humid":  {"type": "gaussian",   "center": 0.88, "sigma": 0.10},

    # -------------------------------------------------------------
    # Wind Normalized: norm = min(1.0, speed_kmh / 60)
    # -------------------------------------------------------------
    "wind_calm":   {"type": "gaussian",   "center": 0.00, "sigma": 0.10},
    "wind_breeze": {"type": "triangular", "center": 0.25, "width": 0.20},
    "wind_windy":  {"type": "triangular", "center": 0.50, "width": 0.22},
    "wind_strong": {"type": "triangular", "center": 0.75, "width": 0.20},
    "wind_storm":  {"type": "gaussian",   "center": 0.95, "sigma": 0.08},

    # -------------------------------------------------------------
    # Circadian Hours: continuous hour in [0.0, 24.0)
    # -------------------------------------------------------------
    "time_late_night":    {"type": "triangular", "center": 2.0,  "width": 5.0},
    "time_early_morning": {"type": "triangular", "center": 6.0,  "width": 2.0},
    "time_morning":       {"type": "triangular", "center": 9.25, "width": 2.25},
    "time_afternoon":     {"type": "triangular", "center": 15.0, "width": 2.5},
    "time_evening":       {"type": "triangular", "center": 19.0, "width": 1.5},
    "time_night":         {"type": "triangular", "center": 21.5, "width": 2.5},
}


def apply_fuzzy(feature_name: str, value: float) -> float:
    """Evaluates membership degree of a numerical value for a named fuzzy set."""
    cfg = FUZZY_CONFIG.get(feature_name)
    if not cfg:
        return 0.0

    t = cfg["type"]
    if t == "triangular":
        return fuzzy_triangular(value, cfg["center"], cfg["width"])
    if t == "gaussian":
        return fuzzy_gaussian(value, cfg["center"], cfg["sigma"])
    if t == "trapezoidal":
        return fuzzy_trapezoidal(value, cfg["a"], cfg["b"], cfg["c"], cfg["d"])
    return 0.0


def add_with_diminishing(current: float, increment: float, factor: float = 0.70) -> float:
    r"""
    Adds a value with diminishing returns (bounded t-conorm operator) in [0.0, 1.0].
    $y = x + \Delta \cdot (1 - x) \cdot \alpha$
    """
    current = max(0.0, min(1.0, current))
    headroom = 1.0 - current
    return max(0.0, min(1.0, current + (increment * headroom * factor)))


def subtract_with_diminishing(current: float, decrement: float, factor: float = 0.70) -> float:
    """Subtracts a value with diminishing returns in [0.0, 1.0]."""
    current = max(0.0, min(1.0, current))
    return max(0.0, min(1.0, current - (decrement * current * factor)))
