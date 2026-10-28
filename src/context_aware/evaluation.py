"""
ContextAware: Empirical Evaluation & Benchmarking Suite
======================================================
Provides quantitative metrics and ablation evaluation functions:
1. Intra-Bundle Diversity (Jaccard Distance over preference vectors).
2. Freshness Decay Kinetics over multi-turn interaction cycles.
3. Candidate Coverage and Category Distribution.
4. Latency Benchmark across varying candidate pool sizes.
5. Scenario Sensitivity & Perturbation Analysis.
"""

import time
import math
from typing import List, Dict, Any, Tuple
from .engine import SuggestionEngine
from .selector import SmartSelector
from .context_builder import ContextBuilder
from .weather_service import create_simulated_weather


def compute_jaccard_distance(prefs_a: Dict[str, float], prefs_b: Dict[str, float], threshold: float = 0.20) -> float:
    r"""
    Computes Jaccard distance between two feature activation profiles:
    $d_J(A, B) = 1 - \frac{|A \cap B|}{|A \cup B|}$
    """
    set_a = {k for k, v in prefs_a.items() if abs(v) >= threshold}
    set_b = {k for k, v in prefs_b.items() if abs(v) >= threshold}

    if not set_a and not set_b:
        return 0.0

    union = set_a | set_b
    intersection = set_a & set_b
    return 1.0 - (len(intersection) / len(union))


def evaluate_bundle_diversity(bundle: Dict[str, Any]) -> float:
    """
    Evaluates Intra-Bundle Diversity: Average pairwise Jaccard distance
    between the Main Hero item and each Sidekick item.
    Returns float in [0.0, 1.0]. High diversity indicates well-rounded bundles.
    """
    main_item = bundle.get("main", {}).get("item", {})
    sidekicks = bundle.get("sidekicks", [])

    if not sidekicks:
        return 0.0

    main_prefs = main_item.get("preferencesJson", {})
    distances = []

    for sk in sidekicks:
        sk_item = sk.get("item", {})
        sk_prefs = sk_item.get("preferencesJson", {})
        distances.append(compute_jaccard_distance(main_prefs, sk_prefs))

    return sum(distances) / len(distances) if distances else 0.0


def simulate_freshness_decay_curve(
    engine: SuggestionEngine,
    sample_item: Dict[str, Any],
    turns: int = 10,
) -> List[Tuple[int, float]]:
    r"""
    Simulates consecutive recommendations of the same item to trace
    its exponential weight decay curve $\gamma_t = \max(\gamma_{\min}, \gamma_0 \cdot \lambda^t)$.
    """
    history = []
    engine._ensure_item_state(sample_item)
    iid = engine._item_id(sample_item)

    # Reset to 1.0 for clean simulation
    engine.freshness_state[iid]["weight"] = 1.00

    for t in range(turns + 1):
        w = engine.get_item_freshness(sample_item)
        history.append((t, round(w, 4)))
        if t < turns:
            engine.apply_decay_for_items([sample_item])

    return history


def benchmark_inference_latency(
    engine: SuggestionEngine,
    context: Dict[str, float],
    day_info: Dict[str, Any],
    iterations: int = 100,
) -> Dict[str, float]:
    """
    Measures algorithmic latency (scoring and bundle synthesis) across repeated iterations.
    Returns latency percentiles (mean, min, p95, p99) in milliseconds.
    """
    selector = SmartSelector(engine)
    latencies_ms = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = selector.generate_bundle(context, day_info)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)

    latencies_ms.sort()
    mean_val = sum(latencies_ms) / len(latencies_ms)
    min_val = latencies_ms[0]
    max_val = latencies_ms[-1]
    p50_val = latencies_ms[int(0.50 * len(latencies_ms))]
    p95_val = latencies_ms[int(0.95 * len(latencies_ms))]
    p99_val = latencies_ms[int(0.99 * len(latencies_ms))]

    return {
        "mean_ms": round(mean_val, 3),
        "min_ms": round(min_val, 3),
        "p50_ms": round(p50_val, 3),
        "p95_ms": round(p95_val, 3),
        "p99_ms": round(p99_val, 3),
        "max_ms": round(max_val, 3),
        "candidate_pool_size": len(engine.suggestions),
    }


def run_scenario_benchmark(
    engine: SuggestionEngine,
    scenarios: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Runs a suite of environmental scenarios and records the generated recommendation bundles."""
    builder = ContextBuilder()
    selector = SmartSelector(engine)
    results = []

    for sc in scenarios:
        w_data, sr, ss = create_simulated_weather(**sc["weather_params"])
        ctx = builder.build(w_data, sr, ss, sc["day_info"])
        bundle = selector.generate_bundle(ctx, sc["day_info"], custom_hour=sc.get("custom_hour"))

        diversity = evaluate_bundle_diversity(bundle) if bundle else 0.0
        main_text = bundle["main"]["item"]["text"] if bundle and "main" in bundle else "None"
        main_cat = bundle["main"]["item"]["category"] if bundle and "main" in bundle else "None"
        main_sub = bundle["main"]["item"]["subcategory"] if bundle and "main" in bundle else "None"
        main_score = bundle["main"]["score"] if bundle and "main" in bundle else 0.0

        sidekicks_texts = [
            f"{sk['item'].get('subcategory')}: {sk['item'].get('text')}"
            for sk in (bundle.get("sidekicks", []) if bundle else [])
        ]

        results.append({
            "scenario_name": sc["name"],
            "main_text": main_text,
            "main_category": main_cat,
            "main_subcategory": main_sub,
            "main_score": round(main_score, 3),
            "sidekicks": sidekicks_texts,
            "diversity_score": round(diversity, 3),
        })

    return results
