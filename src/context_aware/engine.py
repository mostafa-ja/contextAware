"""
ContextAware: Suggestion Engine & Freshness Tracker
==================================================
Handles schema validation, multi-criteria veto constraints,
weighted dot-product scoring with cosine normalization, and
the non-linear freshness decay & subcategory-specific recovery engine.
"""

import os
import json
import math
from typing import List, Dict, Any, Tuple, Optional
from .config import (
    FEATURE_NAMES,
    GROUP_WEIGHTS,
    FEATURE_GROUP,
    FRESHNESS_DECAY_FACTOR,
    FRESHNESS_MIN_WEIGHT,
    FRESHNESS_MAX_WEIGHT,
    DEFAULT_RECOVERY_RATE,
    RECOVERY_RATE_BY_SUBCATEGORY,
    MIN_SCORE_THRESHOLD,
    VETO_THRESHOLD,
    VETO_CONTEXT_MIN,
    DEFAULT_FRESHNESS_STATE_FILE,
)


def validate_suggestion(item: Dict[str, Any], filename: str = "", index: int = 0) -> List[str]:
    """
    Validates a suggestion item against the strict ContextAware schema.
    Returns a list of error strings (empty if valid).
    """
    errors = []
    prefix = f"[{filename}#{index}] " if filename else ""

    for required_key in ("text", "category", "subcategory", "preferencesJson"):
        if required_key not in item:
            errors.append(f"{prefix}Missing required attribute '{required_key}'")

    prefs = item.get("preferencesJson")
    if not isinstance(prefs, dict):
        errors.append(f"{prefix}'preferencesJson' must be a valid key-value mapping")
        return errors

    for feat, val in prefs.items():
        if feat not in FEATURE_NAMES:
            errors.append(f"{prefix}Unknown feature key '{feat}'")
        if not isinstance(val, (int, float)):
            errors.append(f"{prefix}Non-numeric preference score for '{feat}': {val}")
        elif val < -10.0 or val > 1.0:
            errors.append(f"{prefix}Value {val} out of valid range [-10.0, 1.0] for '{feat}'")

    return errors


class SuggestionEngine:
    """
    Core Recommender Engine:
    Loads knowledge base, evaluates contextual matching, enforces safety vetoes,
    and manages dynamic item freshness to eliminate repetitive suggestions.
    """

    def __init__(self, state_file: str = DEFAULT_FRESHNESS_STATE_FILE):
        self.suggestions: List[Dict[str, Any]] = []
        self.state_file = state_file
        self.freshness_state: Dict[str, Dict[str, Any]] = self._load_freshness_state()

    # -------------------------------------------------------------
    # Freshness State Engine (Anti-Repetition & Habituation Control)
    # -------------------------------------------------------------

    def _item_id(self, item: Dict[str, Any]) -> str:
        """Returns a stable, collision-free identifier for any suggestion item."""
        if "id" in item:
            return str(item["id"])
        cat = item.get("category", "")
        sub = item.get("subcategory", "")
        txt = item.get("text", "")
        return f"{cat}:{sub}:{txt}"

    def _load_freshness_state(self) -> Dict[str, Dict[str, Any]]:
        """Loads persistent freshness weights from disk if available."""
        if not os.path.exists(self.state_file):
            return {}
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_freshness_state(self) -> None:
        """Persists current freshness states to JSON."""
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.freshness_state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Warning: Could not save freshness state to {self.state_file}: {e}")

    def _ensure_item_state(self, item: Dict[str, Any]) -> None:
        """Registers a suggestion item in the freshness registry if not present."""
        iid = self._item_id(item)
        if iid not in self.freshness_state:
            self.freshness_state[iid] = {
                "weight": FRESHNESS_MAX_WEIGHT,
                "subcategory": item.get("subcategory", ""),
            }

    def _get_recovery_rate(self, subcat: str) -> float:
        """Looks up the subcategory-specific recovery slope per interaction cycle."""
        return RECOVERY_RATE_BY_SUBCATEGORY.get(subcat, DEFAULT_RECOVERY_RATE)

    def refresh_all_before_run(self) -> None:
        """
        Passive recovery tick: executed at the beginning of each session.
        Gradually replenishes the freshness weight of previously recommended items.
        """
        for item in self.suggestions:
            sub = item.get("subcategory", "")
            iid = self._item_id(item)
            state = self.freshness_state.setdefault(
                iid,
                {"weight": FRESHNESS_MAX_WEIGHT, "subcategory": sub},
            )
            w = state.get("weight", FRESHNESS_MAX_WEIGHT)
            w += self._get_recovery_rate(sub)
            state["weight"] = min(FRESHNESS_MAX_WEIGHT, max(FRESHNESS_MIN_WEIGHT, w))
            state["subcategory"] = sub

    def get_item_freshness(self, item: Dict[str, Any]) -> float:
        """Returns the current freshness multiplier in [FRESHNESS_MIN_WEIGHT, 1.0]."""
        iid = self._item_id(item)
        st = self.freshness_state.get(iid)
        if not st:
            return FRESHNESS_MAX_WEIGHT
        w = st.get("weight", FRESHNESS_MAX_WEIGHT)
        return min(FRESHNESS_MAX_WEIGHT, max(FRESHNESS_MIN_WEIGHT, w))

    def apply_decay_for_items(self, items: List[Dict[str, Any]]) -> None:
        r"""
        Applies exponential decay $\gamma \leftarrow \gamma \cdot \lambda$
        to items served to the user.
        """
        for item in items:
            sub = item.get("subcategory", "")
            iid = self._item_id(item)
            state = self.freshness_state.setdefault(
                iid,
                {"weight": FRESHNESS_MAX_WEIGHT, "subcategory": sub},
            )
            w = state.get("weight", FRESHNESS_MAX_WEIGHT) * FRESHNESS_DECAY_FACTOR
            state["weight"] = min(FRESHNESS_MAX_WEIGHT, max(FRESHNESS_MIN_WEIGHT, w))
            state["subcategory"] = sub

    # -------------------------------------------------------------
    # Knowledge Base Ingestion
    # -------------------------------------------------------------

    def load_data(self, root_dir: str) -> int:
        """
        Recursively scans and parses all JSON files in root_dir.
        Performs schema validation and registers items.
        Returns total count of valid items loaded.
        """
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset directory '{root_dir}' does not exist.")

        count_valid = 0
        count_total = 0

        for r, _, files in os.walk(root_dir):
            for file in sorted(files):
                if not file.endswith(".json"):
                    continue

                path = os.path.join(r, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"❌ Error reading {file}: {e}")
                    continue

                if not isinstance(data, list):
                    continue

                for i, item in enumerate(data):
                    count_total += 1
                    errs = validate_suggestion(item, file, i)
                    if errs:
                        continue

                    self.suggestions.append(item)
                    self._ensure_item_state(item)
                    count_valid += 1

        return count_valid

    # -------------------------------------------------------------
    # Algorithmic Scoring & Feasibility Veto
    # -------------------------------------------------------------

    def is_veto(self, ctx: Dict[str, float], prefs: Dict[str, float]) -> bool:
        """
        Safety & Physical Feasibility Veto:
        If an item explicitly prohibits a context condition (e.g. preference <= -9.0)
        and that environmental condition is active in the context (> 0.10),
        the candidate is immediately pruned.
        """
        for feat, pref in prefs.items():
            if pref <= VETO_THRESHOLD and ctx.get(feat, 0.0) > VETO_CONTEXT_MIN:
                return True
        return False

    def score_single_raw(self, ctx: Dict[str, float], prefs: Dict[str, float]) -> float:
        r"""
        Computes weighted dot-product:
        $\text{Score}_{\text{raw}} = \sum_k w_{g(k)} \cdot c_k \cdot p_k$
        """
        total = 0.0
        for feat, pref_val in prefs.items():
            ctx_val = ctx.get(feat, 0.0)
            if ctx_val <= 0.0:
                continue
            group = FEATURE_GROUP.get(feat)
            gw = GROUP_WEIGHTS.get(group, 0.50)
            total += gw * (pref_val * ctx_val)
        return total

    def calculate_item_score(self, item: Dict[str, Any], ctx: Dict[str, float]) -> float:
        """
        Calculates the complete contextual match score for an item:
        1. Veto check (-9999.0 if vetoed).
        2. Weighted dot product.
        3. Cosine-like L2 normalization.
        4. Freshness multiplier discount.
        """
        prefs = item.get("preferencesJson", {})

        # 1. Hard feasibility veto
        if self.is_veto(ctx, prefs):
            return -9999.0

        # 2. Raw score
        raw = self.score_single_raw(ctx, prefs)

        # 3. Normalization (cosine projection)
        ctx_norm = math.sqrt(sum(v * v for v in ctx.values()))
        pref_norm = math.sqrt(sum(v * v for v in prefs.values()))
        norm_score = raw / (ctx_norm * pref_norm + 1e-9)

        # 4. Freshness modulation
        freshness = self.get_item_freshness(item)
        return norm_score * freshness

    def score_all(self, ctx: Dict[str, float]) -> List[Tuple[float, Dict[str, Any]]]:
        """Scores all candidates, filters below threshold, and sorts descending."""
        scored = []
        for item in self.suggestions:
            score = self.calculate_item_score(item, ctx)
            if score >= MIN_SCORE_THRESHOLD:
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def get_top_by_subcategory(
        self,
        ctx: Dict[str, float],
        top_n: int = 3,
    ) -> Dict[str, List[Tuple[float, Dict[str, Any]]]]:
        """Groups highest-scoring recommendations by 'Category > Subcategory'."""
        scored = self.score_all(ctx)
        grouped: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}

        for score, item in scored:
            cat = item.get("category", "Unknown")
            sub = item.get("subcategory", "Unknown")
            key = f"{cat} > {sub}"
            grouped.setdefault(key, []).append((score, item))

        for k in grouped:
            grouped[k] = grouped[k][:top_n]

        return grouped
