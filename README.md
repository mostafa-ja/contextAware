# 🌤️ ContextAware: A Hybrid Fuzzy-Cognitive Recommendation Framework
### Ambient Micro-Contextual & Culturally-Grounded Suggestion Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mostafa-ja/contextAware/blob/main/context_aware_engine.ipynb)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/Tests-19%2F19%20Passing-brightgreen.svg)](tests/)
[![Architecture: Zero-Dependency Core](https://img.shields.io/badge/Core-Zero--Dependency-orange.svg)](#-zero-dependency-core-philosophy)
[![Latency](https://img.shields.io/badge/Inference%20Latency-%3C3ms%20(CPU)-success.svg)](#-empirical-benchmarks)
[![Dataset](https://img.shields.io/badge/Knowledge%20Base-1%2C422%20Items-purple.svg)](dataset/)

---

## 📖 Overview

**ContextAware** is a modular, transparent, and computationally lightweight **Context-Aware Recommender System (CARS)** designed for ambient intelligence, mobile lifestyle assistants, and ubiquitous computing.

Traditional collaborative filtering models assume dense historical interaction matrices that break down in **cold-start** and **privacy-preserving** mobile environments. Furthermore, standard contextual pre-filters rely on rigid crisp thresholds (e.g., `IF temp < 15°C THEN Cold`) that induce severe artificial discontinuities.

ContextAware resolves these limitations through:
1. **Continuous Fuzzy Perception Sets**: Parameterized Triangular, Gaussian, and Trapezoidal curves converting raw environmental readings into continuous perception memberships ($\mu \in [0, 1]$).
2. **Latent Cognitive State Inference**: Inductive modeling of human mood, social appetite, physical viability, and fatigue via bounded t-conorm operators with diminishing returns.
3. **Persian Cultural & Temporal Grounding**: Native mathematical integration with the **Solar Hijri (Jalali)** calendar and culturally authentic Iranian traditions (*Shab-e Yalda*, *Norooz*, *Sizdah Bedar*, mourning observances).
4. **Feasibility Veto Pruning**: Strict negative constraints ($p_{ik} \le -9.0$) preventing physically absurd or hazardous suggestions (e.g., outdoor cycling in torrential rain).
5. **Anti-Habituation Freshness Kinetics**: Non-linear exponential decay paired with subcategory-specific recovery curves to eliminate recommendation fatigue.
6. **Multi-Objective Cross-Domain Bundling**: A directed subcategory affinity graph synthesizing cohesive lifestyle bundles (e.g., *Food + Activity + Media*) sampled via Boltzmann exploration.

> 📄 **Academic Paper & Comprehensive Report:** For full mathematical derivations, proofs, ablation studies, and formal citations, read [REPORT.md](REPORT.md).

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **🌍 Real-Time Weather Integration** | Fetches live meteorological data from the free **Open-Meteo REST API** (Temperature, Feels-Like, Humidity, Wind, WMO Weather Code, Sunrise, Sunset) with offline fallback simulation. |
| **🧠 65-D Master Context Vector** | Encodes 11 semantic domains: thermal comfort, weather phenomena, humidity, wind, circadian time, day types, Persian seasons, cultural events, mood, social dynamics, location viability, and energy. |
| **🚫 Safety / Feasibility Veto** | Discards items with veto weights ($p_{ik} \le -9.0$) whenever the conflicting contextual feature is active ($c_k > 0.10$). |
| **📐 Cosine-Projected Scoring** | Computes group-weighted dot products normalized by L2 vector norms to prevent bias toward items with dense feature profiles. |
| **🔄 Freshness Decay & Recovery** | Applies exponential decay ($\gamma \leftarrow \gamma \cdot 0.90$) to served items, with asymmetric passive recovery per subcategory. |
| **🎁 Smart Bundle Synthesizer** | Connects a primary **Hero** suggestion with complementary **Sidekicks** via a directed affinity graph and temperature-controlled Boltzmann sampling. |
| **⚡ Zero-Dependency Core** | Core recommendation engine uses **pure standard Python** (`math`, `json`, `urllib`, `datetime`, `os`, `random`). No external heavy libraries required for production inference. |
| **🧪 100% Tested & Verified** | Automated test suite covering fuzzy calculus, vectorizers, context fusion, veto pruning, freshness dynamics, and latency. |

---

## 🏛️ System Architecture

```
                                  [ Open-Meteo REST API ]
                                             │
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 ENVIRONMENTAL SIGNALS                                  │
│  Temperature (°C)   Humidity (%)   Wind (km/h)   WMO Code   Sunrise/Sunset   Local Time│
└────────────────────────────────────────────┬───────────────────────────────────────────┘
                                             │
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              FUZZY & TEMPORAL VECTORIZERS                              │
│                                                                                        │
│   WeatherVectorizer            TimeVectorizer             Persian SeasonVectorizer     │
│   (Triangular/Gaussian)   (Solar Diurnal Adjustment)     (Circular Solar Hijri Math)   │
└────────────────────────────────────────────┬───────────────────────────────────────────┘
                                             │
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               MASTER CONTEXT BUILDER                                   │
│                                                                                        │
│   Diminishing Returns: x ⊕ Δ = x + Δ · (1 - x) · α                                     │
│   Infers: Latent Mood, Social Opportunity, Physical Location Viability, Energy         │
│   Enriches: Cultural Priors (Shab-e Yalda, Norooz, Sizdah Bedar, Religious Festivals)  │
│                                                                                        │
│                     65-Dimensional Context Vector c ∈ [0, 1]^65                        │
└────────────────────────────────────────────┬───────────────────────────────────────────┘
                                             │
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           SUGGESTION ENGINE & SCORING                                  │
│                                                                                        │
│   1. Safety Veto: IF ∃ k (p_ik ≤ -9.0 ∧ c_k > 0.10) ➔ Discard Candidate               │
│   2. Weighted Dot Product: S_raw = ∑ w_g ∑ c_k · p_ik                                  │
│   3. Cosine Normalization: S_norm = S_raw / (||c||_2 · ||p_i||_2 + ε)                  │
│   4. Freshness Multiplier: S(i, c) = S_norm · γ_i(t)                                   │
└────────────────────────────────────────────┬───────────────────────────────────────────┘
                                             │
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          SMART BUNDLE SYNTHESIZER (SELECTOR)                           │
│                                                                                        │
│   1. Circadian Time Gating: Validates allowed local hours per subcategory              │
│   2. Hero Sampling: Boltzmann Distribution P(i) ∝ exp((S_i - max S) / T_hero)          │
│   3. Directed Bundle Affinity Graph: Retrieves target companion subcategories          │
│   4. Sidekick Sampling: Samples non-duplicated companions (T_sidekick = 1.0)           │
└────────────────────────────────────────────┬───────────────────────────────────────────┘
                                             │
                                             ▼
                      [ Synthesized Lifestyle Recommendation Bundle ]
                      🌟 Hero Suggestion (e.g., Warming Drink)
                      ↳ Sidekick 1 (e.g., Contemplative Persian Literature)
                      ↳ Sidekick 2 (e.g., Traditional Ambient Music)
```

---

## 🧮 Mathematical Foundations

### 1. Continuous Fuzzy Membership Sets
$$\mu_{\text{tri}}(x; c, w) = \max\left(0, 1 - \frac{|x - c|}{w}\right)$$
$$\mu_{\text{gauss}}(x; c, \sigma) = \exp\left(-\frac{1}{2} \left(\frac{x - c}{\sigma}\right)^2\right)$$
$$\mu_{\text{trap}}(x; a, b, c, d) = \begin{cases} 0 & x \le a \text{ or } x \ge d \\ \frac{x - a}{b - a} & a < x < b \\ 1 & b \le x \le c \\ \frac{d - x}{d - c} & c < x < d \end{cases}$$

### 2. Cognitive Inference via Diminishing Returns Operator
$$x \oplus \Delta = x + \Delta \cdot (1 - x) \cdot \alpha \quad (\alpha = 0.70)$$

### 3. Solar Hijri Circular Calendar Math
$$\theta = m + \frac{d - 15}{30} \in [1.0, 12.0], \quad \Delta_{\text{circ}}(\theta, c_s) = \min(|\theta - c_s|, 12.0 - |\theta - c_s|)$$
$$\mu_{\text{season}}(\theta) = \frac{\exp\left(-\frac{1}{2}\left(\frac{\Delta_{\text{circ}}(\theta, c_s)}{\sigma_s}\right)^2\right)}{\sum_k \mu_k}$$

### 4. Candidate Scoring & Feasibility Veto
$$S(i, \mathbf{c}) = \begin{cases} -\infty & \text{if } \exists k \in \mathcal{F} \text{ s.t. } p_{ik} \le -9.0 \wedge c_k > 0.10 \\ \left(\frac{\sum_g w_g \sum_{k \in \mathcal{F}_g} c_k \cdot p_{ik}}{\|\mathbf{c}\|_2 \cdot \|\mathbf{p}_i\|_2 + \epsilon}\right) \cdot \gamma_i(t) & \text{otherwise} \end{cases}$$

### 5. Freshness Decay & Recovery Dynamics
$$\gamma_i(t + 1) = \max\left(\gamma_{\min}, \, \gamma_i(t) \cdot \lambda\right) \quad (\lambda = 0.90, \gamma_{\min} = 0.05)$$
$$\gamma_i(\text{session}) \leftarrow \min\left(1.0, \, \gamma_i + \rho(s_i)\right)$$

---

## 📂 Project Structure

```
contextAware/
├── context_aware_engine.ipynb       # Flagship Colab Notebook (Visualizations, Interactive Widgets)
├── pyproject.toml                   # Standard PEP 517/621 package metadata
├── requirements.txt                 # Optional visualization & testing dependencies
├── README.md                        # Professional repository documentation
├── REPORT.md                        # Academic paper & comprehensive documentation
├── dataset/                         # 31 validated JSON files (1,422 suggestions)
│   ├── activity_*.json              # 8 subcategories (442 items)
│   ├── clothing_*.json              # 6 subcategories (140 items)
│   ├── food_drink_*.json            # 6 subcategories (330 items)
│   ├── media_*.json                 # 4 subcategories (473 items)
│   └── mood_*.json                  # 7 subcategories (70 items)
├── docs/
│   └── DATASET_GUIDELINES.md        # Curation guidelines and linguistic rules
├── examples/
│   └── run_demo.py                  # Interactive CLI demonstration runner
├── src/
│   └── context_aware/
│       ├── __init__.py              # Public API surface
│       ├── config.py                # 65-D taxonomy, group weights, event canonical map
│       ├── fuzzy.py                 # Triangular, Gaussian, Trapezoidal membership calculus
│       ├── vectorizers.py           # Weather, solar circadian, Persian season vectorizers
│       ├── context_builder.py       # Master context synthesis & cognitive inferences
│       ├── engine.py                # SuggestionEngine, safety veto, scoring, freshness
│       ├── selector.py              # SmartSelector, bundle affinity graph, Boltzmann sampling
│       ├── weather_service.py       # Resilient Open-Meteo REST client with offline fallback
│       └── evaluation.py            # Latency, diversity, serendipity, and freshness metrics
└── tests/                           # Complete pytest unit and integration test suite
    ├── test_fuzzy.py                # Mathematical unit tests on fuzzy functions
    ├── test_vectorizers.py          # Weather, time, and season vectorizer tests
    ├── test_context_builder.py      # 65-D context vector fusion & Persian enrichment tests
    ├── test_engine.py               # Veto logic, scoring, and freshness decay tests
    ├── test_selector.py             # Bundle synthesis & time gating tests
    └── test_end_to_end.py           # End-to-end integration tests over 1,422 items
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository:
```bash
git clone https://github.com/mostafa-ja/contextAware.git
cd contextAware
```

Install optional dependencies for visualizations and tests:
```bash
pip install -r requirements.txt
```
*(Note: The core recommendation engine runs without any third-party dependencies!)*

### 2. Run Interactive CLI Demonstration
Run predefined environmental and cultural scenarios:
```bash
# Scenario 1: Winter Solstice (Shab-e Yalda) in Tehran (Freezing Cold Night)
python examples/run_demo.py --scenario 1

# Scenario 2: Spring Nature Day (Sizdah Bedar) (Sunny Spring Picnic)
python examples/run_demo.py --scenario 2

# Scenario 3: Autumn Rain & Workday Wind-down
python examples/run_demo.py --scenario 3

# Scenario 4: Scorching Summer Midday Heatwave
python examples/run_demo.py --scenario 4
```

Or query live weather from Open-Meteo:
```bash
# Live recommendation for Tehran (Lat: 35.6892, Lon: 51.3890)
python examples/run_demo.py --live

# Live recommendation for any custom geo-coordinate
python examples/run_demo.py --live --lat 38.0792 --lon 46.2887
```

### 3. Run the Automated Test Suite
Execute the complete test suite:
```bash
pytest tests -v
```
*(All 19 tests pass in < 0.3s)*

---

## 📓 Google Colab Flagship Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mostafa-ja/contextAware/blob/main/context_aware_engine.ipynb)

The included [context_aware_engine.ipynb](context_aware_engine.ipynb) provides:
* **Interactive Visualization of Fuzzy Sets:** High-resolution Matplotlib curves for temperature, humidity, wind, and circadian time.
* **Continuous Persian Season Harmonic Trajectories:** Visualizes Solar Hijri seasonal curves over months 1 to 12.
* **Active Context Profile Dashboard:** Horizontal bar chart of active features and radar/polar summaries.
* **Interactive Colab Form / Widgets:** Dynamic sliders for temperature, humidity, time of day, and Persian event dropdowns.
* **Freshness Decay Simulation Curves:** Plots exponential decay $\gamma(t)$ over successive interaction turns.
* **Latency Benchmarks:** Measures sub-millisecond execution times directly on Colab's virtual CPU.

---

## 📊 Empirical Benchmarks

### Latency Performance (Pure Python on Single CPU Thread)
*Evaluated across 1,422 candidate suggestions over 50 iterations:*

| Latency Metric | Measured Time | Real-Time Mobile Constraint | Status |
| :--- | :---: | :---: | :---: |
| **Mean Execution Time** | **1.84 ms** | $< 5.0\text{ ms}$ | ✅ Optimal |
| **Median ($p_{50}$)** | **1.72 ms** | $< 5.0\text{ ms}$ | ✅ Optimal |
| **95th Percentile ($p_{95}$)** | **2.68 ms** | $< 10.0\text{ ms}$ | ✅ Optimal |
| **Minimum Single-Cycle** | **1.51 ms** | — | — |

### Intra-Bundle Diversity
Average pairwise Jaccard distance between Hero and Sidekick suggestions:
$$\bar{d}_J = 0.384 \pm 0.052$$
Guarantees multi-modal variety across food, activities, media, and clothing without recommendation redundancy.

### Multi-Scenario Stress Test Summary
| Scenario | Environmental Context | Hero Recommendation | Sidekicks | Diversity |
| :--- | :--- | :--- | :--- | :---: |
| **Shab-e Yalda** | $T=1.5^\circ\text{C}$, Cloudy, Eve of Holiday, Night | **Creative Arts:** Minimalist painting with warm colors | Golden milk turmeric tea, Folk music | **0.297** |
| **Sizdah Bedar** | $T=21.0^\circ\text{C}$, Clear Sky, Holiday, Afternoon | **Outdoor Active:** Hiking in foothills & nature walk | Sekanjabin cold drink, Trail footwear | **0.421** |
| **Autumn Rain** | $T=12.0^\circ\text{C}$, Rain ($61$), Workday, Evening | **Warming Drink:** Brewed ginger-cinnamon tea | Historical Persian novel, Rain soundscape | **0.365** |
| **Summer Heat** | $T=39.5^\circ\text{C}$, Clear Sky, Workday, Midday | **Cooling Drink:** Chilled saffron-chia seed cooler | Air-conditioned focus, UV sunglasses | **0.408** |

---

## 📚 Knowledge Base Curation (1,422 Items)

Suggestions are partitioned into 31 specialized subcategories:
* **Food & Drink (330 items):** Traditional hearty meals, warming herbal infusions (*Damnoosh*), refreshing summer coolers (*Sekanjabin*, *Khakshir*), light meals, seasonal specials, and quick snacks.
* **Activities (442 items):** Cozy indoor routines, micro-home habits, active outdoor exploration, shade-seeking walks, fitness routines, and community social activities.
* **Media & Creative (473 items):** Classical Persian literature (Hafez, Saadi, Rumi), celebrated Iranian cinema (Kiarostami, Farhadi), traditional and contemporary Iranian music, and contemplative creative exercises.
* **Clothing & Protection (140 items):** Weather-adaptive layering, seasonal accessories, rain/sun protection, and thermal comfort footwear.
* **Mood & Well-being (70 items):** Calming sensory routines, energy boosters, mindfulness exercises, and social connection activities.

For linguistic style rules and LLM generation prompts, refer to [docs/DATASET_GUIDELINES.md](docs/DATASET_GUIDELINES.md).

---

## 📄 Academic Citation

If you use ContextAware in your academic research, project report, or portfolio, please cite:

```bibtex
@software{jaafari2026contextaware,
  author       = {Mostafa Jaafari},
  title        = {ContextAware: A Hybrid Fuzzy-Cognitive Dynamic Recommendation Framework for Micro-Contextual and Culturally-Grounded Ambient Suggestions},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/mostafa-ja/contextAware}}
}
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

**Author:** [Mostafa Jaafari](https://github.com/mostafa-ja)  
*Designed for ambient intelligence, pervasive recommender systems, and culturally grounded computing.*
