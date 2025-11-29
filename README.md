# 🌤️ AI Contextual Suggestion Engine (Python)

A standalone Python implementation of an **AI-powered fuzzy-logic recommendation system**.
This engine analyzes **real-time weather**, **time of day**, **season**, and **inferred human context** (mood, energy, social state) to suggest the most relevant Persian-cultural activities, foods, drinks, clothing, and media for the current moment.

It is a faithful Python port of the engine used in the Android app—lightweight, dependency-free, and runnable on any machine.

---

## ✨ Key Features

### 🌍 **Real-Time Weather Awareness**

Automatically fetches live data from the free **Open-Meteo API**, including:

* Temperature & Feels-Like temperature
* Humidity
* Wind speed
* Weather condition code

No API keys required.

### 🧠 **Fuzzy Logic Core**

Uses fuzzy membership functions to convert raw weather/time into human-like perceptions:

* “15°C → 70% cool, 30% cold”
* “23:00 → strong night + medium late night”

This produces a **rich 65-dimensional context vector**.

### 📐 **AI Scoring via Weighted Dot Product**

Every suggestion in the dataset has a custom preference vector.
The engine computes:

```
Score = Σ (Context_feature × Suggestion_preference × Group_weight)
```

This prioritizes:

* feasibility (weather, temperature)
* emotional fit (mood)
* physical constraints (location, wind, rain)
* cultural timing (events, season)

### 🚫 **Safety / Feasibility Veto**

If a suggestion contains a `-10.0` veto feature (e.g., picnic in thunderstorm)
and the context activates that feature → it is **immediately removed**.

### 📂 **JSON Data Driven**

Loads all suggestions dynamically from a `dataset/` directory:

* food & drink
* activities
* media
* clothing
* mood

Each file is simple JSON—easy to extend or auto-generate.

### 🚀 **Zero Dependencies**

Pure Python.
No `pip install` required.
Uses only:

* `json`
* `urllib`
* `datetime`
* `math`
* `os`

---

## 🧩 How It Works (Simplified)

1. **Fetch weather**
2. **Vectorize weather/time** using fuzzy logic
3. **Infer mood, social state, location, and energy**
4. **Build a 65-dimensional context vector**
5. **Load all JSON suggestions** from `dataset/`
6. **Score each item** using weighted dot product
7. **Apply veto logic**
8. **Group top suggestions by category/subcategory**
9. **Print + save results** to `suggestion_output.txt`

---

## 📊 Flowchart

```
                          ┌────────────────────────┐
                          │    Start Application   │
                          └────────────┬───────────┘
                                       │
                                       ▼
                     ┌─────────────────────────────────┐
                     │ Fetch Real-Time Weather (API)   │
                     └──────────────────┬──────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │ Build Context Vector     │
                          └─────────────┬────────────┘
                                        │
      ┌─────────────────────────────────┼────────────────────────────────┐
      ▼                                 ▼                                ▼
┌───────────────────┐         ┌─────────────────────┐         ┌─────────────────────┐
│ Weather Vectorizer│         │ Time Vectorizer     │         │ Infer Human Context │
│ (temp/wind/etc.)  │         │ (morning/evening)   │         │ (mood/social/etc.)  │
└───────────┬───────┘         └──────────┬──────────┘         └──────────┬──────────┘
            │                              │                           │
            └──────────────┬───────────────┴──────────────┬─────────────┘
                           ▼                               ▼
              ┌─────────────────────────────────────────────────────────┐
              │ Combine All Into 65-Dimensional Context Vector          │
              └─────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │ Load Suggestions (JSONs) │
                       └─────────────┬────────────┘
                                     │
                                     ▼
                       ┌──────────────────────────┐
                       │ Score Suggestions        │
                       │ - Veto rules             │
                       │ - Weighted dot product   │
                       └─────────────┬────────────┘
                                     │
                                     ▼
                   ┌──────────────────────────────────────────┐
                   │ Group Top Results per Category/Subcat     │
                   └─────────────────────┬──────────────────────┘
                                         │
                                         ▼
                           ┌────────────────────────────┐
                           │ Display & Save Results     │
                           └────────────────────────────┘
```

---

## 🚀 How to Run

### ✔ Requirements

* Python **3.6+**

### ✔ Run the script

```bash
python context_aware_engine_origin.py
```

### ✔ What you’ll see

* Current weather & time
* Top active context features
* Best suggestions grouped by:
  `Category > Subcategory`
* Full report saved to:
  **`suggestion_output.txt`**

---

## ⚙️ Configuration

Modify these values at the top of the script:

```python
# Default Location (Tehran)
LATITUDE = 35.6892
LONGITUDE = 51.3890

# Folder containing all JSON suggestion files
DATA_DIR = "dataset"
```

---

## 🛠️ Technical Details

### ✔ Context Vector (65+ Features)

Includes fuzzy scores for:

**Temperature:**
`extreme_cold`, `cold`, `cool`, `warm`, `hot`

**Weather:**
`clear`, `rain`, `drizzle`, `snow`, `fog`, `thunderstorm`, etc.

**Humidity:**
very_dry → very_humid

**Wind:**
calm → storm

**Time of Day:**
late_night → afternoon → evening → night

**Day Type:**
workday / weekend (Iran logic)

**Season:**
spring, summer, autumn, winter

**Events:**
romantic, festival, mourning, cultural_tradition

**Inferred Human Context:**

* mood (calm, thoughtful, relaxed, nostalgic...)
* social (solo, family, friends...)
* location (indoor, outdoor, home)
* energy (very_low → very_high)

---

### ✔ Scoring (Weighted Dot Product)

Every suggestion has `preferencesJson` like:

```json
{
  "temp_cold": 1.0,
  "weather_rain": 0.7,
  ...
}
```

The engine computes:

```
Score = Σ (Context_i × Preference_i × GroupWeight_i)
```

Where group weights reflect human priorities:

* Temperature: **1.0**
* Weather: **0.9**
* Social/Mood: **0.9**
* Time/Location: **0.8**
* Season: **0.5**
* Energy: **0.5**

---

### ✔ Veto Logic (Safety & Feasibility)

If a suggestion has a **-10.0** weight and the context strongly activates that feature:

* It is **discarded immediately**
* Example:

  * “Picnic” has `weather_thunderstorm = -10`
  * If it’s storming → never shown

---

## 📂 Data Folder Structure

```
dataset/
├── food_drink/
├── activity/
├── media/
├── clothing/
└── mood/
```

Each `.json` file contains an **array of suggestions**.

---

## 📄 License

This project is open-source and free for personal and educational use.

