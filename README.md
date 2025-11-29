# 🌤️ AI Contextual Suggestion Engine (Python)

A standalone Python implementation of a **fuzzy-logic recommendation system**. This engine analyzes real-time weather, time of day, and seasonal context to suggest the best activities, foods, drinks, and media for the current moment.

It is a port of the logic used in an Android application, designed to be lightweight, dependency-free, and easy to run on any machine.

## ✨ Features

- **🌍 Real-Time Weather**: Automatically fetches live weather data (Temperature, Humidity, Wind, Weather Code) from the [Open-Meteo API](https://open-meteo.com/) (No API key required).
- **🧠 Fuzzy Logic Core**: Uses fuzzy membership functions to model human perception (e.g., converting "15°C" into a mix of "Cool" and "Cold" scores).
- **📐 Vector Similarity**: Matches the user's current context vector (65+ dimensions) against a database of suggestion vectors using a **Weighted Dot Product** algorithm.
- **🚀 Zero Dependencies**: Written in pure Python using only standard libraries (`urllib`, `json`, `math`, `datetime`). No `pip install` needed!
- **📂 JSON Data Driven**: Loads suggestions dynamically from a local folder of JSON files.

## 🚀 How to Run

### Prerequisites
- Python 3.6 or higher.

### Setup
1.  Ensure the `context_aware_engine_origin.py` script is in the same directory as your data folder (or update the `DATA_DIR` path in the script).
    - Default expected path: `dataset`
2.  Run the script:

```bash
context_aware_engine_origin.py
```

### Output
The script will:
1.  Print the current weather and time to the console.
2.  Calculate the top suggestions for each subcategory.
3.  Display the results in the terminal.
4.  Save a detailed report to **`suggestion_output.txt`**.

## ⚙️ Configuration

You can customize the engine by editing the constants at the top of `context_aware_engine_origin.py`:

```python
# Location (Default: Tehran)
LATITUDE = 35.6892
LONGITUDE = 51.3890

# Data Directory
DATA_DIR = "dataset"
```

## 🛠️ Technical Details

### The Context Vector
The engine builds a **Context Vector** representing the current state of the world. It includes features like:
- **Temperature**: `extreme_cold`, `cold`, `cool`, `warm`, `hot`
- **Weather**: `clear`, `rain`, `snow`, `cloudy`, etc.
- **Time**: `morning`, `afternoon`, `evening`, `night`, etc.
- **Social/Mood**: Inferred based on time and weather (e.g., "Rainy Evening" -> `mood_calm`, `location_indoor`).

### Scoring Algorithm
Each suggestion in the database has a `preferencesJson` defining its affinity for these features (ranging from `-1.0` to `+1.0`).

The score is calculated as:
$$ Score = \sum (Weight_{group} \times \sum (Context_i \times Suggestion_i)) $$

- **Positive Match**: High Context (1.0) × High Affinity (1.0) = **+Score**
- **Negative Match**: High Context (1.0) × Negative Affinity (-1.0) = **-Score** (Penalty)
- **Veto**: If any single feature score is `< -9.0`, the suggestion is immediately discarded (Safety/Feasibility check).

## 📄 License
This project is open-source and available for educational and personal use.

