"""
backend.py - AI Nutrition Assistant (Backend)

Follows the methodology diagram in the project presentation:
  User Food Input -> Food / Nutrition Dataset -> Data Processing
  -> ML / AI Analysis -> Nutrition Result -> Budget-Friendly Alternative
  -> Website Output (handled by frontend.py)

This file runs inside the browser (PyScript / Pyodide), so no server is needed.

NOTE: The values below are sample data (per 100 g, price in Rs per 100 g, approximate).
To use a real dataset (for example USDA or IFCT), replace FOOD_DATA with the
rows of your CSV file. The rest of the code stays the same.
"""

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. FOOD / NUTRITION DATASET  (per 100 g)
# ---------------------------------------------------------------------------
FOOD_DATA = [
    # food,            calories, protein, carbs, fat, fiber, price_per_100g (Rs)
    ("Almonds",           579,   21.2,  21.6, 49.9, 12.5,  90.0),
    ("Peanuts",           567,   25.8,  16.1, 49.2,  8.5,  15.0),
    ("Cashews",           553,   18.2,  30.2, 43.9,  3.3,  85.0),
    ("Walnuts",           654,   15.2,  13.7, 65.2,  6.7, 100.0),
    ("Pistachios",        560,   20.2,  27.2, 45.3, 10.6, 120.0),
    ("Roasted Chana",     364,   19.3,  60.6,  5.3, 17.0,  10.0),
    ("Sesame Seeds",      573,   17.7,  23.5, 49.7, 11.8,  20.0),
    ("Sunflower Seeds",   584,   20.8,  20.0, 51.5,  8.6,  18.0),
    ("Paneer",            265,   18.3,   1.2, 20.8,  0.0,  35.0),
    ("Tofu",               76,    8.0,   1.9,  4.8,  0.3,  12.0),
    ("Chicken Breast",    165,   31.0,   0.0,  3.6,  0.0,  28.0),
    ("Eggs",              155,   13.0,   1.1, 11.0,  0.0,  12.0),
    ("Salmon",            208,   20.0,   0.0, 13.0,  0.0,  90.0),
    ("Sardine",           208,   24.6,   0.0, 11.5,  0.0,  15.0),
    ("Quinoa",            120,    4.4,  21.3,  1.9,  2.8,  45.0),
    ("Brown Rice",        111,    2.6,  23.0,  0.9,  1.8,   8.0),
    ("Oats",              389,   16.9,  66.3,  6.9, 10.6,  12.0),
    ("Ragi",              336,    7.3,  72.0,  1.3, 11.5,   6.0),
    ("Avocado",           160,    2.0,   8.5, 14.7,  6.7,  40.0),
    ("Banana",             89,    1.1,  22.8,  0.3,  2.6,   5.0),
    ("Blueberries",        57,    0.7,  14.5,  0.3,  2.4,  70.0),
    ("Guava",              68,    2.6,  14.3,  1.0,  5.4,   6.0),
    ("Apple",              52,    0.3,  13.8,  0.2,  2.4,  15.0),
    ("Spinach",            23,    2.9,   3.6,  0.4,  2.2,   4.0),
    ("Broccoli",           34,    2.8,   6.6,  0.4,  2.6,  12.0),
    ("Moong Dal",         347,   24.0,  63.0,  1.2, 16.3,  11.0),
    ("Chia Seeds",        486,   16.5,  42.1, 30.7, 34.4,  60.0),
    ("Flax Seeds",        534,   18.3,  28.9, 42.2, 27.3,  15.0),
]

COLUMNS = ["food", "calories", "protein", "carbs", "fat", "fiber", "price"]
FEATURES = ["calories", "protein", "carbs", "fat", "fiber"]


# ---------------------------------------------------------------------------
# 2. DATA PROCESSING  (clean the data, scale the nutrients)
# ---------------------------------------------------------------------------
def load_and_clean_data() -> pd.DataFrame:
    """Load the dataset and remove duplicates, missing values and extra spaces."""
    df = pd.DataFrame(FOOD_DATA, columns=COLUMNS)
    df["food"] = df["food"].str.strip()
    df = df.drop_duplicates(subset="food")
    df = df.dropna()
    return df.reset_index(drop=True)


DF = load_and_clean_data()

# Scaling so that calories (big numbers) do not dominate protein or fat (small numbers).
_SCALER = StandardScaler()
_SCALED = _SCALER.fit_transform(DF[FEATURES])

# ---------------------------------------------------------------------------
# 3. ML / AI ANALYSIS  (K-Nearest Neighbors finds nutritionally similar foods)
# ---------------------------------------------------------------------------
_MODEL = NearestNeighbors(n_neighbors=len(DF), metric="euclidean")
_MODEL.fit(_SCALED)


# ---------------------------------------------------------------------------
# Functions used by the frontend
# ---------------------------------------------------------------------------
def get_food_list() -> list:
    """All food names for the dropdown."""
    return sorted(DF["food"].tolist())


def get_nutrition(food: str) -> dict:
    """Nutrition Result: calories, protein, carbs, fat, fiber and price."""
    row = DF[DF["food"] == food].iloc[0]
    return {
        "food": str(row["food"]),
        "calories": float(row["calories"]),
        "protein": float(row["protein"]),
        "carbs": float(row["carbs"]),
        "fat": float(row["fat"]),
        "fiber": float(row["fiber"]),
        "price": float(row["price"]),
    }


def get_alternatives(food: str, top_n: int = 3) -> list:
    """
    Budget-Friendly Alternative:
      1. The ML model finds foods with similar nutrition (nearest neighbors).
      2. Keep only the foods that are cheaper than the selected food.
      3. Return the top_n most similar cheaper foods as a list of dictionaries.
    """
    idx = int(DF.index[DF["food"] == food][0])
    selected_price = float(DF.loc[idx, "price"])

    distances, indices = _MODEL.kneighbors(_SCALED[idx].reshape(1, -1))

    results = []
    for dist, i in zip(distances[0], indices[0]):
        i = int(i)
        if i == idx:
            continue
        candidate = DF.loc[i]
        if float(candidate["price"]) < selected_price:
            # Convert the distance to a simple 0-100 similarity score.
            similarity = round(100 / (1 + float(dist)), 1)
            saving = round((1 - float(candidate["price"]) / selected_price) * 100)
            results.append(
                {
                    "food": str(candidate["food"]),
                    "calories": float(candidate["calories"]),
                    "protein": float(candidate["protein"]),
                    "carbs": float(candidate["carbs"]),
                    "fat": float(candidate["fat"]),
                    "price": float(candidate["price"]),
                    "similarity": similarity,
                    "saving": int(saving),
                }
            )
        if len(results) == top_n:
            break

    return results


def get_tip(food: str) -> str:
    """Simple health tip based on the nutrition values."""
    n = get_nutrition(food)
    tips = []
    if n["protein"] >= 15:
        tips.append("High in protein - good for muscle and energy.")
    if n["fiber"] >= 8:
        tips.append("High in fiber - good for digestion.")
    if n["fat"] >= 40:
        tips.append("High in fat (mostly healthy fats) - eat in small portions.")
    if n["calories"] >= 500:
        tips.append("Calorie dense - a small handful is enough.")
    if n["calories"] <= 60:
        tips.append("Low in calories - good for weight management.")
    if not tips:
        tips.append("A balanced food - enjoy it as part of a varied diet.")
    return " ".join(tips)
