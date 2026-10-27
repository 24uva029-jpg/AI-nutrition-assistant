import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="AI Nutrition Assistant", page_icon="🥗")
st.title("🥗 AI Nutrition Assistant")
st.write("Select a food to see its nutrition and cheaper, similar alternatives.")

# Dataset + Data Processing
df = pd.read_csv("foods.csv").dropna().drop_duplicates("food").reset_index(drop=True)
features = ["calories", "protein", "carbs", "fat", "fiber"]
X = StandardScaler().fit_transform(df[features])

# ML / AI: find similar foods
model = NearestNeighbors(n_neighbors=len(df)).fit(X)

# User Food Input
food = st.selectbox("Choose a food", df["food"])
grams = st.slider("Quantity (grams)", 50, 500, 100, step=50)
idx = df.index[df["food"] == food][0]
row = df.loc[idx]
factor = grams / 100

# Nutrition Result
st.subheader(f"Nutrition for {grams} g of {food}")
c1, c2, c3 = st.columns(3)
c1.metric("Calories", f"{row.calories * factor:.0f} kcal")
c2.metric("Protein", f"{row.protein * factor:.1f} g")
c3.metric("Carbs", f"{row.carbs * factor:.1f} g")
c4, c5, c6 = st.columns(3)
c4.metric("Fat", f"{row.fat * factor:.1f} g")
c5.metric("Fiber", f"{row.fiber * factor:.1f} g")
c6.metric("Cost", f"₹{row.price * factor:.0f}")

# Budget-Friendly Alternative
_, neighbors = model.kneighbors([X[idx]])
alts = [i for i in neighbors[0] if i != idx and df.loc[i, "price"] < row.price][:3]

st.subheader("💰 Budget-friendly alternatives")
if alts:
    for i in alts:
        a = df.loc[i]
        saving = (row.price - a.price) * factor
        st.success(
            f"**{a.food}** – {a.calories * factor:.0f} kcal, "
            f"{a.protein * factor:.1f} g protein, ₹{a.price * factor:.0f} "
            f"(save ₹{saving:.0f})"
        )
else:
    st.info("This is already one of the cheapest options in its group 👍")

st.caption("Nutrition values are approximate per 100 g. Prices are sample values.")
