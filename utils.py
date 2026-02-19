import os
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# === 1. Fixed paths (aligned with swiggy.ipynb) ===

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

print(f"[utils] DATA_DIR = {DATA_DIR}")
print(f"[utils] MODELS_DIR = {MODELS_DIR}")

# === 2. Load cleaned & encoded data ===

cleaned_path = os.path.join(DATA_DIR, "cleaned_data.csv")
encoded_path = os.path.join(DATA_DIR, "encoded_data.csv")

df_clean = pd.read_csv(cleaned_path)
df_encoded = pd.read_csv(encoded_path)

# df_clean has: id, name, city, rating, rating_count, cost, cuisine (plus dropped cols already handled in notebook)

# === 3. Load fitted models & encoders ===

with open(os.path.join(MODELS_DIR, "mlb_cuisine.pkl"), "rb") as f:
    mlb = pickle.load(f)

with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODELS_DIR, "nn_model.pkl"), "rb") as f:
    nn_model = pickle.load(f)


# === 4. Helper APIs for Streamlit ===

def get_city_options():
    """List of unique cities for the sidebar dropdown."""
    return sorted(df_clean["city"].dropna().unique().tolist())


def get_cuisine_options():
    """List of unique cuisines from the MultiLabelBinarizer."""
    return sorted(mlb.classes_.tolist())


# === 5. Core recommendation function ===

def recommend(city, min_rating, max_cost, preferred_cuisines, top_n=10):
    """
    Recommendation logic aligned with swiggy.ipynb:

    1) Filter df_clean on city, min_rating, max_cost, and cuisines.
    2) On the filtered subset, use cosine-based NearestNeighbors
       over the scaled encoded features.
    3) Return a DataFrame of recommendations with similarity scores
       and numeric columns ready for display.
    """

    # 5.1 Basic masks: city, rating, cost
    rating_num = pd.to_numeric(df_clean["rating"], errors="coerce")
    cost_num = pd.to_numeric(df_clean["cost"], errors="coerce")

    mask_city = df_clean["city"] == city
    mask_rating = rating_num >= float(min_rating)
    mask_cost = cost_num <= float(max_cost)

    mask = mask_city & mask_rating & mask_cost

    # 5.2 Cuisine mask (at least one of preferred cuisines)
    if preferred_cuisines:
        cuisine_cols = [
            f"cuisine_{c}" for c in preferred_cuisines
            if f"cuisine_{c}" in df_encoded.columns
        ]
        if cuisine_cols:
            cuisine_mask = df_encoded[cuisine_cols].sum(axis=1) > 0
            mask = mask & cuisine_mask

    filtered_idx = np.where(mask.values)[0]

    if len(filtered_idx) == 0:
        # Return empty with expected columns
        return pd.DataFrame(
            columns=["id", "name", "city", "rating", "rating_count",
                     "cost", "cuisine", "similarity"]
        )

    # 5.3 Limit top_n to available rows
    request_n = min(top_n, len(filtered_idx))

    # 5.4 Similarity on encoded features (scaled)
    X_scaled_full = scaler.transform(df_encoded.values)
    X_subset = X_scaled_full[filtered_idx]

    subset_nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=request_n)
    subset_nn.fit(X_subset)

    # Centroid of subset as user profile
    user_profile = X_subset.mean(axis=0).reshape(1, -1)
    distances, local_indices = subset_nn.kneighbors(user_profile)
    global_indices = filtered_idx[local_indices[0]]

    # 5.5 Build result dataframe
    base_cols = ["id", "name", "city", "rating", "rating_count", "cost", "cuisine"]
    result = df_clean.iloc[global_indices].loc[:, base_cols].reset_index(drop=True)

    # similarity = 1 - cosine distance
    result["similarity"] = 1 - distances[0]

    # Ensure numeric versions for rating, rating_count, cost (for UI)
    result["rating"] = pd.to_numeric(result["rating"], errors="coerce")
    result["rating_count"] = pd.to_numeric(result["rating_count"], errors="coerce")
    result["cost"] = pd.to_numeric(result["cost"], errors="coerce")

    # Simple median fill for NaNs (just for display/metrics)
    if result["rating"].isna().all():
        result["rating"] = result["rating"].fillna(4.0)
    else:
        result["rating"] = result["rating"].fillna(result["rating"].median())

    if result["rating_count"].isna().all():
        result["rating_count"] = result["rating_count"].fillna(100.0)
    else:
        result["rating_count"] = result["rating_count"].fillna(result["rating_count"].median())

    if result["cost"].isna().all():
        result["cost"] = result["cost"].fillna(300.0)
    else:
        result["cost"] = result["cost"].fillna(result["cost"].median())

    return result
