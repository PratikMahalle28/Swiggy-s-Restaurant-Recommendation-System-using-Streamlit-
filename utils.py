import subprocess
import sys
import os

# Auto-fix imports
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'sklearn': 'scikit-learn'
}

for name, pkg in packages.items():
    try:
        if name == 'pandas':
            import pandas as pd
        elif name == 'numpy':
            import numpy as np
        elif name == 'sklearn':
            from sklearn.neighbors import NearestNeighbors
            import pickle
    except ImportError:
        install(pkg)

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle

# FIXED PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

print(f"Data: {DATA_DIR}")
print(f"Models: {MODELS_DIR}")

# Load data
df_clean = pd.read_csv(os.path.join(DATA_DIR, "cleaned_data.csv"))
df_encoded = pd.read_csv(os.path.join(DATA_DIR, "encoded_data.csv"))

with open(os.path.join(MODELS_DIR, "mlb_cuisine.pkl"), "rb") as f:
    mlb = pickle.load(f)
with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(MODELS_DIR, "nn_model.pkl"), "rb") as f:
    nn_model = pickle.load(f)

def get_city_options():
    return sorted(df_clean["city"].dropna().unique().tolist())

def get_cuisine_options():
    return sorted(mlb.classes_.tolist())

def recommend(city, min_rating, max_cost, preferred_cuisines, top_n=10):
    """100% SAFE - no index errors"""
    
    # Create masks
    mask_city = df_clean["city"] == city
    mask_rating = df_clean["rating_clean"] >= min_rating
    mask_cost = df_clean["cost_clean"] <= max_cost
    
    mask = mask_city & mask_rating & mask_cost
    
    # Cuisine filter
    if preferred_cuisines:
        cuisine_cols = [f"cuisine_{c}" for c in preferred_cuisines if f"cuisine_{c}" in df_encoded.columns]
        if cuisine_cols:
            cuisine_mask = df_encoded[cuisine_cols].sum(axis=1) > 0
            mask = mask & cuisine_mask
    
    # Get filtered indices SAFELY
    filtered_mask = mask.values  # boolean array
    filtered_idx = np.nonzero(filtered_mask)[0]  # integer indices
    
    if len(filtered_idx) == 0:
        return pd.DataFrame(columns=["id", "name", "city", "rating", "rating_count", "cost", "cuisine"])
    
    num_available = len(filtered_idx)
    request_n = min(top_n, num_available)
    
    # Scale
    X_scaled = scaler.transform(df_encoded.values)
    X_subset = X_scaled[filtered_idx]
    
    # Create TEMP NearestNeighbors for subset (AVOIDS global model index issues)
    subset_nn = NearestNeighbors(n_neighbors=request_n, metric='cosine')
    subset_nn.fit(X_subset)
    
    # User profile = average of filtered restaurants
    user_profile = X_subset.mean(axis=0).reshape(1, -1)
    
    # Find similar
    distances, local_indices = subset_nn.kneighbors(user_profile)
    
    # Map back to global indices
    global_indices = filtered_idx[local_indices[0]]
    
    # Get results
    cols = ["id", "name", "city", "rating", "rating_count", "cost", "cuisine"]
    result = df_clean.iloc[global_indices].loc[:, cols].reset_index(drop=True)
    result['similarity'] = 1 - distances[0]
    
    return result
