import streamlit as st
import pandas as pd  # ← ADDED THIS
import subprocess
import sys

from utils import get_city_options, get_cuisine_options, recommend

st.set_page_config(page_title="🍽️ Swiggy Recommendations", layout="wide")
st.title("🍽️ Swiggy Restaurant Recommendations")
st.markdown("**Unsupervised ML**: Similarity Matching")

with st.sidebar:
    st.header("🔍 Preferences")
    cities = get_city_options()
    city = st.selectbox("City", cities)
    
    min_rating = st.slider("Min Rating", 0.0, 5.0, 3.5)
    max_cost = st.slider("Max Cost ₹", 50, 1000, 400)
    
    cuisines = get_cuisine_options()
    selected_cuisines = st.multiselect("Cuisines", cuisines)
    
    top_n = st.slider("Show Top", 5, 20, 10)
    
    if st.button("🚀 Get Recommendations", type="primary"):
        with st.spinner("Matching restaurants..."):
            results = recommend(city, min_rating, max_cost, selected_cuisines, top_n)
        
        if results.empty:
            st.error("❌ No matches. Try: lower rating, higher budget, or fewer cuisines.")
        else:
            st.success(f"✅ Found **{len(results)}** restaurants!")
            
            # Safe metrics
            ratings_numeric = pd.to_numeric(results['rating'], errors='coerce').dropna()
            costs_numeric = pd.to_numeric(results['cost'].str.replace('₹', '').str.replace(',', ''), errors='coerce').dropna()
            
            col1, col2, col3 = st.columns(3)
            if len(ratings_numeric) > 0:
                col1.metric("Avg Rating", f"{ratings_numeric.mean():.1f}")
            if len(costs_numeric) > 0:
                col2.metric("Avg Cost", f"₹{int(costs_numeric.mean())}")
            col3.metric("Best Match", f"{results['similarity'].max():.1%}")
            
            st.dataframe(
                results.style.format({'similarity': '{:.1%}'}),
                use_container_width=True
            )

st.markdown("---")
st.info("🎯 MultiLabelBinarizer + Cosine Similarity | Swiggy Dataset")
