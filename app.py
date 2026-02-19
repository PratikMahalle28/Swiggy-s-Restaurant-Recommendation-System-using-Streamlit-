import streamlit as st
import pandas as pd

from utils import get_city_options, get_cuisine_options, recommend

st.set_page_config(page_title="🍽️ Swiggy Recommendations", layout="wide")

# === HEADER: title + found count ===
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("🍽️ Swiggy Restaurant Recommendations")
with header_col2:
    if "found_count" in st.session_state and st.session_state.found_count > 0:
        st.success(f"✅ Found **{st.session_state.found_count}** restaurants!", icon="⭐")

# === SIDEBAR: user preferences ===
with st.sidebar:
    st.header("🔍 Preferences")

    cities = get_city_options()
    city = st.selectbox("City", cities)

    min_rating = st.slider("Min Rating", 0.0, 5.0, 3.5, 0.1)
    max_cost = st.slider("Max Cost ₹", 50, 2000, 400, 50)

    cuisines = get_cuisine_options()
    selected_cuisines = st.multiselect("Cuisines", cuisines)

    top_n = st.slider("Show Top", 5, 30, 10)

    if st.button("🚀 Get Recommendations", type="primary"):
        with st.spinner("Matching restaurants..."):
            results = recommend(
                city=city,
                min_rating=min_rating,
                max_cost=max_cost,
                preferred_cuisines=selected_cuisines,
                top_n=top_n,
            )
            st.session_state.found_count = len(results)
            st.session_state.results = results

        st.rerun()

# === MAIN CONTENT ===
if "found_count" in st.session_state and st.session_state.found_count > 0:
    results = st.session_state.results

    if results.empty:
        st.error("❌ No matches. Try: lower rating, higher budget, or fewer cuisines.")
    else:
        # Ensure numeric types for metrics
        ratings_numeric = pd.to_numeric(results["rating"], errors="coerce")
        rating_counts_numeric = pd.to_numeric(results["rating_count"], errors="coerce")
        costs_numeric = pd.to_numeric(results["cost"], errors="coerce")

        # Fallback fills for metrics only
        if ratings_numeric.isna().all():
            ratings_numeric = ratings_numeric.fillna(4.0)
        else:
            ratings_numeric = ratings_numeric.fillna(ratings_numeric.median())

        if rating_counts_numeric.isna().all():
            rating_counts_numeric = rating_counts_numeric.fillna(100.0)
        else:
            rating_counts_numeric = rating_counts_numeric.fillna(rating_counts_numeric.median())

        if costs_numeric.isna().all():
            costs_numeric = costs_numeric.fillna(300.0)
        else:
            costs_numeric = costs_numeric.fillna(costs_numeric.median())

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rating", f"{ratings_numeric.median():.1f}")
        col2.metric("Rating Count", f"{int(rating_counts_numeric.median())}")
        col3.metric("Avg Cost", f"₹{int(costs_numeric.mean())}")
        col4.metric("Best Match Similarity", f"{results['similarity'].max():.1%}")

        # Display table
        display_df = results[
            ["name", "city", "rating", "rating_count", "cost", "cuisine", "similarity"]
        ].copy()

        # Round numeric columns for nicer UI
        display_df["rating"] = pd.to_numeric(display_df["rating"], errors="coerce").fillna(
            ratings_numeric.median()
        )
        display_df["rating_count"] = pd.to_numeric(display_df["rating_count"], errors="coerce").fillna(
            rating_counts_numeric.median()
        )
        display_df["cost"] = pd.to_numeric(display_df["cost"], errors="coerce").fillna(
            costs_numeric.median()
        )

        display_df["rating"] = display_df["rating"].round(1)
        display_df["rating_count"] = display_df["rating_count"].round(0).astype(int)
        display_df["cost"] = display_df["cost"].round(0).astype(int)

        st.dataframe(
            display_df.style.format(
                {
                    "similarity": "{:.1%}",
                    "rating": "{:.1f}",
                    "cost": "₹{:,.0f}",
                }
            ),
            use_container_width=True,
        )
else:
    st.info("👈 Select preferences and click Get Recommendations")


