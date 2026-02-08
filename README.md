# Swiggy’s Restaurant Recommendation System using Streamlit
An end‑to‑end **restaurant recommendation system** built on Swiggy‑style data.  
The project cleans and preprocesses raw restaurant data from CSV, encodes categorical features, applies **unsupervised learning (clustering / similarity)**, and exposes recommendations through an interactive **Streamlit web application**.

---

## 🧩 Project Overview

**Goal:**  
Recommend restaurants to users based on their preferences such as **city, rating, cost, and cuisines**, using only the information available in the dataset (unsupervised learning).

**Key Features**

- Data cleaning and preprocessing on raw Swiggy restaurant data  
- Encoding categorical features (city, cuisines) for ML  
- Unsupervised recommendation engine using:
  - K‑Means clustering and/or
  - Cosine‑similarity–based nearest neighbors
- Streamlit web app for interactive querying and visualization  
- Reusable artifacts: cleaned data, encoded data, and saved encoders/models

**Domain:** Recommendation Systems & Data Analytics  
**Tech stack:** Python, pandas, NumPy, scikit‑learn, Streamlit

---

## 📂 Dataset

The dataset is provided as a CSV file with the following columns:

```text
['id', 'name', 'city', 'rating', 'rating_count', 'cost', 'cuisine',
 'lic_no', 'link', 'address', 'menu']
