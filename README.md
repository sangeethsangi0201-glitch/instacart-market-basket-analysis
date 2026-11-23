# 🛒 Instacart Market Basket Analysis: Segmentation, Association Rules & Reorder Prediction

[![Open In Colab](https://google.com)](https://google.com)

An end-to-end customer analytics and machine learning pipeline built on Instacart's real-world transactional dataset (**3.4M+ orders**, **32.4M+ line items**, **206K+ users**). This project uncovers cross-sell product pairings, segments shoppers into actionable personas, and predicts next-basket item reorders.

---

## 📌 Business Objectives
1. **Behavioral & Replenishment Analysis (EDA):** Identify peak ordering windows, cart-position habits, and replenishment cycles.
2. **Cross-Sell Discovery (Market Basket Analysis):** Mine high-lift product combinations to power *"Frequently Bought Together"* recommendations.
3. **Customer Segmentation:** Group 206K+ shoppers by proportional aisle preferences to enable targeted marketing campaigns.
4. **Next-Basket Reorder Prediction:** Predict which previously purchased items a customer will reorder in their next basket to power a *"Buy It Again"* widget.

---

## 🛠️ Tech Stack & Engineering Highlights
* **Languages & Libraries:** Python, `pandas`, `NumPy`, `scikit-learn`, `mlxtend` (FP-Growth), `LightGBM`, `Matplotlib`, `Seaborn`
* **Memory Engineering:** Enforced explicit integer downcasting (`int32`, `int16`, `int8`, `category`) on ingestion across 32.4M+ rows, cutting memory usage by **~70%** (~430 MB total RAM).
* **Leak-Free Validation:** Engineered RFM and cart-interaction features strictly from `prior` orders and performed a `user_id`-level train/validation split.

---

## 📊 Key Results & Visual Insights

### 1. Exploratory Data Analysis & Shopping Habits
* **~59% Baseline Reorder Rate:** Repeat purchases dominate grocery basket volume.
* **Replenishment Cycles:** Sharp ordering spikes occur at **7 days (weekly)**, **14 days (bi-weekly)**, and **30 days (monthly)**.
* **Habitual Cart Ordering:** The **1st item added to the cart has a ~68% reorder probability**, declining steadily as impulse/discovery items are added later.

![EDA Customer Shopping Patterns](01.instacart_EDA%20%26%20DATA_prep.png)

---

### 2. Market Basket Analysis (FP-Growth Association Rules)
* Used **FP-Growth** with boolean basket encoding and filtered by **Lift > 1.2** to separate true culinary pairings from high-frequency staples like bananas.
* **Top High-Lift Rules:**
  * *Organic Cilantro → Limes* (**Lift: 4.22**, Confidence: 25.3%)
  * *Organic Yellow Onion ↔ Organic Garlic* (**Lift: 4.17**)
  * *Strawberries ↔ Raspberries* (**Lift: 3.45**)

![Association Rules Lift](02.%20Market%20Basket%20Analysis%20(Association%20Rules)(FP_Growth%26Generate%20Association%20Rule).png)

---

### 3. Customer Segmentation (PCA + K-Means Clustering)
* Constructed a row-normalized **User × Aisle matrix (`206,209 × 134`)** so clusters reflect dietary lifestyle rather than raw order volume.
* Compressed dimensionality using **PCA** and selected **k = 4 clusters** via the Elbow Method, uncovering distinct shopper personas (e.g., *Hydration & Beverage Loyalists* vs. *Everyday Fresh-Food Shoppers*).

![Customer Segmentation PCA KMeans](03.%20Customer%20Segmentation%20(PCA%20%26%20K-Means%20Clustering).png)

---

### 4. Reorder Prediction Model (LightGBM)
* Framed next-basket prediction across **8.47 million `(user_id, product_id)` pairs** (~9.78% positive reorder rate).
* **Validation ROC-AUC:** `0.8304`
* **Validation F1-Score:** `0.4340` (at an optimized probability threshold of `0.22` to handle class imbalance)
* **Top Predictive Drivers:** `uxp_orders_since_last` (recency), `days_since_prior_order`, `user_avg_days_between`, `prod_reorder_rate`, and `uxp_reorder_ratio`.

![LightGBM Feature Importance](04.%20Reorder%20Prediction%20Model%20(Feature%20Engineering%20%26%20LightGBM).png)

---

## 🚀 Future Improvements
* Incorporate consecutive order streak features (whether an item was bought in the last 1–2 orders).
* Feed the K-Means customer cluster ID from Phase 3 directly into the LightGBM model as a categorical feature.
