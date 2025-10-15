import os
import json
import io
import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import shap
from scipy.stats import ks_2samp

# ----------------------------
# Paths & lazy loading
# ----------------------------
REPO_DIR = os.path.dirname(os.path.dirname(__file__))
ART_DIR = os.path.join(REPO_DIR, "app", "artifacts")
PIPE_PATH = os.path.join(ART_DIR, "inference_pipeline.joblib")
COLS_PATH = os.path.join(ART_DIR, "expected_columns.json")
REF_PATH = os.path.join(ART_DIR, "reference_sample.csv")
GRAPHS_DIR = os.path.join(REPO_DIR, "graphs")  # <— put all images here
DEFAULT_DATASET = "/mnt/data/voltas_featured.csv"  # <— attached dataset path
EXP_CSV_PATH = os.path.join(ART_DIR, "experiments.csv")  # optional experiments table

LABEL_MAP = {0: "In Stock", 1: "Out of Stock"}

# ----------------------------
# Friendly feature descriptions
# ----------------------------
FEATURE_DESCRIPTIONS = {
    "product_id": "Unique identifier for each Voltas product.",
    "product_category": "Broad category of the product (e.g., AC, Refrigerator, Cooler).",
    "sub_type": "Specific type within the category (e.g., Split AC, Window AC).",
    "model_name": "Product's model name or number.",
    "capacity_tons": "Cooling capacity of the product in tons (used for ACs).",
    "capacity_liters": "Volume capacity in liters (used for refrigerators, coolers, etc.).",
    "capacity_kg": "Weight capacity (used for washing machines, dryers, etc.).",
    "capacity_place_settings": "Number of place settings (used for dishwashers).",
    "technology": "Type of technology used (e.g., Inverter, Non-Inverter, Smart, Eco Mode).",
    "feature_1": "Highlighted feature or key selling point of the product.",
    "energy_rating_stars": "Energy efficiency rating (1–5 stars).",
    "color": "Color variant of the product.",
    "price_inr": "Selling price in Indian Rupees.",
    "manufacturing_date": "Date the product was manufactured.",
    "warranty_years": "Warranty period in years.",
    "customer_rating": "Average customer rating (scale 1–5).",
    "city": "City where the product was sold or reviewed.",
    "platform": "E-commerce or retail platform (e.g., Amazon, Flipkart, Croma).",
    "discount_offered": "Percentage or amount of discount applied.",
    "availability": "Stock availability status (e.g., In Stock, Out of Stock).",
    "warranty_duration_months": "Total warranty period in months (derived or separate from years).",
    "review_sentiment": "Sentiment of the customer review (e.g., Positive, Neutral, Negative).",
    "return_status": "Indicates if the product was returned (Yes/No).",
    "complaint_text": "Customer complaint or issue reported, if any.",
    "resolved_status": "Whether the complaint was resolved (Resolved/Unresolved/Pending).",
    "review_date": "Date the product was reviewed.",
    "reviewer_location": "Location of the customer who reviewed the product.",
    "product_name": "Full commercial name of the product.",
    "username": "Anonymized identifier of the reviewer or customer.",
}

@st.cache_resource(show_spinner=False)
def load_artifacts():
    load_error = None
    pipeline, expected_cols, ref = None, None, None
    try:
        import joblib
        pipeline = joblib.load(PIPE_PATH)
        expected_cols = json.load(open(COLS_PATH))["expected_input_cols"]
    except Exception as e:
        load_error = f"Artifact load failed: {e}"

    if os.path.exists(REF_PATH):
        try:
            ref = pd.read_csv(REF_PATH)
        except Exception as e:
            load_error = f"Reference sample load failed: {e}"

    return pipeline, expected_cols, ref, load_error


inference_pipeline, EXPECTED_COLS, REF, LOAD_ERR = load_artifacts()

st.set_page_config(page_title="Voltas Availability Dashboard", layout="wide")
st.title("📊 Voltas Availability — Predictions & Story-Driven Insights")
st.caption("A storytelling dashboard that moves from context ➜ data ➜ EDA ➜ modeling ➜ XAI ➜ fairness ➜ monitoring.")

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])

    # Image size control (applies to plots/images across tabs)
    img_width = st.slider("Image width (px)", min_value=480, max_value=1000, value=720, step=20)

    detected_sensitive = "city"
    detected_target = ""
    if uploaded is not None:
        try:
            _tmp_df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            cand_sens = [
                c for c in _tmp_df.columns
                if (pd.api.types.is_object_dtype(_tmp_df[c]) or pd.api.types.is_categorical_dtype(_tmp_df[c]))
                and _tmp_df[c].nunique() <= 30
            ]
            for pref in ["city", "platform", "reviewer_location"]:
                if pref in cand_sens:
                    detected_sensitive = pref
                    break
            if not cand_sens:
                detected_sensitive = "city"
            elif detected_sensitive not in cand_sens:
                detected_sensitive = cand_sens[0]

            for pref_t in ["availability", "is_out_of_stock", "target", "label"]:
                if pref_t in _tmp_df.columns:
                    detected_target = pref_t
                    break
        except Exception:
            pass

    sensitive_attr = st.text_input("Sensitive attribute (grouping column)", value=detected_sensitive)
    target_attr = st.text_input("Ground-truth column (optional)", value=detected_target)

    threshold = st.slider("Probability threshold for 'Out of Stock'", 0.0, 1.0, 0.5, 0.01)
    st.divider()
    st.write("Artifacts status:", "`OK`" if inference_pipeline else f"`Degraded: {LOAD_ERR}`")
    st.write("Graphs folder:", f"`{GRAPHS_DIR}`")

# ----------------------------
# Helpers
# ----------------------------
def align_columns(df: pd.DataFrame, expected_cols):
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    return df[expected_cols]

def predict_df(df: pd.DataFrame):
    model = getattr(inference_pipeline, "named_steps", {}).get("model", None)
    proba = None
    if hasattr(inference_pipeline, "predict_proba"):
        proba = inference_pipeline.predict_proba(df)[:, 1]
    elif model is not None and hasattr(model, "predict_proba"):
        proba = model.predict_proba(inference_pipeline.named_steps["preprocess"].transform(df))[:, 1]
    preds = inference_pipeline.predict(df)
    return preds, proba

def psi(reference: pd.Series, current: pd.Series, bins: int = 10):
    ref = reference.dropna().astype(float)
    cur = current.dropna().astype(float)
    if len(ref) < 10 or len(cur) < 10:
        return np.nan
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(ref, quantiles)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    ref_counts = np.histogram(ref, bins=edges)[0]
    cur_counts = np.histogram(cur, bins=edges)[0]
    ref_perc = np.where(ref_counts == 0, 1e-6, ref_counts) / max(1, ref_counts.sum())
    cur_perc = np.where(cur_counts == 0, 1e-6, cur_counts) / max(1, cur_counts.sum())
    return float(np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc)))

def list_graph_images(patterns=("*.png","*.jpg","*.jpeg","*.webp")):
    imgs = []
    if os.path.isdir(GRAPHS_DIR):
        for pat in patterns:
            imgs.extend(glob.glob(os.path.join(GRAPHS_DIR, pat)))
    return sorted(imgs)

def load_dataset_for_info():
    """Priority: uploaded -> DEFAULT_DATASET -> REF -> None"""
    if uploaded is not None:
        try:
            return pd.read_csv(io.BytesIO(uploaded.getvalue())), "uploaded"
        except Exception:
            pass
    if os.path.exists(DEFAULT_DATASET):
        try:
            return pd.read_csv(DEFAULT_DATASET), "default_path"
        except Exception:
            pass
    if REF is not None:
        return REF.copy(), "reference_sample"
    return None, "none"

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def show_centered_image(path, caption=None, width=720):
    """Center + scale images without using deprecated args."""
    if not path:
        return
    left, mid, right = st.columns([1,3,1])
    with mid:
        st.image(path, caption=caption, width=width)

# ----------------------------
# Tabs 
# ----------------------------
tab_problem, tab_dataset, tab_eda, tab_experiments, tab_pred, tab_shap_img, tab_fair, tab_drift = st.tabs(
    [
        "🧩 Problem",
        "🗂️ Dataset",
        "📈 EDA",
        "🧪 ML Experiments",
        "🔮 Predict",
        "🔎 SHAP",
        "⚖️ Fairness",
        "🌊 Drift",
    ]
)

# ----------------------------
# Problem Statement tab
# ----------------------------
with tab_problem:
    st.subheader("Project Problem Statement")
    st.markdown(
        """
**Voltas** operates in a competitive consumer electronics market. While it enjoys strong offline trust,
online behavior on platforms like Instagram, YouTube and MouthShut increasingly shapes **brand perception** and **sales**.

We analyze **digital performance & audience engagement** using data scraped via e-commerce + video APIs:
product descriptions, prices, reviews, ratings, promo views, and engagement metrics (likes, comments, shares).

**Why this matters**
- Digital signals (reviews, ratings, unboxing videos, ad campaigns) now influence purchase journeys.
- Understanding which levers drive **visibility**, **sentiment**, and **availability (stockouts)** improves planning and marketing ROI.

**This dashboard tells a story in chapters**
1. **Meet the data** → what features we have and how they look.  
2. **Explore** → quick visuals to understand distributions, outliers, and relationships.  
3. **Model** → which algorithms worked best and why we chose the winner.  
4. **Explain** → SHAP reveals *why* the model predicts stockouts.  
5. **Check** → Fairness by groups (e.g., city/platform).  
6. **Monitor** → Drift vs. a reference sample to know when to retrain.

**Methods used**
- Exploratory Data Analysis (EDA)  
- Sentiment signals from reviews  
- Trend/seasonality in engagement  
- Comparative analysis (by platform & category)  
- Visual storytelling with dashboards, graphs, and heatmaps
"""
    )

    st.markdown("#### Full narrative (paste/edit)")
    st.text_area(
        "Problem narrative",
        height=260,
        value=(
            "Voltas, a leading Indian brand in air conditioning and cooling solutions, operates in a highly "
            "competitive consumer electronics market. While the brand enjoys strong offline retail visibility and "
            "legacy trust, the shift towards online consumer behavior—particularly on platforms like Instagram and "
            "Mouthshut—has made it critical to understand how digital engagement impacts brand performance and "
            "customer perception.\n\n"
            "With consumers increasingly researching products online before purchasing, elements such as product "
            "reviews, ratings, unboxing videos, and ad campaigns contribute significantly to brand visibility and "
            "influence. Voltas' online presence includes product listings, YouTube advertisements, influencer content, "
            "and customer feedback—all of which can impact sales and reputation.\n\n"
            "This project aims to analyze the digital performance and audience engagement of Voltas through a dataset "
            "collected via web scraping and APIs from e-commerce and video-sharing platforms. The dataset includes "
            "product descriptions, prices, reviews, ratings, number of views on promotional content, and engagement "
            "metrics like likes, comments, and shares.\n\n"
            "To derive actionable insights, the project applies: EDA, sentiment analysis of reviews, trend/seasonality "
            "detection, competitive comparisons, and visual storytelling. The goal is to identify which digital factors "
            "drive positive audience sentiment and product visibility—supporting smarter marketing and improved customer "
            "satisfaction."
        ),
    )

# ----------------------------
# Dataset tab
# ----------------------------
with tab_dataset:
    st.subheader("Dataset Overview & Feature Glossary")

    data, source = load_dataset_for_info()
    if data is None or data.empty:
        st.warning("No dataset found. Upload a CSV or place it at `/mnt/data/voltas_featured.csv`.")
    else:
        st.caption(f"Loaded from: **{source}**")
        st.write(f"Shape: **{data.shape[0]} rows × {data.shape[1]} columns**")
        st.dataframe(data.head(10), use_container_width=True)

        def sample_val(s):
            try:
                return s.dropna().iloc[0]
            except Exception:
                return None

        # feature summary (compact, just the facts)
        summary_rows = []
        for c in data.columns:
            dtype = str(data[c].dtype)
            miss = int(data[c].isna().sum())
            u = int(data[c].nunique())
            ex = sample_val(data[c])
            summary_rows.append(
                {"feature": c, "dtype": dtype, "missing": miss, "unique": u, "example": ex}
            )
        st.markdown("#### Feature summary")
        st.dataframe(pd.DataFrame(summary_rows).sort_values("feature"), use_container_width=True)

        # data dictionary (your provided descriptions)
        st.markdown("#### Data dictionary")
        st.dataframe(
            pd.DataFrame(
                [{"Column Name": k, "Description": v} for k, v in FEATURE_DESCRIPTIONS.items()]
            ),
            use_container_width=True,
        )

        # simple target distribution if present
        target_guess = None
        for t in ["availability", "is_out_of_stock", "target", "label"]:
            if t in data.columns:
                target_guess = t
                break
        if target_guess:
            st.markdown("#### Target distribution (if applicable)")
            td = (
                data[target_guess]
                .astype(str).str.lower()
                .map({"in stock": 0, "instock": 0, "available": 0, "0": 0,
                      "out of stock": 1, "outofstock": 1, "unavailable": 1, "1": 1})
                .fillna(-1)
            )
            vc = td.value_counts().rename(index={0: "In Stock (0)", 1: "Out of Stock (1)", -1: "Unknown/Other"})
            st.dataframe(vc.to_frame("count"), use_container_width=True)

# ----------------------------
# EDA 
# ----------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis — fixed order")

    SECTIONS = [
        {
            "title": "1) Distribution of Product Categories",
            "paths": [
                "graphs/distribution_of_product_categories.png",
                "graphs/distribution.png",
                "graphs/categories_distribution.png",
            ],
            "points_md": """
**Key Observations**
- **Refrigerators dominate** — highest count (~100 products).
- **Dishwashers & Water Dispensers** — second tier (~80–85 each), balanced representation.
- **Washing Machines, Air Coolers, Air Conditioners** — slightly fewer each (~75–78).
- **Overall balance** — categories fall in the 75–100 range, good for modeling without heavy class rebalancing.
""",
        },
        {
            "title": "2) Histograms of Numeric Features",
            "paths": ["graphs/histogram.png", "graphs/histograms.png"],
            "points_md": """
**Feature-wise Inferences**
- **price_inr** — fairly uniform from ~₹10k to ₹80k → budget to premium covered.
- **capacity_tons** — strong peak at **1.5**; smaller at **1.0** and **2.0**.
- **capacity_liters** — right-skewed; most near **~50 L**, few very large (200–550 L).
- **capacity_kg** — peak at **7 kg**; smaller at **6** and **9**.
- **energy_rating_stars** — spread across 1–5, peaks at **3★** and **5★**.
- **customer_rating** — mostly 3.0–5.0 (peak 4.0–4.5) → generally positive.
- **price_density_score** — centered around **30k–40k**.
- **warranty_quality_interaction** — right-skewed; many low (3–10), fewer up to 50.

**Overall**
- Clear clustering in capacities; wide price coverage; positive rating skew; variety in energy efficiency.
""",
        },
        {
            "title": "3) Box Plots (Outliers & Spread)",
            "paths": ["graphs/boxplot.png", "graphs/box_plot.png"],
            "points_md": """
**Highlights**
- **price_inr** — median ~₹50k, wide IQR (₹25k–₹70k), a few very low outliers (₹5k–₹10k).
- **capacity_tons** — mostly **1.5**; outliers at **1.0** and **2.0**.
- **capacity_liters** — high outliers (200–550 L), bulk near ~50 L.
- **capacity_kg** — tight at **7 kg**; small outliers at **6** and **9**.
- **energy_rating_stars** — even spread, median **3★**.
- **customer_rating** — median ~4.0; IQR 3.5–4.5; few <3.
- **price_density_score** — median ~30k; wide IQR; outliers low (~5k) & high (~80k).
- **warranty_quality_interaction** — median ~10; right-skewed; outliers up to 50.
""",
        },
        {
            "title": "4) Correlation Matrix of Numerical Features",
            "paths": [
                "graphs/correlation_matrix.png",
                "graphs/corelation_matrix.png",
                "graphs/corr_matrix.png",
            ],
            "points_md": """
**Strong Positives**
- `warranty_years` ↔ `warranty_duration_months` (~0.94) → redundant.
- `warranty_years` ↔ `warranty_quality_interaction` (~0.97) → avoid double counting.
- `price_inr` ↔ `price_density_score` (~0.98) → near-duplicates.

**Moderate / Weak**
- `customer_rating` ↔ `warranty_quality_interaction` (~0.23) → mild link.
- `discount_offered` weakly related to others (<0.1).
- Capacity vs price mostly weak → pricing driven by brand/tech/features more than size.
""",
        },
    ]

    for sec in SECTIONS:
        st.markdown(f"### {sec['title']}")
        img_path = first_existing(sec["paths"])
        if img_path:
            show_centered_image(img_path, caption=os.path.basename(img_path), width=img_width)
        else:
            st.warning(f"Image not found at any of: {', '.join(sec['paths'])}")
        st.markdown(sec["points_md"])
        st.divider()

# ----------------------------
# ML Experiments tab 
# ----------------------------
with tab_experiments:
    st.subheader("ML Modeling & Experiment Tracking")

    # VALIDATION COMPARISON (given numbers)
    val_rows = [
        {"model": "RandomForest", "pr_auc_pos": 0.600866, "f1_pos": 0.611765, "roc_auc": 0.579416, "accuracy": 0.560000, "cm_TN": 16, "cm_FP": 20, "cm_FN": 13, "cm_TP": 26},
        {"model": "LogisticRegression", "pr_auc_pos": 0.583544, "f1_pos": 0.500000, "roc_auc": 0.512821, "accuracy": 0.466667, "cm_TN": 15, "cm_FP": 21, "cm_FN": 19, "cm_TP": 20},
        {"model": "LinearSVC+Calibrated", "pr_auc_pos": 0.524325, "f1_pos": 0.641509, "roc_auc": 0.480769, "accuracy": 0.493333, "cm_TN": 3,  "cm_FP": 33, "cm_FN": 5,  "cm_TP": 34},
        {"model": "RBF SVC (prob=True)", "pr_auc_pos": 0.469474, "f1_pos": 0.583333, "roc_auc": 0.418803, "accuracy": 0.466667, "cm_TN": 7,  "cm_FP": 29, "cm_FN": 11, "cm_TP": 28},
    ]
    st.markdown("#### Validation comparison (per model)")
    st.dataframe(pd.DataFrame(val_rows), use_container_width=True)

    st.markdown("#### Test metrics (best model from validation)")
    st.dataframe(pd.DataFrame([{
        "model (TEST)": "RandomForest",
        "f1_pos": 0.522727, "pr_auc_pos": 0.46099, "roc_auc": 0.357143, "accuracy": 0.44,
        "cm_TN": 10, "cm_FP": 25, "cm_FN": 17, "cm_TP": 23
    }]), use_container_width=True)

    # Threshold tuning (story: how we chose the operating point)
    st.markdown("#### Threshold tuning")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Threshold @ max F1 (validation)", "0.440", help="F1 = 0.712")
    with c2:
        st.metric("Threshold @ min cost (FP=1, FN=5)", "0.320", help="Business cost = 33.0")
    with c3:
        st.metric("Chosen operating threshold", "0.440")

    # Leaderboard by PR-AUC
    leaderboard = pd.DataFrame([
        {"model": "stockout_randomforest.pkl",           "pr_auc_pos": 0.600866, "roc_auc": 0.579416, "f1_pos@0.5": 0.611765, "accuracy@0.5": 0.560000, "cm_TN": 16, "cm_FP": 20, "cm_FN": 13, "cm_TP": 26},
        {"model": "stockout_logisticregression.pkl",     "pr_auc_pos": 0.583544, "roc_auc": 0.512821, "f1_pos@0.5": 0.500000, "accuracy@0.5": 0.466667, "cm_TN": 15, "cm_FP": 21, "cm_FN": 19, "cm_TP": 20},
        {"model": "stockout_linearsvcpluscalibrated.pkl","pr_auc_pos": 0.524325, "roc_auc": 0.480769, "f1_pos@0.5": 0.641509, "accuracy@0.5": 0.493333, "cm_TN": 3,  "cm_FP": 33, "cm_FN": 5,  "cm_TP": 34},
    ])
    st.markdown("#### Validation leaderboard (picked by PR-AUC)")
    st.dataframe(leaderboard.sort_values("pr_auc_pos", ascending=False), use_container_width=True)
    st.success("Chosen winner: **stockout_randomforest.pkl**  |  Tuned threshold on validation: **0.440**")

    # Test metrics at tuned threshold
    st.markdown("#### Test metrics (winner @ tuned threshold)")
    st.dataframe(pd.DataFrame([{
        "model": "stockout_randomforest.pkl", "threshold": 0.44,
        "PR-AUC_test": 0.46099, "ROC-AUC_test": 0.357143,
        "F1_pos_test": 0.594059, "Accuracy_test": 0.453333,
        "TN": 4, "FP": 31, "FN": 10, "TP": 30
    }]), use_container_width=True)

    st.divider()
    st.markdown("### Plots & explanations")

    # PR Curve (Validation)
    pr_val_path = first_existing(["graphs/pr_curve_validation.png", "/mnt/data/pr_curve_validation.png"])
    if pr_val_path:
        show_centered_image(pr_val_path, caption=os.path.basename(pr_val_path), width=img_width)
    st.markdown("""
**Precision–Recall (Validation)**
- *X (Recall):* of all **real stockouts**, how many did we catch?  
- *Y (Precision):* of all **predicted stockouts**, how many were correct?  
- **Takeaway:** moderate ability to identify stockouts (**AP ≈ 0.584**).
""")

    # PR Curve (Test)
    pr_test_path = first_existing(["graphs/pr_curve_testing.png", "/mnt/data/pr_curve_testing.png"])
    if pr_test_path:
        show_centered_image(pr_test_path, caption=os.path.basename(pr_test_path), width=img_width)
    st.markdown("""
**Precision–Recall (Test)**
- Same interpretation as validation.  
- **Takeaway:** reasonable but weaker (**AP ≈ 0.461**).
""")

    # Confusion Matrix (Validation)
    cm_val_path = first_existing(["graphs/confusion_matrix_validation.png", "/mnt/data/confusion_matrix_validation.png"])
    if cm_val_path:
        show_centered_image(cm_val_path, caption=os.path.basename(cm_val_path), width=img_width)
    st.markdown("""
**Validation Confusion Matrix**
- Correctly predicted ~**33** stockouts; missed **~6** (FN).  
- **~25** In-Stock cases flagged as stockouts (FP) → more false alarms.  
- Model leans toward predicting **OutOfStock** to catch more true stockouts.
""")

    # Confusion Matrix (Test)
    cm_test_path = first_existing(["graphs/confusion_matrix_testing.png", "/mnt/data/confusion_matrix_testing.png"])
    if cm_test_path:
        show_centered_image(cm_test_path, caption=os.path.basename(cm_test_path), width=img_width)
    st.markdown("""
**Test Confusion Matrix**
- Catches most stockouts (e.g., **~32 TP**) but ~**28 FP** on In-Stock items.  
- Similar to validation: **good recall** for stockouts, weaker on In-Stock.  
- Threshold tuning balances early alerts vs false alarms.
""")

# ----------------------------
# Predict tab
# ----------------------------
with tab_pred:
    st.subheader("Batch predictions from CSV (or try a single record)")
    if inference_pipeline is None or EXPECTED_COLS is None:
        st.warning("Artifacts not loaded. Check /app/artifacts files.")
    else:
        df_in = None
        if uploaded is not None:
            try:
                df_in = pd.read_csv(io.BytesIO(uploaded.getvalue()))
                if df_in.empty:
                    st.error("Uploaded CSV is empty.")
                    df_in = None
                else:
                    st.dataframe(df_in.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                df_in = None
        else:
            st.info("Upload a CSV or use a minimal single record below.")
            demo = {
                "product_id": "AC-001",
                "price_inr": 40990,
                "city": "Mumbai",
                "platform": "Croma",
                "energy_rating_stars": 5,
                "warranty_years": 2,
                "capacity_unified": 1.5,
                "price_density_score": 0.62,
                "warranty_quality_interaction": 1.24,
            }
            df_in = pd.DataFrame([demo])

        if df_in is not None:
            al = align_columns(df_in.copy(), EXPECTED_COLS)
            preds, proba = predict_df(al)
            out = pd.DataFrame(
                {"prediction": preds.astype(int),
                 "label": [LABEL_MAP.get(int(p), str(p)) for p in preds],
                 "proba_out_of_stock": proba if proba is not None else np.nan}
            )
            st.success(f"Predicted {len(out)} rows.")
            st.dataframe(out.head(50), use_container_width=True)

            gt_col = None
            for c in ["availability", "Availability", "is_out_of_stock", "target"]:
                if c in df_in.columns:
                    gt_col = c
                    break
            if gt_col:
                y_true_raw = (
                    df_in[gt_col].astype(str).str.lower().str.strip().map(
                        {"in stock": 0, "instock": 0, "available": 0, "0": 0, "false": 0, "no": 0,
                         "out of stock": 1, "outofstock": 1, "unavailable": 1, "1": 1, "true": 1, "yes": 1}
                    )
                )
                y_true = y_true_raw.fillna(0).astype(int).values
                y_pred = (
                    (out["proba_out_of_stock"].fillna(0.0).values >= threshold).astype(int)
                    if proba is not None else out["prediction"].values
                )
                st.write("**Metrics**")
                report_dict = classification_report(y_true, y_pred, output_dict=True, digits=4)
                report_df = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "label"})
                cols = [c for c in ["label", "precision", "recall", "f1-score", "support"] if c in report_df.columns]
                st.dataframe(report_df[cols], use_container_width=True)

                cm = confusion_matrix(y_true, y_pred)
                cm_df = pd.DataFrame(cm,
                                     index=["True 0 (In Stock)", "True 1 (Out of Stock)"],
                                     columns=["Pred 0", "Pred 1"])
                st.write("**Confusion matrix**")
                st.dataframe(cm_df, use_container_width=True)

# ----------------------------
# SHAP (XAI) 
# ----------------------------
with tab_shap_img:
    st.subheader("Explainable AI (XAI) — SHAP")

    st.markdown("""
We applied **SHAP (SHapley Additive exPlanations)** to interpret the trained models.

- For **Random Forest**, we used **TreeExplainer** in **interventional** mode with a sampled training background.
- We produced **global** explanations: a **summary/beeswarm** plot for average impact and a **dependence/interaction**
  view to see how value changes affect **“Out of Stock”** likelihood.

This surfaces which features drive availability predictions and adds transparency to model decisions.
""")

    # 1) Random Forest (interventional)
    st.markdown("### 1) Random Forest (SHAP — interventional)")
    rf_intervention_path = first_existing(["graphs/shap_rf_interventional.png", "graphs/shap_rf.png", "/mnt/data/shap_rf.png"])
    if rf_intervention_path:
        show_centered_image(rf_intervention_path, caption=os.path.basename(rf_intervention_path), width=img_width)
    else:
        st.info("Add the RF interventional SHAP image at `graphs/shap_rf_interventional.png` (or `/mnt/data/shap_rf.png`).")
    st.markdown("""
**Inference**
- Only **capacity_liters** and **capacity_tons** are highlighted with values near zero → limited influence.
- Minimal interaction spread → these fields are not strong signals for RF in this dataset.
""")

    # 2) Random Forest (additivity disabled) — optional
    st.markdown("### 2) Random Forest (SHAP with additivity check disabled)")
    rf_addoff_path = first_existing(["graphs/shap_rf_additivity_off.png", "graphs/shap_rf_bigvals.png", "graphs/shap_rf_addoff.png"])
    if rf_addoff_path:
        show_centered_image(rf_addoff_path, caption=os.path.basename(rf_addoff_path), width=img_width)
    else:
        st.info("Optionally add 'additivity disabled' RF SHAP at `graphs/shap_rf_additivity_off.png`.")
    st.markdown("""
**Inference**
- Similar features appear, but with **extremely large SHAP values (±500k)** → numerical/scaling artifacts.
- Conclusion: RF SHAP here is unreliable unless features are carefully scaled/preprocessed.
""")

    # 3) Logistic Regression (Linear SHAP — summary/beeswarm)
    st.markdown("### 3) Logistic Regression (Linear SHAP — summary/beeswarm)")
    lr_summary_path = first_existing(["graphs/shap_summary.png", "/mnt/data/shap_summary.png"])
    if lr_summary_path:
        show_centered_image(lr_summary_path, caption=os.path.basename(lr_summary_path), width=img_width)
    else:
        st.info("Add the Logistic Regression SHAP summary at `graphs/shap_summary.png` (or `/mnt/data/shap_summary.png`).")

    st.markdown("""
**Key drivers (clear & stable)**
- **Warranty**: `warranty_quality_interaction`, `warranty_years`, `warranty_duration_months`
- **Energy**: `energy_rating_stars`
- **Price**: `price_inr`, `price_density_score`
- **Channel & geography**: `platform_*`, `reviewer_location_*`

**Overall takeaway**
- RF explanations unstable and dominated by capacity fields.
- Logistic Regression explanations are interpretable and consistent → main drivers are warranty, price, ratings, and channel.
""")

# ----------------------------
# Fairness tab 
# ----------------------------
with tab_fair:
    st.subheader("Group comparison (selection rate & metrics)")
    if uploaded is None:
        st.info("Upload a CSV to compute group metrics.")
    else:
        try:
            df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None

        if df is not None:
            if sensitive_attr not in df.columns:
                cand = [
                    c for c in df.columns
                    if (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]))
                    and df[c].nunique() <= 30
                ]
                st.warning(
                    f"Sensitive attribute `{sensitive_attr}` not found. "
                    f"Try a low-cardinality categorical column (e.g., `city`, `platform`, `reviewer_location`). "
                    f"Detected candidates: {', '.join(cand[:10]) if cand else 'None'}"
                )
            else:
                al = align_columns(df.copy(), EXPECTED_COLS)
                preds, proba = predict_df(al)
                pred_hat = (proba >= threshold).astype(int) if proba is not None else preds.astype(int)

                grp = df.groupby(df[sensitive_attr].astype(str), dropna=False)
                summary = grp.apply(lambda g: pd.Series({
                    "n": int(len(g)),
                    "selection_rate": float((pred_hat[g.index] == 1).mean()),
                }))
                st.write("**Selection rate by group**")
                st.dataframe(summary.sort_values("selection_rate", ascending=False), use_container_width=True)

                has_gt = bool(target_attr) and (target_attr in df.columns)
                if has_gt:
                    y_true = (
                        df[target_attr]
                        .astype(str).str.lower().str.strip().map(
                            {"in stock": 0, "instock": 0, "available": 0, "0": 0, "false": 0, "no": 0,
                             "out of stock": 1, "outofstock": 1, "unavailable": 1, "1": 1, "true": 1, "yes": 1}
                        )
                        .fillna(0).astype(int).values
                    )
                    acc_by_grp = grp.apply(lambda g: accuracy_score(y_true[g.index], pred_hat[g.index]))
                    f1_by_grp = grp.apply(lambda g: f1_score(y_true[g.index], pred_hat[g.index]))
                    st.write("**Accuracy by group**")
                    st.dataframe(acc_by_grp.to_frame("accuracy").sort_values("accuracy", ascending=False), use_container_width=True)
                    st.write("**F1 by group**")
                    st.dataframe(f1_by_grp.to_frame("f1").sort_values("f1", ascending=False), use_container_width=True)

                    overall_sr = float((pred_hat == 1).mean())
                    summary["selection_rate_gap_vs_overall"] = (summary["selection_rate"] - overall_sr)
                    st.write("**Selection-rate gap vs overall** (positive = predicts 'Out of Stock' more often than average)")
                    st.dataframe(summary[["n", "selection_rate", "selection_rate_gap_vs_overall"]], use_container_width=True)
                else:
                    st.info("No ground-truth column set; showing selection rates only. For per-group accuracy/F1, set a ground-truth column.")

# ----------------------------
# Drift tab 
# ----------------------------
with tab_drift:
    st.subheader("Data drift checks (PSI & KS)")
    if REF is None:
        st.info("Missing reference_sample.csv in artifacts. Upload a CSV to compare, or add reference file.")
    elif uploaded is None:
        st.info("Upload a CSV to compare with the reference sample.")
    else:
        try:
            cur = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        except Exception as e:
            st.error(f"Could not read uploaded CSV for drift: {e}")
            cur = None
        if cur is not None:
            num_cols = list(set(REF.columns).intersection(cur.columns))
            num_cols = [
                c for c in num_cols
                if pd.api.types.is_numeric_dtype(REF[c]) and pd.api.types.is_numeric_dtype(cur[c])
            ]
            if not num_cols:
                st.warning("No common numeric columns for drift.")
            else:
                rows = []
                for c in sorted(num_cols):
                    p = psi(REF[c], cur[c])
                    ks = ks_2samp(REF[c].dropna().astype(float), cur[c].dropna().astype(float)).pvalue
                    rows.append(
                        {"feature": c, "psi": p, "ks_pvalue": ks,
                         "drift_flag": ("HIGH" if (not np.isnan(p) and p >= 0.2)
                                        else ("MED" if (not np.isnan(p) and p >= 0.1) else "LOW"))}
                    )
                drift_df = pd.DataFrame(rows).sort_values("psi", ascending=False)
                st.dataframe(drift_df, use_container_width=True)
                st.caption("Heuristic: PSI ≥ 0.2 = high drift, 0.1–0.2 = medium.")
