import os
import json
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
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

LABEL_MAP = {0: "In Stock", 1: "Out of Stock"}

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
st.title("📊 Voltas Availability — Predictions & Insights")
st.caption("Streamlit dashboard for predictions, SHAP explanations, fairness, and drift checks.")

# ----------------------------
# Sidebar controls (auto-detect sensitive & target)
# ----------------------------
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    # Auto-detect candidate sensitive & target columns from the uploaded CSV (if any)
    detected_sensitive = "city"
    detected_target = ""
    if uploaded is not None:
        try:
            _tmp_df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            # Candidate sensitive cols: categorical with reasonable cardinality (<=30)
            cand_sens = [
                c for c in _tmp_df.columns
                if (pd.api.types.is_object_dtype(_tmp_df[c]) or pd.api.types.is_categorical_dtype(_tmp_df[c]))
                and _tmp_df[c].nunique() <= 30
            ]
            # Prefer common choices if present
            for pref in ["city", "platform", "reviewer_location"]:
                if pref in cand_sens:
                    detected_sensitive = pref
                    break
                elif cand_sens:
                    detected_sensitive = cand_sens[0]

            # Candidate ground truth columns
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


# ----------------------------
# Helper functions
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
    edges[0] -= 1e-9; edges[-1] += 1e-9
    ref_counts = np.histogram(ref, bins=edges)[0]
    cur_counts = np.histogram(cur, bins=edges)[0]
    ref_perc = np.where(ref_counts == 0, 1e-6, ref_counts) / max(1, ref_counts.sum())
    cur_perc = np.where(cur_counts == 0, 1e-6, cur_counts) / max(1, cur_counts.sum())
    return float(np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc)))

def model_type(model):
    if isinstance(model, RandomForestClassifier): return "rf"
    if isinstance(model, LogisticRegression): return "lr"
    return "other"

def get_feature_names(preprocess):
    names = []
    for name, trans, cols in preprocess.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            try:
                oh = trans.named_steps["oh"]
                names.extend(list(oh.get_feature_names_out(cols)))
            except Exception:
                names.extend(cols)
    return names

def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

# ----------------------------
# Tabs
# ----------------------------
tab_pred, tab_shap, tab_fair, tab_drift = st.tabs(["🔮 Predict", "🔎 SHAP", "⚖️ Fairness", "🌊 Drift"])

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
                "warranty_quality_interaction": 1.24
            }
            df_in = pd.DataFrame([demo])

        if df_in is not None:
            al = align_columns(df_in.copy(), EXPECTED_COLS)
            preds, proba = predict_df(al)
            out = pd.DataFrame({
                "prediction": preds.astype(int),
                "label": [LABEL_MAP.get(int(p), str(p)) for p in preds],
                "proba_out_of_stock": proba if proba is not None else np.nan
            })
            st.success(f"Predicted {len(out)} rows.")
            st.dataframe(out.head(50), use_container_width=True)

            # optional ground truth
            gt_col = None
            for c in ["availability", "Availability", "is_out_of_stock", "target"]:
                if c in df_in.columns: gt_col = c; break
            if gt_col:
                y_true_raw = df_in[gt_col].astype(str).str.lower().str.strip().map({
                    "in stock": 0, "instock": 0, "available": 0, "0": 0, "false": 0, "no": 0,
                    "out of stock": 1, "outofstock": 1, "unavailable": 1, "1": 1, "true": 1, "yes": 1,
                })
                y_true = y_true_raw.fillna(0).astype(int).values
                y_pred = (out["proba_out_of_stock"].fillna(0.0).values >= threshold).astype(int) if proba is not None else out["prediction"].values
                st.write("**Metrics**")
                y_pred = (out["proba_out_of_stock"].fillna(0.0).values >= threshold).astype(int) if proba is not None else out["prediction"].values
                # tabular classification report
                report_dict = classification_report(y_true, y_pred, output_dict=True, digits=4)
                report_df = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "label"})
                # order columns nicely if present
                cols = [c for c in ["label", "precision", "recall", "f1-score", "support"] if c in report_df.columns]
                st.dataframe(report_df[cols], use_container_width=True)

                # confusion matrix as a table
                cm = confusion_matrix(y_true, y_pred)
                cm_df = pd.DataFrame(cm, index=["True 0 (In Stock)", "True 1 (Out of Stock)"],
                                        columns=["Pred 0", "Pred 1"])
                st.write("**Confusion matrix**")
                st.dataframe(cm_df, use_container_width=True)               

# ----------------------------
# SHAP tab (robust bar-only; handles 1D/2D/3D SHAP)
# ----------------------------
with tab_shap:
    st.subheader("Global feature importance (SHAP) — Robust view")

    if inference_pipeline is None or EXPECTED_COLS is None:
        st.warning("Artifacts not loaded.")
    else:
        # pick base data
        base = REF.copy() if REF is not None else None
        if base is None and uploaded is not None:
            try:
                tmp = pd.read_csv(io.BytesIO(uploaded.getvalue()))
                base = tmp.sample(n=min(200, len(tmp)), random_state=42)
            except Exception as e:
                st.error(f"Error using uploaded CSV for SHAP: {e}")
                base = None

        if base is None:
            st.info("Need reference_sample.csv (in artifacts) or an uploaded CSV to compute SHAP.")
        else:
            # align & encode
            al = align_columns(base.copy(), EXPECTED_COLS)
            pre = inference_pipeline.named_steps["preprocess"]
            X_enc = pre.transform(al)
            X_dense = X_enc.toarray() if hasattr(X_enc, "toarray") else np.asarray(X_enc)

            mdl = inference_pipeline.named_steps["model"]

            # feature names (best effort)
            def _names_from_pre(preprocessor):
                names = []
                for name, trans, cols in preprocessor.transformers_:
                    if name == "num":
                        names.extend(cols)
                    elif name == "cat":
                        try:
                            oh = trans.named_steps["oh"]
                            names.extend(list(oh.get_feature_names_out(cols)))
                        except Exception:
                            names.extend(cols)
                return names
            try:
                feat_names = _names_from_pre(pre)
            except Exception:
                feat_names = [f"f{i}" for i in range(X_dense.shape[1])]

            # compute SHAP
            try:
                import shap
                if isinstance(mdl, RandomForestClassifier):
                    expl = shap.TreeExplainer(mdl)
                    vals = expl.shap_values(X_dense, check_additivity=False)
                    if isinstance(vals, list):
                        sv = vals[1] if len(vals) > 1 else np.array(vals[0])
                    else:
                        sv = np.array(vals)
                elif isinstance(mdl, LogisticRegression):
                    expl = shap.LinearExplainer(mdl, X_dense)
                    sv = np.array(expl(X_dense).values)
                else:
                    bg = shap.sample(X_dense, min(100, X_dense.shape[0]))
                    expl = shap.KernelExplainer(mdl.predict_proba, bg)
                    sv = np.array(expl.shap_values(X_dense[:50])[1])
            except Exception as e:
                st.error(f"SHAP failed: {e}")
                st.stop()

            # -------- normalize shapes to (n_samples, n_features) --------
            sv = np.asarray(sv)

            if sv.ndim == 1:
                sv = sv.reshape(-1, 1)
            elif sv.ndim == 3:
                # common layout: (classes, n_samples, n_features) -> use last class
                if sv.shape[0] <= 3 and sv.shape[-1] >= 2:
                    sv = sv[-1]
                else:
                    # fallback: collapse leading dims into samples
                    sv = sv.reshape(-1, sv.shape[-1])
            elif sv.ndim > 3:
                sv = sv.reshape(sv.shape[-2], sv.shape[-1])

            # compute importance & flatten
            mean_abs = np.mean(np.abs(sv), axis=0)
            mean_abs = np.ravel(mean_abs)  # ensure 1D

            # align feat_names length with shap width
            if len(feat_names) != len(mean_abs):
                feat_names = [f"f{i}" for i in range(len(mean_abs))]

            # build importance df safely
            imp_df = pd.DataFrame({
                "feature": pd.Index(feat_names, dtype="object"),
                "mean_abs_shap": pd.Series(mean_abs, dtype="float64")
            }).sort_values("mean_abs_shap", ascending=False)

            st.write("**Top features by mean(|SHAP|)**")
            st.dataframe(imp_df.head(25), use_container_width=True)

            # bar chart
            fig, ax = plt.subplots(figsize=(8, 6))
            top = imp_df.head(20)
            ax.barh(range(len(top)), top["mean_abs_shap"].to_numpy())
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top["feature"].tolist())
            ax.invert_yaxis()
            ax.set_xlabel("mean(|SHAP value|)")
            ax.set_title("SHAP Feature Importance (Top 20)")
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

            st.caption("Bar-only mode is used to avoid one-hot shape mismatches that can break beeswarm/dependence plots.")

# ----------------------------
# Fairness tab (selection rate + optional metrics)
# ----------------------------
with tab_fair:
    st.subheader("Group comparison (selection rate & metrics)")

    if uploaded is None:
        st.info("Upload a CSV to compute group metrics.")
        st.stop()

    # Read the uploaded CSV safely
    try:
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # Check sensitive attribute presence
    if sensitive_attr not in df.columns:
        # Suggest possible alternatives
        cand = [
            c for c in df.columns
            if (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]))
            and df[c].nunique() <= 30
        ]
        st.warning(
            f"Sensitive attribute `{sensitive_attr}` not found in uploaded CSV. "
            f"Try a categorical column with small cardinality (e.g., `city`, `platform`, `reviewer_location`). "
            f"Detected candidates: {', '.join(cand[:10]) if cand else 'None'}"
        )
        st.stop()

    # Align and predict
    al = align_columns(df.copy(), EXPECTED_COLS)
    preds, proba = predict_df(al)
    pred_hat = (proba >= threshold).astype(int) if proba is not None else preds.astype(int)

    # Group-level selection rate
    grp = df.groupby(df[sensitive_attr].astype(str), dropna=False)
    summary = grp.apply(lambda g: pd.Series({
        "n": int(len(g)),
        "selection_rate": float((pred_hat[g.index] == 1).mean()),  # predicted 'Out of Stock' rate
    }))
    st.write("**Selection rate by group**")
    st.dataframe(summary.sort_values("selection_rate", ascending=False), use_container_width=True)

    # Optional metrics (only if ground-truth is provided)
    has_gt = bool(target_attr) and (target_attr in df.columns)

    if has_gt:
        # Normalize common string labels into 0/1
        y_true = df[target_attr].astype(str).str.lower().str.strip().map({
            "in stock": 0, "instock": 0, "available": 0, "0": 0, "false": 0, "no": 0,
            "out of stock": 1, "outofstock": 1, "unavailable": 1, "1": 1, "true": 1, "yes": 1,
        }).fillna(0).astype(int).values

        acc_by_grp = grp.apply(lambda g: accuracy_score(y_true[g.index], pred_hat[g.index]))
        f1_by_grp = grp.apply(lambda g: f1_score(y_true[g.index], pred_hat[g.index]))

        st.write("**Accuracy by group**")
        st.dataframe(acc_by_grp.to_frame("accuracy").sort_values("accuracy", ascending=False), use_container_width=True)

        st.write("**F1 by group**")
        st.dataframe(f1_by_grp.to_frame("f1").sort_values("f1", ascending=False), use_container_width=True)

        # Simple parity gaps (selection rate difference vs. overall)
        overall_sr = float((pred_hat == 1).mean())
        summary["selection_rate_gap_vs_overall"] = summary["selection_rate"] - overall_sr
        st.write("**Selection-rate gap vs overall** (positive = predicts 'Out of Stock' more often than average)")
        st.dataframe(summary[["n", "selection_rate", "selection_rate_gap_vs_overall"]], use_container_width=True)
    else:
        st.info(
            "No ground-truth column selected; showing **selection rates** only. "
            "If you want accuracy/F1 per group, set a ground-truth column (e.g., `availability`, `is_out_of_stock`, `target`, or `label`)."
        )

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
            num_cols = [c for c in num_cols if pd.api.types.is_numeric_dtype(REF[c]) and pd.api.types.is_numeric_dtype(cur[c])]
            if not num_cols:
                st.warning("No common numeric columns for drift.")
            else:
                rows = []
                for c in sorted(num_cols):
                    p = psi(REF[c], cur[c])
                    ks = ks_2samp(REF[c].dropna().astype(float), cur[c].dropna().astype(float)).pvalue
                    rows.append({
                        "feature": c,
                        "psi": p,
                        "ks_pvalue": ks,
                        "drift_flag": "HIGH" if (not np.isnan(p) and p >= 0.2)
                                      else ("MED" if (not np.isnan(p) and p >= 0.1) else "LOW")
                    })
                drift_df = pd.DataFrame(rows).sort_values("psi", ascending=False)
                st.dataframe(drift_df, use_container_width=True)
                st.caption("Heuristic: PSI ≥ 0.2 = high drift, 0.1–0.2 = medium.")
