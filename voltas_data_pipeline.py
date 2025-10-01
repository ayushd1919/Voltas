import os
import sys
import argparse
import json
import warnings
from datetime import datetime

import pandas as pd
import numpy as np

# Optional imports (handled gracefully)
def _try_import(module_name, attr=None):
    try:
        mod = __import__(module_name, fromlist=[attr] if attr else [])
        return getattr(mod, attr) if attr else mod
    except Exception:
        return None

# Detect Colab
COLAB = _try_import("google.colab") is not None

# ydata_profiling (optional)
ProfileReport = None
_ydp = _try_import("ydata_profiling", "ProfileReport")
if _ydp is not None:
    ProfileReport = _ydp

# great_expectations (optional, legacy PandasDataset path preferred)
gx = _try_import("great_expectations")
PandasDataset = _try_import("great_expectations.dataset", "PandasDataset")

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def ensure_outdir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def save_df(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    log(f"Saved: {path} ({len(df)} rows)")

def profile_df(df: pd.DataFrame, title: str, out_html: str):
    if ProfileReport is None:
        log("ydata-profiling not installed; skipping profile.")
        return False
    try:
        log(f"Building profiling report: {title}")
        profile = ProfileReport(df, title=title, explorative=True)
        profile.to_file(out_html)
        log(f"Profiling report saved: {out_html}")
        return True
    except Exception as e:
        log(f"Profiling failed: {e}")
        return False

def clean_voltas_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Empty DataFrame provided to clean_voltas_df().")

    dfc = df.copy()
    log(f"Cleaning start — input: {dfc.shape[0]} rows, {dfc.shape[1]} cols")

    # Drop duplicates on key columns if present
    key_cols = [c for c in ['product_id', 'username', 'review_date'] if c in dfc.columns]
    if key_cols:
        before = len(dfc)
        dfc.drop_duplicates(subset=key_cols, inplace=True)
        log(f"Dropped {before - len(dfc)} duplicates on keys {key_cols}")
    else:
        log("No key columns found for de-dup (skipped).")

    # Coerce dates
    for date_col in ['review_date', 'manufacturing_date']:
        if date_col in dfc.columns:
            dfc[date_col] = pd.to_datetime(dfc[date_col], errors='coerce', dayfirst=True)
            nulls = dfc[date_col].isna().sum()
            log(f"Parsed '{date_col}' → NaT count: {nulls}")

    # Fill numeric NA with median
    num_cols = dfc.select_dtypes(include='number').columns.tolist()
    if num_cols:
        for c in num_cols:
            med = dfc[c].median() if not dfc[c].dropna().empty else 0
            dfc[c] = dfc[c].fillna(med)
        log(f"Filled numeric NA with medians for {len(num_cols)} columns")
    else:
        log("No numeric columns found (skipped numeric fills).")

    # Fill categorical NA with mode
    cat_cols = dfc.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        for c in cat_cols:
            if dfc[c].notna().any():
                mode_val = dfc[c].mode(dropna=True)
                mode_val = mode_val.iloc[0] if not mode_val.empty else ""
                dfc[c] = dfc[c].fillna(mode_val)
        log(f"Filled categorical NA with mode for {len(cat_cols)} columns")
    else:
        log("No object/categorical columns found (skipped cat fills).")

    log(f"Cleaning complete — output: {dfc.shape[0]} rows, {dfc.shape[1]} cols")
    return dfc

def engineer_voltas_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Empty DataFrame provided to engineer_voltas_features().")

    dff = df.copy()
    log(f"Feature engineering start — input: {dff.shape[0]} rows")

    # Ensure dates are datetime
    for date_col in ['review_date', 'manufacturing_date']:
        if date_col in dff.columns and not np.issubdtype(dff[date_col].dtype, np.datetime64):
            dff[date_col] = pd.to_datetime(dff[date_col], errors='coerce', dayfirst=True)

    # Text-based features
    if 'review_text' in dff.columns:
        dff['review_text'] = dff['review_text'].astype(str)
        dff['review_length_words'] = dff['review_text'].apply(lambda x: len(x.split()))
        dff['review_length_chars'] = dff['review_text'].apply(len)
        log("Added review_length_words & review_length_chars")

    # Rating bucket
    if 'rating' in dff.columns:
        try:
            dff['rating'] = pd.to_numeric(dff['rating'], errors='coerce')
            dff['rating_bucket'] = pd.cut(
                dff['rating'],
                bins=[0, 2, 4, 5],
                labels=["Low", "Medium", "High"],
                include_lowest=True
            )
            log("Added rating_bucket (Low/Medium/High)")
        except Exception as e:
            log(f"rating_bucket creation skipped: {e}")

    # Competitor price diff %
    if all(col in dff.columns for col in ['price_inr', 'competitor_price']):
        dff['price_inr'] = pd.to_numeric(dff['price_inr'], errors='coerce')
        dff['competitor_price'] = pd.to_numeric(dff['competitor_price'], errors='coerce')
        dff['competitor_price_diff_percent'] = (
            (dff['price_inr'] - dff['competitor_price']) / (dff['competitor_price'] + 1e-6)
        ) * 100
        log("Added competitor_price_diff_percent")

    # Capacity unified
    capacity_cols = ['capacity_tons', 'capacity_liters', 'capacity_kg', 'capacity_place_settings']
    if any(c in dff.columns for c in capacity_cols) and 'capacity_unified' not in dff.columns:
        def unify_capacity(row):
            if 'capacity_tons' in row and pd.notna(row['capacity_tons']):   return f"{row['capacity_tons']} Tons"
            if 'capacity_kg' in row and pd.notna(row['capacity_kg']):       return f"{row['capacity_kg']} Kg"
            if 'capacity_liters' in row and pd.notna(row['capacity_liters']): return f"{row['capacity_liters']} Liters"
            if 'capacity_place_settings' in row and pd.notna(row['capacity_place_settings']): return f"{row['capacity_place_settings']} Place Settings"
            return "Not Applicable"
        dff['capacity_unified'] = dff.apply(unify_capacity, axis=1)
        log("Built capacity_unified from capacity_* columns")

    # Price density score
    if 'price_inr' in dff.columns and 'capacity_unified' in dff.columns:
        dff['capacity_numeric'] = pd.to_numeric(
            dff['capacity_unified'].astype(str).str.extract(r'(\\d+\\.?\\d*)')[0], errors='coerce'
        )
        dff['price_density_score'] = dff['price_inr'] / (dff['capacity_numeric'] + 1e-6)
        dff['price_density_score'].replace([np.inf, -np.inf], np.nan, inplace=True)
        dff['price_density_score'] = dff['price_density_score'].fillna(0)
        dff.drop(columns=['capacity_numeric'], inplace=True)
        log("Added price_density_score")

    # Warranty × rating interaction
    if 'warranty_years' in dff.columns and 'customer_rating' in dff.columns:
        dff['warranty_years'] = pd.to_numeric(dff['warranty_years'], errors='coerce').fillna(0)
        dff['customer_rating'] = pd.to_numeric(dff['customer_rating'], errors='coerce').fillna(0)
        dff['warranty_quality_interaction'] = dff['warranty_years'] * dff['customer_rating']
        log("Added warranty_quality_interaction")

    log("Feature engineering complete.")
    return dff

def run_ge_validation(df: pd.DataFrame, out_path_txt: str = None) -> dict:
    summary = {"success": None, "stats": {}, "missing_columns": [], "extra_columns": []}
    if gx is None or PandasDataset is None:
        log("great_expectations not installed; skipping GE validation.")
        return summary

    ge_df = PandasDataset(df)

    def col_exists(c: str) -> bool:
        return c in ge_df.columns

    expected_core = [c for c in [
        'product_id','username','review_date','review_text','rating',
        'price_inr','competitor_price','warranty_years','customer_rating',
        'capacity_unified','review_length_words','review_length_chars',
        'rating_bucket','competitor_price_diff_percent','price_density_score',
        'warranty_quality_interaction'
    ] if col_exists(c)]

    if expected_core:
        ge_df.expect_table_columns_to_match_set(column_set=expected_core, exact_match=False)

    if col_exists('rating'):
        ge_df.expect_column_values_to_be_between('rating', min_value=0, max_value=5)

    if col_exists('customer_rating'):
        ge_df.expect_column_values_to_be_between('customer_rating', min_value=0, max_value=5)

    if col_exists('warranty_years'):
        ge_df.expect_column_values_to_be_between('warranty_years', min_value=0)

    for c in ['product_id', 'username', 'review_date']:
        if col_exists(c):
            ge_df.expect_column_values_to_not_be_null(c)

    if col_exists('rating_bucket'):
        ge_df.expect_column_values_to_be_in_set('rating_bucket', ['Low','Medium','High'])

    result = ge_df.validate()
    summary["success"] = bool(result.success)

    try:
        stats = result["statistics"]
        summary["stats"] = {
            "evaluated_expectations": stats.get("evaluated_expectations"),
            "successful_expectations": stats.get("successful_expectations"),
            "unsuccessful_expectations": stats.get("unsuccessful_expectations"),
        }
    except Exception:
        pass

    if expected_core:
        observed = set(ge_df.columns)
        expected = set(expected_core)
        summary["missing_columns"] = sorted(expected - observed)
        summary["extra_columns"] = sorted(observed - expected)

    msg_lines = [
        "Validation Summary:",
        f"SUCCESS: {summary['success']}",
        f"Stats: {json.dumps(summary['stats'])}",
    ]
    if summary["missing_columns"]:
        msg_lines.append(f"Missing columns: {summary['missing_columns']}")
    if summary["extra_columns"]:
        msg_lines.append(f"Extra columns: {summary['extra_columns']}")

    msg = "\n".join(msg_lines)
    log(msg)

    if out_path_txt:
        with open(out_path_txt, "w", encoding="utf-8") as f:
            f.write(msg + "\n")
        log(f"GE summary saved: {out_path_txt}")

    return summary

def maybe_colab_upload() -> str:
    if not COLAB:
        return None
    try:
        from google.colab import files
        log("Colab upload dialog opened — select your CSV file.")
        uploaded = files.upload()
        filename = next(iter(uploaded)) if uploaded else None
        if filename:
            log(f"Uploaded: {filename}")
        return filename
    except Exception as e:
        log(f"Colab upload failed: {e}")
        return None

def maybe_colab_download(path: str):
    if not (COLAB and os.path.exists(path)):
        return
    try:
        from google.colab import files
        files.download(path)
        log(f"Colab download triggered: {path}")
    except Exception as e:
        log(f"Colab auto-download not available: {e}")

def parse_args():
    p = argparse.ArgumentParser(description="Voltas Data Pipeline — Single File")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--infile", type=str, help="Path to input CSV (local run).")
    src.add_argument("--colab_upload", action="store_true", help="Use Colab upload dialog.")
    p.add_argument("--outdir", type=str, default=".", help="Output directory (default: .)")

    # Step toggles
    p.add_argument("--raw_profile", dest="raw_profile", action="store_true", help="Run raw profiling step.")
    p.add_argument("--no_raw_profile", dest="raw_profile", action="store_false", help="Skip raw profiling step.")
    p.set_defaults(raw_profile=True)

    p.add_argument("--clean", dest="clean", action="store_true", help="Run cleaning step.")
    p.add_argument("--no_clean", dest="clean", action="store_false", help="Skip cleaning step.")
    p.set_defaults(clean=True)

    p.add_argument("--features", dest="features", action="store_true", help="Run feature engineering step.")
    p.add_argument("--no_features", dest="features", action="store_false", help="Skip feature engineering step.")
    p.set_defaults(features=True)

    p.add_argument("--ge", dest="ge", action="store_true", help="Run GE validation step.")
    p.add_argument("--no_ge", dest="ge", action="store_false", help="Skip GE validation step.")
    p.set_defaults(ge=True)

    p.add_argument("--clean_profile", dest="clean_profile", action="store_true", help="Run cleaned profiling step.")
    p.add_argument("--no_clean_profile", dest="clean_profile", action="store_false", help="Skip cleaned profiling step.")
    p.set_defaults(clean_profile=True)

    p.add_argument("--do_all", action="store_true", help="Enable all steps (equivalent to defaults).")

    return p.parse_args()

def main():
    warnings.filterwarnings("ignore")
    args = parse_args()

    if args.do_all:
        args.raw_profile = True
        args.clean = True
        args.features = True
        args.ge = True
        args.clean_profile = True

    ensure_outdir(args.outdir)

    infile = args.infile
    if args.colab_upload:
        infile = maybe_colab_upload()

    if not infile or not os.path.exists(infile):
        if infile:
            log(f"Input file not found: {infile}")
        else:
            log("No input file provided.")
        sys.exit(1)

    log(f"Reading CSV: {infile}")
    df_raw = pd.read_csv(infile)
    log(f"Loaded raw data: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")

    # Step 1: Raw profiling
    if args.raw_profile:
        raw_profile_path = os.path.join(args.outdir, "voltas_raw_profile.html")
        profile_df(df_raw, "Voltas Dataset — RAW Profiling Report", raw_profile_path)
        maybe_colab_download(raw_profile_path)

    # Step 2: Cleaning
    df_cleaned = df_raw
    cleaned_csv_path = os.path.join(args.outdir, "voltas_cleaned.csv")
    if args.clean:
        df_cleaned = clean_voltas_df(df_raw)
        save_df(df_cleaned, cleaned_csv_path)
        maybe_colab_download(cleaned_csv_path)
    else:
        log("Skipping cleaning step; using raw as cleaned.")

    # Step 3: Features
    df_feat = df_cleaned
    featured_csv_path = os.path.join(args.outdir, "voltas_featured.csv")
    if args.features:
        df_feat = engineer_voltas_features(df_cleaned)
        save_df(df_feat, featured_csv_path)
        maybe_colab_download(featured_csv_path)
    else:
        log("Skipping feature engineering step.")

    # Step 4: GE Validation
    if args.ge:
        ge_txt = os.path.join(args.outdir, "ge_validation_summary.txt")
        run_ge_validation(df_feat, ge_txt)
        maybe_colab_download(ge_txt)
    else:
        log("Skipping GE validation step.")

    # Step 5: Cleaned profiling
    if args.clean_profile:
        cleaned_profile_path = os.path.join(args.outdir, "voltas_cleaned_profile.html")
        profile_df(df_cleaned, "Voltas Dataset — CLEANED Profiling Report", cleaned_profile_path)
        maybe_colab_download(cleaned_profile_path)
    else:
        log("Skipping cleaned profiling step.")

    log("Pipeline complete.")

if __name__ == "__main__":
    main()
out_path = "/mnt/data/voltas_pipeline.py"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(script)

out_path