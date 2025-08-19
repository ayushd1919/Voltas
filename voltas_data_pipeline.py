# voltas_data_pipeline.py
import pandas as pd
import sweetviz as sv
import great_expectations as ge

# -------------------------------
# 1. Load raw dataset
# -------------------------------
df_raw = pd.read_csv("voltas_raw_scraped_like.csv")
print(f"✅ Loaded dataset with {df_raw.shape[0]} rows and {df_raw.shape[1]} columns")

# -------------------------------
# 2. Profile RAW dataset (Sweetviz)
# -------------------------------
report_raw = sv.analyze(df_raw)
report_raw.show_html("voltas_raw_profile.html")
print("📄 Raw data profiling saved as voltas_raw_profile.html")

# -------------------------------
# 3. Cleaning steps
# -------------------------------
df_cleaned = df_raw.copy()

# Drop duplicates based on key columns
key_columns = [col for col in ['product_id', 'username', 'review_date'] if col in df_cleaned.columns]
if key_columns:
    df_cleaned.drop_duplicates(subset=key_columns, inplace=True)

# Convert dates to datetime
for date_col in ['review_date', 'manufacturing_date']:
    if date_col in df_cleaned.columns:
        df_cleaned[date_col] = pd.to_datetime(df_cleaned[date_col], errors='coerce', dayfirst=True)

# Fill missing numeric columns with median
num_cols = df_cleaned.select_dtypes(include='number').columns
df_cleaned[num_cols] = df_cleaned[num_cols].apply(lambda col: col.fillna(col.median()))

# Fill missing categorical columns with mode
cat_cols = df_cleaned.select_dtypes(include='object').columns
for col in cat_cols:
    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

df_cleaned.to_csv("voltas_cleaned.csv", index=False)
print("✅ Cleaned dataset saved as voltas_cleaned.csv")

# -------------------------------
# 4. Feature Engineering
# -------------------------------
if 'review_text' in df_cleaned.columns:
    df_cleaned["review_length_words"] = df_cleaned["review_text"].astype(str).apply(lambda x: len(x.split()))
    df_cleaned["review_length_chars"] = df_cleaned["review_text"].astype(str).apply(len)

if 'rating' in df_cleaned.columns:
    df_cleaned["rating_bucket"] = pd.cut(df_cleaned["rating"], bins=[0, 2, 4, 5],
                                         labels=["Low", "Medium", "High"], include_lowest=True)

df_cleaned.to_csv("voltas_featured.csv", index=False)
print("✨ Feature engineered dataset saved as voltas_featured.csv")

# -------------------------------
# 5. Data Validation (Great Expectations)
# -------------------------------
ge_df = ge.from_pandas(df_cleaned)

# Example validations
if 'rating' in ge_df.columns:
    ge_df.expect_column_values_to_be_between("rating", min_value=0, max_value=5)
if 'review_date' in ge_df.columns:
    ge_df.expect_column_values_to_not_be_null("review_date")
if 'username' in ge_df.columns:
    ge_df.expect_column_values_to_not_be_null("username")

validation_results = ge_df.validate()
print("📊 Validation Results:")
print(validation_results)

# -------------------------------
# 6. Profile CLEANED dataset (Sweetviz)
# -------------------------------
report_clean = sv.analyze(df_cleaned)
report_clean.show_html("voltas_cleaned_profile.html")
print("📄 Cleaned data profiling saved as voltas_cleaned_profile.html")

print("✅ Pipeline completed successfully")
