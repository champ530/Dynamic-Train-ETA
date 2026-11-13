
import pandas as pd

print("LOADING RAILWAY DATASET...")

df = pd.read_parquet(
    "data/processed/train_delay_combined.parquet"
)

# 1. Date coverage
print("\n===== DATE COVERAGE =====")

df["start_date"] = pd.to_datetime(
    df["start_date"],
    errors="coerce"
)

print("Earliest journey date:", df["start_date"].min())
print("Latest journey date:", df["start_date"].max())

print("Unique journey dates:",
      df["start_date"].nunique())

print("\nRecords per date:")
print(df.groupby("start_date").size().head(20))

# 2. Journey identification
print("\n===== JOURNEY ANALYSIS =====")

journey_columns = ["train_number", "start_date"]

print(
    "Unique train journeys:",
    df[journey_columns].drop_duplicates().shape[0]
)

print("\nRecords per journey:")
print(
    df.groupby(journey_columns).size().describe()
)

# 3. Train status
print("\n===== TRAIN STATUS =====")

print(df["status"].value_counts(dropna=False))

# 4. Station sequence
print("\n===== STATION SEQUENCE =====")

print("Minimum sequence:", df["seq"].min())
print("Maximum sequence:", df["seq"].max())

print("\nRecords by station sequence:")
print(df["seq"].value_counts().sort_index().head(20))

# 5. Duplicate train-station records
print("\n===== DUPLICATE STATION RECORDS =====")

duplicate_mask = df.duplicated(
    subset=[
        "train_number",
        "start_date",
        "seq"
    ],
    keep=False
)

print("Duplicate station records:", duplicate_mask.sum())

# 6. Target distribution by date
print("\n===== DAILY ARRIVAL DELAY =====")

daily_delay = (
    df.groupby("start_date")["arrival_delay_min"]
    .agg(["count", "mean", "median"])
)

print(daily_delay.head(20))

# 7. Save analysis
daily_delay.to_csv(
    "reports/daily_delay_analysis.csv"
)

print("\nJOURNEY ANALYSIS COMPLETED!")
