
import pandas as pd
import numpy as np
import os

# -----------------------------------------
# 1. LOAD DATASET
# -----------------------------------------

print("Loading railway dataset...")

df = pd.read_parquet(
    "data/processed/train_delay_combined.parquet"
)

os.makedirs("reports/phase2", exist_ok=True)

print("Dataset loaded:", df.shape)

# -----------------------------------------
# 2. CHECK JOURNEY IDENTIFICATION
# -----------------------------------------

print("\n===== JOURNEY ANALYSIS =====")

journey_columns = ["train_number", "start_date"]

journeys = df.groupby(journey_columns, dropna=False)

print("Total unique train journeys:", len(journeys))

print("\nRecords per journey:")
print(journeys.size().describe())

# -----------------------------------------
# 3. CHECK DUPLICATE STATION SEQUENCES
# -----------------------------------------

print("\n===== STATION SEQUENCE CHECK =====")

duplicate_stops = df.duplicated(
    subset=["train_number", "start_date", "seq"],
    keep=False
)

print("Duplicate journey-station records:",
      duplicate_stops.sum())

# -----------------------------------------
# 4. CHECK SEQUENCE VALIDITY
# -----------------------------------------

print("\n===== SEQUENCE CHECK =====")

print("Missing sequence:", df["seq"].isna().sum())

print("Sequence values:")
print(df["seq"].describe())

# -----------------------------------------
# 5. CHECK TRAIN STATUS
# -----------------------------------------

print("\n===== TRAIN STATUS =====")

print(df["status"].value_counts(dropna=False))

# -----------------------------------------
# 6. CHECK ARRIVAL DELAY DISTRIBUTION
# -----------------------------------------

print("\n===== ARRIVAL DELAY ANALYSIS =====")

print(df["arrival_delay_min"].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
))

# -----------------------------------------
# 7. CHECK EXTREME DELAYS
# -----------------------------------------

print("\n===== EXTREME DELAYS =====")

print(
    "Delays above 180 minutes:",
    (df["arrival_delay_min"] > 180).sum()
)

print(
    "Early arrivals:",
    (df["arrival_delay_min"] < 0).sum()
)

# -----------------------------------------
# 8. CHECK DELAY CALCULATION
# -----------------------------------------

print("\n===== DELAY CONSISTENCY =====")

difference = (
    df["actual_arrival"] -
    df["scheduled_arrival"]
).dt.total_seconds() / 60

valid = difference.notna() & df["arrival_delay_min"].notna()

if valid.any():

    error = (
        difference[valid] -
        df.loc[valid, "arrival_delay_min"]
    ).abs()

    print("Records compared:", valid.sum())
    print("Median absolute difference:", error.median())
    print("Maximum absolute difference:", error.max())

# -----------------------------------------
# 9. SAVE AUDIT RESULTS
# -----------------------------------------

missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percentage": df.isna().mean() * 100
})

missing.to_csv(
    "reports/phase2/missing_values.csv"
)

duplicate_rows = df[duplicate_stops]

duplicate_rows.to_csv(
    "reports/phase2/duplicate_station_records.csv",
    index=False
)

print("\n===== AUDIT COMPLETED =====")
print("Reports saved inside reports/phase2/")
