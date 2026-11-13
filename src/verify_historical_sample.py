
import pandas as pd
import os

print("LOADING HISTORICAL SAMPLE...")

file_path = "data/processed/railway_historical_sample.parquet"

df = pd.read_parquet(file_path)

# Convert journey date
df["start_date"] = pd.to_datetime(
    df["start_date"],
    errors="coerce"
)

# Extract calendar date
df["journey_date"] = df["start_date"].dt.date

print("\n===== DATASET SIZE =====")
print("Total records:", len(df))
print("Total columns:", len(df.columns))

# Calendar date coverage
print("\n===== DATE COVERAGE =====")

print("Unique calendar dates:",
      df["journey_date"].nunique())

print("Earliest date:", df["journey_date"].min())
print("Latest date:", df["journey_date"].max())

print("\nRecords per calendar date:")
print(df.groupby("journey_date").size().to_string())

# Train journey analysis
print("\n===== TRAIN JOURNEY ANALYSIS =====")

journey_columns = ["train_number", "journey_date"]

journey_sizes = df.groupby(
    journey_columns
).size()

print("Unique train journeys:", len(journey_sizes))

print("\nRecords per journey:")
print(journey_sizes.describe())

# Station sequence analysis
print("\n===== STATION SEQUENCE =====")

print("Minimum sequence:", df["seq"].min())
print("Maximum sequence:", df["seq"].max())

duplicate_mask = df.duplicated(
    subset=[
        "train_number",
        "journey_date",
        "seq"
    ],
    keep=False
)

print("Duplicate journey-station records:",
      duplicate_mask.sum())

# Arrival delay analysis
print("\n===== ARRIVAL DELAY =====")

print("Valid arrival delays:",
      df["arrival_delay_min"].notna().sum())

print("Missing arrival delays:",
      df["arrival_delay_min"].isna().sum())

print("\nDelay statistics:")
print(df["arrival_delay_min"].describe())

# Save verification report
os.makedirs("reports/phase2", exist_ok=True)

journey_sizes.to_csv(
    "reports/phase2/journey_sizes.csv"
)

print("\n===== VERIFICATION COMPLETED =====")
print("Report saved in reports/phase2/")
