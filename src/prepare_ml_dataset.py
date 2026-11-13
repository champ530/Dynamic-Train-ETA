
import pandas as pd
import numpy as np
import os

print("STARTING DATA CLEANING AND FEATURE ENGINEERING...")

# 1. Load dataset
file_path = "data/processed/complete_journey_sample.parquet"
df = pd.read_parquet(file_path)

print("\nOriginal records:", len(df))

# 2. Convert date and time columns
datetime_columns = [
    "start_date",
    "scheduled_arrival",
    "scheduled_departure"
]

for col in datetime_columns:
    df[col] = pd.to_datetime(
        df[col],
        errors="coerce"
    )

# 3. Convert numeric columns
numeric_columns = [
    "seq",
    "arrival_delay_min",
    "departure_delay_min"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# 4. Create journey date
df["journey_date"] = df["start_date"].dt.date

# Remove records without essential identifiers
df = df.dropna(
    subset=[
        "train_number",
        "journey_date",
        "seq"
    ]
)

# 5. Sort station records
df = df.sort_values(
    ["train_number", "journey_date", "seq"]
).reset_index(drop=True)

# 6. Create temporal features
df["hour"] = df["scheduled_arrival"].dt.hour

df["day_of_week"] = pd.to_datetime(
    df["journey_date"]
).dt.dayofweek

df["month"] = pd.to_datetime(
    df["journey_date"]
).dt.month

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# 7. Create seasonal feature
def get_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4, 5]:
        return "Summer"

    elif month in [6, 7, 8, 9]:
        return "Monsoon"

    else:
        return "Autumn"

df["season"] = df["month"].apply(get_season)

# 8. Calculate scheduled travel time
groups = df.groupby(
    ["train_number", "journey_date"],
    sort=False
)

df["scheduled_travel_minutes"] = (
    groups["scheduled_arrival"]
    .diff()
    .dt.total_seconds() / 60
)

# 9. Previous station arrival delay
df["previous_arrival_delay"] = (
    groups["arrival_delay_min"].shift(1)
)

# 10. Select useful features
feature_columns = [
    "train_number",
    "category",
    "station_code",
    "seq",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "season",
    "scheduled_travel_minutes",
    "previous_arrival_delay",
    "arrival_delay_min",
    "journey_date"
]

ml_df = df[feature_columns].copy()

# 11. Remove rows without prediction target
ml_df = ml_df.dropna(
    subset=[
        "arrival_delay_min",
        "previous_arrival_delay",
        "scheduled_travel_minutes",
        "hour"
    ]
)

# Remove invalid scheduled travel intervals
ml_df = ml_df[
    ml_df["scheduled_travel_minutes"] > 0
].copy()

# 12. Remove infinite values
ml_df = ml_df.replace(
    [np.inf, -np.inf],
    np.nan
)

ml_df = ml_df.dropna(
    subset=[
        "arrival_delay_min",
        "previous_arrival_delay",
        "scheduled_travel_minutes"
    ]
)

# 13. Save processed dataset
os.makedirs("data/processed", exist_ok=True)

output_file = (
    "data/processed/train_eta_features.parquet"
)

ml_df.to_parquet(
    output_file,
    index=False
)

# 14. Display final report
print("\n========== FEATURE ENGINEERING REPORT ==========")

print("Original records:", len(df))
print("Final ML records:", len(ml_df))

print(
    "Unique trains:",
    ml_df["train_number"].nunique()
)

print(
    "Unique stations:",
    ml_df["station_code"].nunique()
)

print(
    "Unique journey dates:",
    ml_df["journey_date"].nunique()
)

print("\nFeature columns:")
print(ml_df.columns.tolist())

print("\nTarget statistics:")
print(ml_df["arrival_delay_min"].describe())

print("\nMissing values:")
print(ml_df.isnull().sum())

print("\nSaved:", output_file)

print("\nFEATURE ENGINEERING COMPLETED!")
