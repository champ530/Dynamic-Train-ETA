
import os
import numpy as np
import pandas as pd

INPUT_FILE = "data/processed/complete_journey_sample.parquet"
OUTPUT_DIR = "data/processed"

print("Loading historical railway data...")

df = pd.read_parquet(INPUT_FILE)

# --------------------------------------------------
# 1. CLEAN AND PREPARE DATA
# --------------------------------------------------

required_columns = [
    "train_number",
    "train_name",
    "category",
    "start_date",
    "seq",
    "station_code",
    "scheduled_arrival",
    "scheduled_departure",
    "arrival_delay_min",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df["journey_date"] = pd.to_datetime(
    df["start_date"], errors="coerce"
).dt.normalize()

for col in ["scheduled_arrival", "scheduled_departure"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

for col in ["seq", "arrival_delay_min"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=["train_number", "journey_date", "seq"]
).copy()

df["seq"] = df["seq"].astype(int)

journey_keys = ["train_number", "journey_date"]

df = df.sort_values(
    journey_keys + ["seq"]
).reset_index(drop=True)

# One row per train, journey date and station sequence.
duplicates = df.duplicated(journey_keys + ["seq"]).sum()

if duplicates:
    raise ValueError(
        f"Found {duplicates} duplicate journey-sequence records. "
        "Resolve duplicates before building prediction datasets."
    )

groups = df.groupby(journey_keys, sort=False)

# --------------------------------------------------
# 2. COMMON FEATURES
# --------------------------------------------------

df["hour"] = df["scheduled_departure"].dt.hour
df["day_of_week"] = df["journey_date"].dt.dayofweek
df["month"] = df["journey_date"].dt.month
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

df["season"] = df["month"].map({
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Summer",
    4: "Summer",
    5: "Summer",
    6: "Monsoon",
    7: "Monsoon",
    8: "Monsoon",
    9: "Monsoon",
    10: "Autumn",
    11: "Autumn",
})

# Current station's observed arrival delay.
df["current_arrival_delay"] = df["arrival_delay_min"]

# Previous station's observed arrival delay.
df["previous_arrival_delay"] = groups[
    "arrival_delay_min"
].shift(1)

# Next station information.
df["next_station_code"] = groups[
    "station_code"
].shift(-1)
df["next_arrival_delay"] = groups[
    "arrival_delay_min"
].shift(-1)

df["next_scheduled_arrival"] = groups[
    "scheduled_arrival"
].shift(-1)

# Correct scheduled travel interval:
# current scheduled departure -> next scheduled arrival.
df["next_scheduled_travel_minutes"] = (
    df["next_scheduled_arrival"]
    - df["scheduled_departure"]
).dt.total_seconds() / 60

# Final station's arrival delay for the same journey.
df["last_seq"] = groups["seq"].transform("max")

final_rows = df.loc[
    df["seq"] == df["last_seq"],
    journey_keys + ["arrival_delay_min"]
].rename(
    columns={
        "arrival_delay_min": "final_arrival_delay"
    }
)

df = df.merge(
    final_rows,
    on=journey_keys,
    how="left",
    validate="many_to_one"
)

# --------------------------------------------------
# 3. NEXT-STATION PREDICTION DATASET
# --------------------------------------------------

next_df = df[
    df["next_station_code"].notna()
    & df["current_arrival_delay"].notna()
    & df["next_scheduled_arrival"].notna()
    & df["scheduled_departure"].notna()
    & (df["next_scheduled_travel_minutes"] > 0)
    & df["next_arrival_delay"].notna()
].copy() if "next_arrival_delay" in df.columns else pd.DataFrame()



next_features = [
    "train_number",
    "category",
    "station_code",
    "next_station_code",
    "seq",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "season",
    "current_arrival_delay",
    "previous_arrival_delay",
    "next_scheduled_travel_minutes",
]

next_df = next_df[
    next_features + [
        "next_arrival_delay",
        "journey_date",
    ]
].copy()

next_df = next_df.replace(
    [np.inf, -np.inf], np.nan
)

next_df = next_df.dropna(
    subset=[
        "train_number",
        "station_code",
        "next_station_code",
        "current_arrival_delay",
        "next_scheduled_travel_minutes",
        "next_arrival_delay",
        "journey_date",
    ]
)

# --------------------------------------------------
# 4. FINAL-DESTINATION PREDICTION DATASET
# --------------------------------------------------

final_df = df[
    (df["seq"] < df["last_seq"])
    & df["current_arrival_delay"].notna()
    & df["final_arrival_delay"].notna()
    & df["scheduled_departure"].notna()
].copy()



# Calculate remaining scheduled time using each journey's
# final scheduled arrival minus the current scheduled departure.
final_schedule = df.loc[
    df["seq"] == df["last_seq"],
    journey_keys + ["scheduled_arrival"]
].rename(
    columns={
        "scheduled_arrival": "final_scheduled_arrival"
    }
)

final_df = final_df.merge(
    final_schedule,
    on=journey_keys,
    how="left",
    validate="many_to_one"
)

final_df["scheduled_minutes_to_destination"] = (
    final_df["final_scheduled_arrival"]
    - final_df["scheduled_departure"]
).dt.total_seconds() / 60

final_features = [
    "train_number",
    "category",
    "station_code",
    "seq",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "season",
    "current_arrival_delay",
    "previous_arrival_delay",
    "scheduled_minutes_to_destination",
]

final_df = final_df[
    final_features + [
        "final_arrival_delay",
        "journey_date",
    ]
].copy()

final_df = final_df.replace(
    [np.inf, -np.inf], np.nan
)

final_df = final_df.dropna(
    subset=[
        "train_number",
        "station_code",
        "current_arrival_delay",
        "scheduled_minutes_to_destination",
        "final_arrival_delay",
        "journey_date",
    ]
)

final_df = final_df[
    final_df["scheduled_minutes_to_destination"] > 0
].copy()

# --------------------------------------------------
# 5. SAVE DATASETS
# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

next_path = os.path.join(
    OUTPUT_DIR,
    "next_station_dataset.parquet"
)

final_path = os.path.join(
    OUTPUT_DIR,
    "final_destination_dataset.parquet"
)

next_df.to_parquet(next_path, index=False)
final_df.to_parquet(final_path, index=False)

# --------------------------------------------------
# 6. DATASET SUMMARY
# --------------------------------------------------

print("\n========== DATASET BUILD REPORT ==========")

print("\nNEXT-STATION PREDICTION")
print("Rows:", len(next_df))
print("Unique journeys:", next_df[journey_keys].drop_duplicates().shape[0])
print("Missing values:", int(next_df.isna().sum().sum()))
print("Target statistics:")
print(next_df["next_arrival_delay"].describe())

print("\nFINAL-DESTINATION PREDICTION")
print("Rows:", len(final_df))
print("Unique journeys:", final_df[journey_keys].drop_duplicates().shape[0])
print("Missing values:", int(final_df.isna().sum().sum()))
print("Target statistics:")
print(final_df["final_arrival_delay"].describe())

print("\nSaved files:")
print(next_path)
print(final_path)

print("\nDATASET PREPARATION COMPLETED")
