
import os
import pandas as pd
from huggingface_hub import hf_hub_download

# Dataset details
REPO_ID = "pjay-in/indian-railways-delays"
REPO_TYPE = "dataset"

# Three selected dates per month
SELECTED_DATES = [
    "2025-02-10", "2025-02-18", "2025-02-26",
    "2025-03-05", "2025-03-15", "2025-03-25",
    "2025-04-05", "2025-04-15", "2025-04-25",
    "2025-05-05", "2025-05-15", "2025-05-25",
    "2025-06-05", "2025-06-15", "2025-06-25",
    "2025-07-05", "2025-07-15", "2025-07-25",
    "2025-08-05", "2025-08-15", "2025-08-25",
    "2025-09-05", "2025-09-15", "2025-09-25",
    "2025-10-05", "2025-10-15", "2025-10-25",
    "2025-11-05", "2025-11-15", "2025-11-25",
    "2025-12-05", "2025-12-15", "2025-12-25",
    "2026-01-05", "2026-01-15", "2026-01-25",
    "2026-02-05"
]

MAX_PER_DAY = 3000

OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_samples = []

for date in SELECTED_DATES:

    print(f"\nDownloading date: {date}")

    year, month, day = date.split("-")

    filename = (
        f"2025-02-08_2026-02-07/"
        f"{year}/{month}/{date}.parquet"
    )

    try:
        local_file = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type=REPO_TYPE
        )

        daily_df = pd.read_parquet(local_file)

        # Sample records from this date
        sample_size = min(
            MAX_PER_DAY,
            len(daily_df)
        )

        daily_sample = daily_df.sample(
            n=sample_size,
            random_state=42
        )

        all_samples.append(daily_sample)

        print("Available:", len(daily_df))
        print("Selected:", len(daily_sample))

    except Exception as e:
        print(f"Could not download {date}: {e}")

if not all_samples:
    raise RuntimeError("No daily datasets were downloaded.")

# Combine selected dates
final_df = pd.concat(
    all_samples,
    ignore_index=True
)

# Sort chronologically
final_df["start_date"] = pd.to_datetime(
    final_df["start_date"],
    errors="coerce"
)

final_df = final_df.sort_values(
    ["start_date", "train_number", "seq"]
).reset_index(drop=True)

# Save new sample
output_file = (
    "data/processed/railway_historical_sample.parquet"
)

final_df.to_parquet(
    output_file,
    index=False
)

print("\n================================")
print("SELECTED DATASET READY")
print("Total records:", len(final_df))
print("Unique dates:", final_df["start_date"].nunique())
print("Saved:", output_file)
print("================================")
