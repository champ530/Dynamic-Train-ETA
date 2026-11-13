
import os
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "pjay-in/indian-railways-delays"

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

MIN_STATIONS = 3
MAX_ROWS_PER_PARTITION = 3000

all_data = []

print("BUILDING COMPLETE JOURNEY DATASET...")

for date in SELECTED_DATES:

    year, month, day = date.split("-")

    filename = (
        f"2025-02-08_2026-02-07/"
        f"{year}/{month}/{date}.parquet"
    )

    print(f"\nProcessing {date}...")

    try:
        local_file = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset"
        )

        daily_df = pd.read_parquet(local_file)

        daily_df["start_date"] = pd.to_datetime(
            daily_df["start_date"],
            errors="coerce"
        )

        daily_df["_journey_date"] = (
            daily_df["start_date"].dt.date
        )

        # Identify complete train journeys
        grouped = daily_df.groupby(
            ["train_number", "_journey_date"],
            sort=False,
            dropna=False
        )

        journey_groups = []

        for _, journey in grouped:

            journey = journey.sort_values("seq")

            # Keep journeys with at least 3 station records
            if len(journey) < MIN_STATIONS:
                continue

            journey_groups.append(journey)

        # Randomize complete journeys, not individual rows
        generator = pd.Series(
            range(len(journey_groups))
        )

        if len(generator) > 0:
            generator = generator.sample(
                frac=1,
                random_state=42
            )

        selected = []
        row_count = 0

        for index in generator:

            journey = journey_groups[index]

            if (
                row_count + len(journey)
                <= MAX_ROWS_PER_PARTITION
            ):
                selected.append(journey)
                row_count += len(journey)

        if selected:
            date_sample = pd.concat(
                selected,
                ignore_index=True
            )

            all_data.append(date_sample)

        print("Available journeys:", len(journey_groups))
        print("Selected journeys:", len(selected))
        print("Selected station records:", row_count)

    except Exception as error:
        print("Error processing", date, ":", error)

if not all_data:
    raise RuntimeError(
        "No complete journeys found. Check the dataset files."
    )

# Combine selected journeys
final_df = pd.concat(
    all_data,
    ignore_index=True
)

final_df = final_df.drop(
    columns=["_journey_date"],
    errors="ignore"
)

# Remove repeated journey-station keys only after
# checking whether they exist
duplicate_mask = final_df.duplicated(
    subset=["train_number", "start_date", "seq"],
    keep=False
)

print("\nDuplicate journey-station rows:", duplicate_mask.sum())

# Sort chronologically
final_df = final_df.sort_values(
    ["start_date", "train_number", "seq"]
).reset_index(drop=True)

# Save separately
os.makedirs("data/processed", exist_ok=True)

output_file = (
    "data/processed/complete_journey_sample.parquet"
)

final_df.to_parquet(
    output_file,
    index=False
)

journeys = final_df.groupby(
    ["train_number", "start_date"]
).size()

print("\n================================")
print("COMPLETE JOURNEY DATASET READY")
print("Total records:", len(final_df))
print("Unique journeys:", len(journeys))
print("Average stations per journey:", journeys.mean())
print("Minimum stations:", journeys.min())
print("Maximum stations:", journeys.max())
print("Saved:", output_file)
print("================================")
