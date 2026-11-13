import os
import pandas as pd
from datasets import load_dataset
from itertools import islice

# STEP 1: Dataset details
DATASET_NAME = "pjay-in/indian-railways-delays"
CONFIG_NAME = "2025-02-08_2026-02-07"

# STEP 2: Download settings
MAX_ROWS = 100_000
BATCH_SIZE = 25_000

# STEP 3: Create destination folder
os.makedirs("data/raw", exist_ok=True)

print("Connecting to Hugging Face...")

# STEP 4: Load dataset in streaming mode
dataset = load_dataset(
    DATASET_NAME,
    CONFIG_NAME,
    split="train",
    streaming=True
)

iterator = iter(dataset)

# STEP 5: Download records in batches
total_downloaded = 0
batch_number = 1

while total_downloaded < MAX_ROWS:

    remaining = MAX_ROWS - total_downloaded

    batch = list(islice(
        iterator,
        min(BATCH_SIZE, remaining)
    ))

    if not batch:
        print("No more records available.")
        break

    df = pd.DataFrame(batch)

    filename = (
        f"data/raw/train_delay_part_{batch_number}.parquet"
    )

    df.to_parquet(filename, index=False)

    total_downloaded += len(df)

    print(f"Downloaded: {total_downloaded} records")

    batch_number += 1

# STEP 6: Completion message
print("\n================================")
print("DATASET DOWNLOAD COMPLETED")
print("Total records:", total_downloaded)
print("Saved location: data/raw/")
print("================================")

