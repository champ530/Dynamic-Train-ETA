
import os
import pandas as pd
from datasets import load_dataset
from itertools import islice

DATASET_NAME = "pjay-in/indian-railways-delays"
CONFIG_NAME = "2025-02-08_2026-02-07"

BATCH_SIZE = 100_000

OUTPUT_DIR = "data/raw/full_history"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Connecting to Hugging Face...")
print("Loading complete historical dataset...")

dataset = load_dataset(
    DATASET_NAME,
    CONFIG_NAME,
    split="train",
    streaming=True
)

iterator = iter(dataset)

total_records = 0
batch_number = 1

while True:

    print(f"\nDownloading batch {batch_number}...")

    batch = list(islice(iterator, BATCH_SIZE))

    if not batch:
        break

    df = pd.DataFrame(batch)

    filename = os.path.join(
        OUTPUT_DIR,
        f"history_part_{batch_number:04d}.parquet"
    )

    df.to_parquet(filename, index=False)

    total_records += len(df)

    print("Batch records:", len(df))
    print("Total downloaded:", total_records)

    batch_number += 1

print("\n================================")
print("FULL HISTORICAL DOWNLOAD FINISHED")
print("Total records:", total_records)
print("Files created:", batch_number - 1)
print("Saved in:", OUTPUT_DIR)
print("================================")
