
import pandas as pd
import glob
import os

print("===================================")
print("INDIAN RAILWAYS DATASET VERIFICATION")
print("===================================")

# Find downloaded Parquet files
files = sorted(glob.glob("data/raw/*.parquet"))

print("\nNumber of files found:", len(files))

if not files:
    print("ERROR: No Parquet files found!")
    raise SystemExit(1)

# Read all downloaded files
dataframes = []

for file in files:
    print("\nReading:", file)

    df_part = pd.read_parquet(file)

    print("Records:", len(df_part))

    dataframes.append(df_part)

# Combine batches
df = pd.concat(dataframes, ignore_index=True)

print("\n========== DATASET INFORMATION ==========")

print("Total rows:", df.shape[0])
print("Total columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nFirst five records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())

# Save combined dataset
os.makedirs("data/processed", exist_ok=True)

df.to_parquet(
    "data/processed/train_delay_combined.parquet",
    index=False
)

print("\nDATASET VERIFICATION COMPLETED")
