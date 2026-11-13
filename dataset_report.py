
import pandas as pd

# Load our combined railway dataset
file_path = "data/processed/train_delay_combined.parquet"

print("Loading Indian Railways dataset...")

df = pd.read_parquet(file_path)

# 1. Dataset dimensions
print("\n========== DATASET SIZE ==========")
print("Total records:", df.shape[0])
print("Total columns:", df.shape[1])

# 2. Column names
print("\n========== COLUMN NAMES ==========")
for column in df.columns:
    print(column)

# 3. Data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# 4. Missing values
print("\n========== MISSING VALUES ==========")
missing = df.isnull().sum()
print(missing)

# 5. Missing value percentages
print("\n========== MISSING VALUE PERCENTAGE ==========")
print((df.isnull().mean() * 100).round(2))

# 6. Duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print(df.duplicated().sum())

# 7. Arrival delay statistics
print("\n========== ARRIVAL DELAY STATISTICS ==========")
print(df["arrival_delay_min"].describe())

# 8. Sample records
print("\n========== FIRST 10 RECORDS ==========")
print(df.head(10).to_string())

print("\nDATASET REPORT COMPLETED!")
