
import pandas as pd

# Load original combined dataset
df = pd.read_parquet(
    "data/processed/train_delay_combined.parquet"
)

print("===== ORIGINAL DATE INFORMATION =====")

print("Original start_date dtype:")
print(df["start_date"].dtype)

print("\nOriginal date samples:")
print(df["start_date"].head(20).to_string(index=False))

print("\nUnique original date values:")
print(df["start_date"].nunique(dropna=True))

print("\nMissing original dates:")
print(df["start_date"].isna().sum())

# Convert dates for analysis
parsed_dates = pd.to_datetime(
    df["start_date"],
    errors="coerce"
)

print("\n===== DATE CONVERSION CHECK =====")

print("Successfully parsed:", parsed_dates.notna().sum())

print("Failed conversions:", parsed_dates.isna().sum())

print("Unique parsed dates:", parsed_dates.nunique())

print("Earliest date:", parsed_dates.min())

print("Latest date:", parsed_dates.max())

print("\n===== DAILY RECORD COUNTS =====")

daily = pd.DataFrame({
    "journey_date": parsed_dates,
    "arrival_delay_min": df["arrival_delay_min"]
})

daily["journey_date"] = daily["journey_date"].dt.date

print(
    daily.groupby("journey_date")
    .agg(
        total_records=("arrival_delay_min", "size"),
        valid_delays=("arrival_delay_min", "count"),
        mean_delay=("arrival_delay_min", "mean")
    )
    .to_string()
)

print("\n===== FINAL CHECK =====")

print("Total records:", len(df))
print("Valid arrival delays:", df["arrival_delay_min"].notna().sum())

print("\nDATE COVERAGE CHECK COMPLETED!")
