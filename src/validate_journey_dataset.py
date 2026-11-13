
import pandas as pd

FILE = "data/processed/complete_journey_sample.parquet"

print("LOADING JOURNEY DATASET...")

df = pd.read_parquet(FILE)

df["start_date"] = pd.to_datetime(
    df["start_date"],
    errors="coerce"
)

# Group records by train and journey date
journeys = df.groupby(
    ["train_number", "start_date"]
)

results = []

for (train, date), group in journeys:

    group = group.sort_values("seq")

    sequences = group["seq"].dropna().astype(int).tolist()

    if len(sequences) < 3:
        continue

    differences = [
        sequences[i + 1] - sequences[i]
        for i in range(len(sequences) - 1)
    ]

    results.append({
        "train_number": train,
        "journey_date": date,
        "station_count": len(sequences),
        "consecutive_pairs": sum(
            difference == 1 for difference in differences
        ),
        "sequence_gaps": sum(
            difference > 1 for difference in differences
        ),
        "valid_delays": group["arrival_delay_min"].notna().sum()
    })

report = pd.DataFrame(results)

print("\n===== JOURNEY VALIDATION =====")

print("Journeys checked:", len(report))

print(
    "Journeys with consecutive stations:",
    (report["sequence_gaps"] == 0).sum()
)

print(
    "Journeys with sequence gaps:",
    (report["sequence_gaps"] > 0).sum()
)

print(
    "Journeys with at least 3 valid delays:",
    (report["valid_delays"] >= 3).sum()
)

print("\n===== STATION STATISTICS =====")

print(report["station_count"].describe())

print("\n===== SEQUENCE GAP STATISTICS =====")

print(report["sequence_gaps"].describe())

# Save validation report
report.to_csv(
    "reports/phase2/journey_validation.csv",
    index=False
)

print("\nVALIDATION COMPLETED!")
print("Saved: reports/phase2/journey_validation.csv")
