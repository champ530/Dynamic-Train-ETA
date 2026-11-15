import joblib
import pandas as pd

# Load trained models
next_station_model = joblib.load(
    "models/next_station_polynomial.joblib"
)

final_destination_model = joblib.load(
    "models/final_destination_linear.joblib"
)

# Sample input
sample = pd.DataFrame([{
    "train_number": "12951",
    "category": "Express",
    "station_code": "NDLS",
    "next_station_code": "MTJ",
    "seq": 5,
    "hour": 14,
    "day_of_week": 2,
    "month": 11,
    "is_weekend": 0,
    "season": "Autumn",
    "current_arrival_delay": 15,
    "previous_arrival_delay": 10,
"next_scheduled_travel_minutes": 90,
"scheduled_minutes_to_destination": 240
}])

# Generate predictions
next_delay = next_station_model.predict(sample)[0]

final_delay = final_destination_model.predict(sample)[0]

print("\n===== TRAIN ETA PREDICTIONS =====")
print(f"Predicted next-station delay: {next_delay:.2f} minutes")
print(f"Predicted final-destination delay: {final_delay:.2f} minutes")