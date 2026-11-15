
import os
import joblib
import pandas as pd

from datetime import datetime
from zoneinfo import ZoneInfo

from src.live_data import get_live_train_status


MODEL_DIR = "models"
IST = ZoneInfo("Asia/Kolkata")

# Best-performing models from the current test results
next_station_model = joblib.load(
    os.path.join(MODEL_DIR, "next_station_polynomial.joblib")
)

final_destination_model = joblib.load(
    os.path.join(MODEL_DIR, "final_destination_linear.joblib")
)


def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Summer"
    if month in [6, 7, 8, 9]:
        return "Monsoon"
    return "Autumn"


def parse_time(value):
    if not value:
        return None

    try:
        return pd.to_datetime(value)
    except (ValueError, TypeError):
        return None


def predict_live_train(train_number):
    # Fetch current train information
    data = get_live_train_status(str(train_number))

    stations = data.get("stations", [])

    if not stations:
        raise ValueError("The API did not return station information.")

    current_code = data.get("current_station_code")

    if not current_code:
        raise ValueError("Current station code is unavailable.")

    # Find the current station in the route
    current_index = next(
        (
            i for i, station in enumerate(stations)
            if station.get("station_code") == current_code
        ),
        None
    )

    if current_index is None:
        raise ValueError("Current station was not found in the route.")

    if current_index >= len(stations) - 1:
        raise ValueError("No next station is available.")

    current = stations[current_index]
    next_stop = stations[current_index + 1]

    destination = next(
        (
            station for station in reversed(stations)
            if station.get("stop_type") == "DESTINATION"
        ),
        stations[-1]
    )
    print("\n--- RAILSENSE MAP DATA ---")
    print("CURRENT STATION:", current)
    print("NEXT STATION:", next_stop)
    print("DESTINATION:", destination)


    # Read live delay at the current station
    current_delay = data.get("delay_minutes")

    if current_delay is None:
        current_delay = current.get("delay_arrival_minutes")

    if current_delay is None:
        raise ValueError("Current train delay is unavailable.")

    # Previous station delay, if available
    previous_delay = 0

    if current_index > 0:
        previous_delay = stations[current_index - 1].get(
            "delay_arrival_minutes"
        )

    if previous_delay is None:
        previous_delay = current_delay

    # Use scheduled times, not predicted ETAs
    current_departure = parse_time(
        current.get("scheduled_departure")
    )

    next_arrival = parse_time(
        next_stop.get("scheduled_arrival")
    )

    destination_arrival = parse_time(
        destination.get("scheduled_arrival")
    )

    if current_departure is None or next_arrival is None:
        raise ValueError("Scheduled station timings are incomplete.")

    next_minutes = (
        next_arrival - current_departure
    ).total_seconds() / 60

    if next_minutes <= 0:
        raise ValueError("Invalid next-station scheduled travel time.")

    if destination_arrival is None:
        raise ValueError("Destination scheduled arrival is unavailable.")

    destination_minutes = (
        destination_arrival - current_departure
    ).total_seconds() / 60

    if destination_minutes <= 0:
        raise ValueError("Invalid destination scheduled travel time.")

    # Build features using the same names as the training pipeline
    now = datetime.now(IST)

    features = {
        "seq": current.get("sequence"),
        "hour": now.hour,
        "day_of_week": now.weekday(),
        "month": now.month,
        "is_weekend": int(now.weekday() >= 5),
        "current_arrival_delay": float(current_delay),
        "previous_arrival_delay": float(previous_delay),
        "train_number": str(train_number),
        "category": data.get("category", "Unknown"),
        "station_code": str(current_code),
        "season": get_season(now.month),
    }

    next_features = pd.DataFrame([{
        **features,
        "next_scheduled_travel_minutes": next_minutes
    }])

    final_features = pd.DataFrame([{
        **features,
        "scheduled_minutes_to_destination": destination_minutes
    }])

    # Generate AI predictions
    next_prediction = float(
        next_station_model.predict(next_features)[0]
    )

    final_prediction = float(
        final_destination_model.predict(final_features)[0]
    )

    freshness = data.get("freshness", {})

    
    return {
        "train_name": data.get("train_name", "Unknown"),
        "train_number": str(train_number),
        "status": data.get("status_text", "Unknown"),

        "current_station": current.get("station_name"),
        "next_station": next_stop.get("station_name"),
        "destination": destination.get("station_name"),

        "reported_delay": current_delay,
        "predicted_next_delay": round(next_prediction, 2),
        "predicted_destination_delay": round(final_prediction, 2),

        "data_age_seconds": freshness.get("age_seconds"),
        "data_state": freshness.get("state", "Unknown"),

        "current_latitude": current.get("latitude"),
        "current_longitude": current.get("longitude"),

        "next_latitude": next_stop.get("latitude"),
        "next_longitude": next_stop.get("longitude"),

        "destination_latitude": destination.get("latitude"),
        "destination_longitude": destination.get("longitude"),

        "route_stations": [
            {
                "station_name": station.get("station_name"),
                "station_code": station.get("station_code"),
                "latitude": station.get("latitude"),
                "longitude": station.get("longitude"),
                "sequence": station.get("sequence"),
                "stop_type": station.get("stop_type"),
                "is_stopping": station.get("is_stopping")
            }
            for station in stations
            if station.get("latitude") is not None
            and station.get("longitude") is not None
        ],

        "updated_at": datetime.now(IST).isoformat()
    }



if __name__ == "__main__":
    train_number = input("Enter train number: ").strip()

    try:
        result = predict_live_train(train_number)

        print("\n========== LIVE TRAIN ETA INTELLIGENCE ==========")
        print("Train:", result["train_name"])
        print("Train Number:", result["train_number"])
        print("Status:", result["status"])
        print("Current Station:", result["current_station"])
        print("Next Station:", result["next_station"])
        print("Destination:", result["destination"])
        print("Currently Reported Delay:", result["reported_delay"], "minutes")

        print("\n========== AI PREDICTIONS ==========")
        print(
            "Next Station Predicted Delay:",
            result["predicted_next_delay"], "minutes"
        )
        print(
            "Final Destination Predicted Delay:",
            result["predicted_destination_delay"], "minutes"
        )

        print("\n========== DATA QUALITY ==========")
        print("Data State:", result["data_state"])
        print("Data Age:", result["data_age_seconds"], "seconds")
        print("Updated At:", result["updated_at"])

    except Exception as error:
        print("\nPrediction could not be completed:", error)
