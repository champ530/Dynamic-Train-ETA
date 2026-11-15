import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RAILCORE_KEY")
BASE_URL = "https://ir.railcore.tech/v1"


def get_live_train_status(train_number):
    if not API_KEY:
        raise ValueError("RAILCORE_KEY is missing from .env")

    url = f"{BASE_URL}/trains/{train_number}/live"

    headers = {
        "X-RailCore-Key": API_KEY
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()
    result = response.json()

    if not result.get("success"):
        raise RuntimeError(result.get("error", "API request failed"))

    return result["data"]


if __name__ == "__main__":
    train_number = input("Enter train number: ").strip()

    try:
        train = get_live_train_status(train_number)

        print("\n===== LIVE TRAIN STATUS =====")
        print("Train:", train.get("train_name"))
        print("Train Number:", train.get("train_number"))
        print("Status:", train.get("status_text"))
        print("Current Station:", train.get("current_station_name"))
        print("Current Delay:", train.get("delay_minutes"), "minutes")
        print("Next Station:", train.get("next_stop", {}).get("station_name"))
        print("Destination ETA:", train.get("eta_destination_at"))
        print("Data Freshness:", train.get("freshness", {}).get("age_seconds"), "seconds")

    except Exception as error:
        print("Error:", error)