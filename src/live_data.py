
import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://ir.railcore.tech/v1"


def get_api_key():
    # First, check the local .env file
    api_key = os.getenv("RAILCORE_KEY")

    if api_key:
        return api_key

    # If not found, check Streamlit Cloud Secrets
    try:
        return st.secrets.get("RAILCORE_KEY")
    except Exception:
        return None


def get_live_train_status(train_number):
    api_key = get_api_key()

    if not api_key:
        raise ValueError(
            "RAILCORE_KEY is missing. Add it to your local .env "
            "or Streamlit Cloud Secrets."
        )

    url = f"{BASE_URL}/trains/{train_number}/live"

    headers = {
        "X-RailCore-Key": api_key
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()
    result = response.json()
    print("RAILCORE TOP-LEVEL KEYS:", list(result.keys()))

    payload = result.get("data", {})

    print("RAILCORE DATA TYPE:", type(payload).__name__)

    if isinstance(payload, dict):
        print("RAILCORE DATA KEYS:", list(payload.keys()))

        stations = payload.get("stations")
        print("STATIONS TYPE:", type(stations).__name__)
        print("STATIONS COUNT:", len(stations) if isinstance(stations, list) else "Not a list")

    if not result.get("success", True):
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