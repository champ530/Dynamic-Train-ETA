# RailSense — Dynamic Train ETA Prediction & Delay Intelligence System

RailSense is a machine-learning-powered railway intelligence dashboard designed to estimate train delays at the next station and final destination.

## Features

- Train number-based live status lookup
- Current reported train location
- Interactive railway route map
- Station-by-station journey progress
- Machine learning-based delay predictions
- Next-station and final-destination estimates
- Data quality and freshness monitoring

## Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Plotly
- Folium
- Joblib

## Project Structure

- `app/` — Streamlit dashboard
- `src/` — Prediction and data-processing modules
- `models/` — Trained machine learning models
- `data/` — Historical and processed railway datasets
- `reports/` — Model evaluation and analysis reports

## Run Locally

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app/app.py
```

## Important Notes

Live train information depends on the configured third-party train-status provider. Train locations represent provider-reported positions, not continuous GPS tracking. Delay predictions are estimates and are not guaranteed arrival times.

## Project Type

Academic machine learning and data science project.