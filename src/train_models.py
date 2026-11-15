
import os
import warnings
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Features available when making predictions
CATEGORICAL_FEATURES = [
    "train_number",
    "category",
    "station_code",
    "season",
]

NEXT_NUMERIC = [
    "seq",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "current_arrival_delay",
    "previous_arrival_delay",
    "next_scheduled_travel_minutes",
]

FINAL_NUMERIC = [
    "seq",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "current_arrival_delay",
    "previous_arrival_delay",
    "scheduled_minutes_to_destination",
]


def make_pipeline(numeric_features, categorical_features, polynomial=False):

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ]

    if polynomial:
        numeric_steps.append(
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False))
        )

    numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=5))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])


def train_and_evaluate(file_path, target, task_name, numeric_features):

    print("\n" + "=" * 55)
    print("TASK:", task_name)
    print("=" * 55)

    df = pd.read_parquet(file_path)

    required = (
        numeric_features
        + CATEGORICAL_FEATURES
        + [target, "journey_date"]
    )

    missing_columns = [c for c in required if c not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in {file_path}: {missing_columns}")

    df = df[required].copy()

    df["journey_date"] = pd.to_datetime(
        df["journey_date"], errors="coerce"
    )

    df[target] = pd.to_numeric(df[target], errors="coerce")

    df = df.dropna(subset=["journey_date", target])
    df = df.sort_values("journey_date").reset_index(drop=True)

    # Split by date to prevent future data entering training.
    dates = sorted(df["journey_date"].dt.date.unique())

    if len(dates) < 2:
        raise ValueError("Not enough unique journey dates for chronological split.")

    split_index = int(len(dates) * 0.80)
    split_index = max(1, min(split_index, len(dates) - 1))

    train_dates = set(dates[:split_index])
    test_dates = set(dates[split_index:])

    train = df[df["journey_date"].dt.date.isin(train_dates)].copy()
    test = df[df["journey_date"].dt.date.isin(test_dates)].copy()

    X_train = train[numeric_features + CATEGORICAL_FEATURES]
    y_train = train[target]

    X_test = test[numeric_features + CATEGORICAL_FEATURES]
    y_test = test[target]

    print("Training rows:", len(train))
    print("Testing rows:", len(test))
    print("Training dates:", len(train_dates))
    print("Testing dates:", len(test_dates))
    print("Test begins:", min(test_dates))

    results = []

    for model_name, polynomial in [
        ("Linear Regression", False),
        ("Polynomial Regression (Degree 2)", True),
    ]:

        print(f"\nTraining {model_name}...")

        pipeline = make_pipeline(
            numeric_features,
            CATEGORICAL_FEATURES,
            polynomial=polynomial
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        print("MAE:", round(mae, 2), "minutes")
        print("RMSE:", round(rmse, 2), "minutes")
        print("R²:", round(r2, 4))

        results.append({
            "task": task_name,
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        filename = (
            f"models/{task_name.lower().replace(' ', '_')}_"
            f"{'polynomial' if polynomial else 'linear'}.joblib"
        )

        joblib.dump(pipeline, filename)
        print("Saved:", filename)

    return results


if __name__ == "__main__":

    all_results = []

    all_results += train_and_evaluate(
        "data/processed/next_station_dataset.parquet",
        "next_arrival_delay",
        "Next Station",
        NEXT_NUMERIC
    )

    all_results += train_and_evaluate(
        "data/processed/final_destination_dataset.parquet",
        "final_arrival_delay",
        "Final Destination",
        FINAL_NUMERIC
    )

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        "reports/model_comparison.csv",
        index=False
    )

    print("\n" + "=" * 55)
    print("FINAL MODEL COMPARISON")
    print("=" * 55)
    print(results_df.round(3).to_string(index=False))

    print("\nSaved comparison to reports/model_comparison.csv")
    print("MODEL TRAINING COMPLETED")
