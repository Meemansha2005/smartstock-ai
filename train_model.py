
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Project paths
DATA_PATH = Path("data/train.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "demand_model.joblib"

# Load and prepare data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Place train.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)
    required_columns = {"date", "store", "item", "sales"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain these columns: {required_columns}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["date", "store", "item", "sales"])

    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    return df


# Create historical sales features
def create_features(df):
    data = df.copy()
    grouped = data.groupby(["store", "item"], sort=False)["sales"]

    data["lag_1"] = grouped.shift(1)
    data["lag_7"] = grouped.shift(7)
    data["lag_14"] = grouped.shift(14)

    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_month"] = data["date"].dt.day
    data["month"] = data["date"].dt.month

    return data.dropna(
        subset=["lag_1", "lag_7", "lag_14"]
    ).reset_index(drop=True)


def main():
    print("Loading sales dataset...")
    df = load_data()

    print(f"Rows loaded: {len(df):,}")
    print(f"Date range: {df['date'].min().date()} to "
          f"{df['date'].max().date()}")
    print(f"Stores: {df['store'].nunique()}")
    print(f"Products: {df['item'].nunique()}")

    featured = create_features(df)

    # Chronological split: final 30 calendar days are the test period
    cutoff = featured["date"].max() - pd.Timedelta(days=30)
    train = featured[featured["date"] <= cutoff].copy()
    test = featured[featured["date"] > cutoff].copy()

    feature_columns = [
        "store",
        "item",
        "lag_1",
        "lag_7",
        "lag_14",
        "day_of_week",
        "day_of_month",
        "month",
    ]

    if train.empty or test.empty:
        raise ValueError("Not enough date coverage to create a train/test split.")

    X_train = train[feature_columns]
    y_train = train["sales"]
    X_test = test[feature_columns]
    y_test = test["sales"]

    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = np.maximum(0, model.predict(X_test))
    mae = mean_absolute_error(y_test, predictions)

    # Simple baseline: use previous day's sales
    baseline_predictions = test["lag_1"].to_numpy()
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)

    print("\n--- Model Evaluation ---")
    print(f"Random Forest MAE: {mae:.3f} units")
    print(f"Previous-day baseline MAE: {baseline_mae:.3f} units")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model and its metadata
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "mae": float(mae),
            "baseline_mae": float(baseline_mae),
            "last_date": df["date"].max(),
        },
        MODEL_PATH,
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print("Training complete!")


if __name__ == "__main__":
    main()
