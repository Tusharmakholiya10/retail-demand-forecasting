from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DATA_PATH = Path("data/processed")
FORECAST_DATA_PATH = Path("data/forecasts")

TRAIN_FILE = (
    PROCESSED_DATA_PATH /
    "train_data.csv"
)

VALIDATION_FILE = (
    PROCESSED_DATA_PATH /
    "validation_data.csv"
)

RESULTS_FILE = (
    FORECAST_DATA_PATH /
    "baseline_results.csv"
)


# ============================================================
# METRICS
# ============================================================

def calculate_mae(actual, predicted):

    return np.mean(
        np.abs(actual - predicted)
    )


def calculate_rmse(actual, predicted):

    return np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )


def calculate_rmsle(actual, predicted):

    actual = np.maximum(
        actual,
        0
    )

    predicted = np.maximum(
        predicted,
        0
    )

    return np.sqrt(
        np.mean(
            (
                np.log1p(actual)
                -
                np.log1p(predicted)
            ) ** 2
        )
    )


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

def load_validation_data():

    print("=" * 70)
    print("BASELINE FORECASTING MODEL")
    print("=" * 70)

    print("\nLoading validation data...")

    validation = pd.read_csv(
        VALIDATION_FILE,
        parse_dates=["date"]
    )

    print(
        f"Validation shape: "
        f"{validation.shape}"
    )

    print(
        f"Validation dates: "
        f"{validation['date'].min().date()} "
        f"-> "
        f"{validation['date'].max().date()}"
    )

    return validation


# ============================================================
# CREATE BASELINE PREDICTIONS
# ============================================================

def create_predictions(df):

    print("\n" + "=" * 70)
    print("CREATING BASELINE PREDICTIONS")
    print("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # Verify required columns
    # --------------------------------------------------------

    required_columns = [
        "sales",
        "lag_1",
        "lag_7",
        "lag_28"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Baseline 1
    # Yesterday's sales
    # --------------------------------------------------------

    print(
        "\nCreating Naive baseline..."
    )

    df["prediction_naive"] = (
        df["lag_1"]
    )

    # --------------------------------------------------------
    # Baseline 2
    # Same day last week
    # --------------------------------------------------------

    print(
        "Creating Weekly Seasonal Naive baseline..."
    )

    df["prediction_weekly"] = (
        df["lag_7"]
    )

    # --------------------------------------------------------
    # Baseline 3
    # Same period 28 days ago
    # --------------------------------------------------------

    print(
        "Creating 28-Day Seasonal Naive baseline..."
    )

    df["prediction_28day"] = (
        df["lag_28"]
    )

    return df


# ============================================================
# EVALUATE BASELINES
# ============================================================

def evaluate_baselines(df):

    print("\n" + "=" * 70)
    print("EVALUATING BASELINES")
    print("=" * 70)

    results = []

    baselines = {
        "Naive_Lag_1": "prediction_naive",
        "Weekly_Seasonal_Lag_7": "prediction_weekly",
        "Seasonal_Lag_28": "prediction_28day"
    }

    for model_name, prediction_column in baselines.items():

        # ----------------------------------------------------
        # Remove rows where prediction is unavailable
        # ----------------------------------------------------

        evaluation_data = df[
            df[prediction_column].notna()
        ].copy()

        actual = (
            evaluation_data["sales"]
            .values
        )

        predicted = (
            evaluation_data[prediction_column]
            .values
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        mae = calculate_mae(
            actual,
            predicted
        )

        rmse = calculate_rmse(
            actual,
            predicted
        )

        rmsle = calculate_rmsle(
            actual,
            predicted
        )

        results.append(
            {
                "model": model_name,
                "rows_evaluated": len(
                    evaluation_data
                ),
                "MAE": mae,
                "RMSE": rmse,
                "RMSLE": rmsle
            }
        )

        print(
            f"\n{model_name}"
        )

        print(
            f"Rows evaluated: "
            f"{len(evaluation_data):,}"
        )

        print(
            f"MAE   : {mae:,.4f}"
        )

        print(
            f"RMSE  : {rmse:,.4f}"
        )

        print(
            f"RMSLE : {rmsle:,.4f}"
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    print("\n" + "=" * 70)
    print("SAVING BASELINE RESULTS")
    print("=" * 70)

    FORECAST_DATA_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    results = results.sort_values(
        "RMSLE"
    )

    results.to_csv(
        RESULTS_FILE,
        index=False
    )

    print(
        f"\n✅ Results saved to:"
        f"\n{RESULTS_FILE}"
    )

    print("\nBaseline comparison:")

    print(
        results.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    validation = (
        load_validation_data()
    )

    validation = (
        create_predictions(
            validation
        )
    )

    results = (
        evaluate_baselines(
            validation
        )
    )

    save_results(
        results
    )

    print("\n" + "=" * 70)
    print("✅ BASELINE MODELING COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()