# ==============================================================
# FINAL MODEL EVALUATION
# Demand Forecasting Project
# ==============================================================

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt


# ==============================================================
# PATHS
# ==============================================================

TRAIN_PATH = "data/processed/train_data.csv"
VALIDATION_PATH = "data/processed/validation_data.csv"

MODEL_PATH = "models/lightgbm/tuned_lightgbm_model.txt"

OUTPUT_DIR = "data/forecasts/final_evaluation"
FIGURE_DIR = "reports/figures/final_evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ==============================================================
# FEATURES
# ==============================================================

FEATURES = [
    "store_nbr",
    "family",
    "onpromotion",
    "city",
    "state",
    "type",
    "cluster",
    "oil_price",
    "year",
    "month",
    "quarter",
    "week_of_year",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "has_national_holiday",
    "is_regional_holiday",
    "is_local_holiday",
    "log_onpromotion",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
]

CATEGORICAL_FEATURES = [
    "family",
    "city",
    "state",
    "type",
]


# ==============================================================
# METRICS
# ==============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    error = actual - predicted

    mae = np.mean(np.abs(error))

    rmse = np.sqrt(np.mean(error ** 2))

    rmsle = np.sqrt(
        np.mean(
            (
                np.log1p(np.maximum(actual, 0))
                - np.log1p(np.maximum(predicted, 0))
            ) ** 2
        )
    )

    # WAPE
    denominator = np.sum(np.abs(actual))

    if denominator != 0:
        wape = np.sum(np.abs(error)) / denominator * 100
    else:
        wape = np.nan

    # MAPE
    non_zero = actual != 0

    if np.sum(non_zero) > 0:
        mape = (
            np.mean(
                np.abs(
                    (actual[non_zero] - predicted[non_zero])
                    / actual[non_zero]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "RMSLE": rmsle,
        "WAPE_percent": wape,
        "MAPE_percent": mape,
    }


# ==============================================================
# LOAD DATA
# ==============================================================

def load_validation_data():

    print("=" * 70)
    print("FINAL MODEL EVALUATION")
    print("=" * 70)

    print("\nLoading validation data...")

    df = pd.read_csv(
        VALIDATION_PATH,
        parse_dates=["date"]
    )

    print(f"Validation shape: {df.shape}")

    return df


# ==============================================================
# PREPARE FEATURES
# ==============================================================

def prepare_features(df):

    print("\nPreparing features...")

    X = df[FEATURES].copy()

    for column in CATEGORICAL_FEATURES:

        if column in X.columns:
            X[column] = X[column].astype("category")

    return X


# ==============================================================
# LOAD MODEL
# ==============================================================

def load_model():

    print("\nLoading tuned LightGBM model...")

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = lgb.Booster(
        model_file=MODEL_PATH
    )

    print("✅ Tuned LightGBM model loaded.")

    return model


# ==============================================================
# GENERATE PREDICTIONS
# ==============================================================

def generate_predictions(model, X):

    print("\nGenerating predictions...")

    predictions = model.predict(X)

    # Sales cannot be negative
    predictions = np.maximum(predictions, 0)

    print("✅ Predictions generated.")

    return predictions


# ==============================================================
# OVERALL EVALUATION
# ==============================================================

def overall_evaluation(df):

    print("\n" + "=" * 70)
    print("OVERALL MODEL PERFORMANCE")
    print("=" * 70)

    metrics = calculate_metrics(
        df["sales"],
        df["prediction"]
    )

    result = pd.DataFrame([metrics])

    print("\nFinal tuned LightGBM performance:\n")

    for metric, value in metrics.items():

        print(
            f"{metric:<18}: {value:.4f}"
        )

    output_path = os.path.join(
        OUTPUT_DIR,
        "overall_metrics.csv"
    )

    result.to_csv(
        output_path,
        index=False
    )

    print(
        f"\n✅ Overall metrics saved:\n{output_path}"
    )

    return metrics


# ==============================================================
# ERROR ANALYSIS
# ==============================================================

def create_error_columns(df):

    df["error"] = (
        df["sales"] - df["prediction"]
    )

    df["absolute_error"] = (
        np.abs(df["error"])
    )

    df["squared_error"] = (
        df["error"] ** 2
    )

    df["percentage_error"] = np.where(
        df["sales"] != 0,
        (
            df["absolute_error"]
            / df["sales"]
        ) * 100,
        np.nan
    )

    return df


# ==============================================================
# MONTHLY PERFORMANCE
# ==============================================================

def monthly_performance(df):

    print("\nCalculating monthly performance...")

    df["year_month"] = (
        df["date"]
        .dt.to_period("M")
        .astype(str)
    )

    records = []

    for period, group in df.groupby("year_month"):

        metrics = calculate_metrics(
            group["sales"],
            group["prediction"]
        )

        records.append({
            "year_month": period,
            **metrics,
            "actual_sales": group["sales"].sum(),
            "predicted_sales": group["prediction"].sum(),
            "rows": len(group),
        })

    result = pd.DataFrame(records)

    path = os.path.join(
        OUTPUT_DIR,
        "monthly_performance.csv"
    )

    result.to_csv(
        path,
        index=False
    )

    print(
        f"✅ Monthly performance saved:\n{path}"
    )

    return result


# ==============================================================
# STORE PERFORMANCE
# ==============================================================

def store_performance(df):

    print("\nCalculating store performance...")

    records = []

    for store, group in df.groupby("store_nbr"):

        metrics = calculate_metrics(
            group["sales"],
            group["prediction"]
        )

        records.append({
            "store_nbr": store,
            **metrics,
            "actual_sales": group["sales"].sum(),
            "predicted_sales": group["prediction"].sum(),
            "sales_difference": (
                group["prediction"].sum()
                - group["sales"].sum()
            ),
            "rows": len(group),
        })

    result = pd.DataFrame(records)

    result = result.sort_values(
        "MAE"
    )

    path = os.path.join(
        OUTPUT_DIR,
        "store_performance.csv"
    )

    result.to_csv(
        path,
        index=False
    )

    print(
        f"✅ Store performance saved:\n{path}"
    )

    return result


# ==============================================================
# PRODUCT FAMILY PERFORMANCE
# ==============================================================

def family_performance(df):

    print("\nCalculating product-family performance...")

    records = []

    for family, group in df.groupby("family"):

        metrics = calculate_metrics(
            group["sales"],
            group["prediction"]
        )

        records.append({
            "family": family,
            **metrics,
            "actual_sales": group["sales"].sum(),
            "predicted_sales": group["prediction"].sum(),
            "sales_difference": (
                group["prediction"].sum()
                - group["sales"].sum()
            ),
            "rows": len(group),
        })

    result = pd.DataFrame(records)

    result = result.sort_values(
        "MAE"
    )

    path = os.path.join(
        OUTPUT_DIR,
        "family_performance.csv"
    )

    result.to_csv(
        path,
        index=False
    )

    print(
        f"✅ Family performance saved:\n{path}"
    )

    return result


# ==============================================================
# ACTUAL VS PREDICTED
# ==============================================================

def plot_actual_vs_predicted(df):

    print("\nCreating actual vs predicted plot...")

    daily = (
        df.groupby("date")
        .agg(
            actual_sales=("sales", "sum"),
            predicted_sales=("prediction", "sum")
        )
        .reset_index()
    )

    plt.figure(figsize=(14, 6))

    plt.plot(
        daily["date"],
        daily["actual_sales"],
        label="Actual"
    )

    plt.plot(
        daily["date"],
        daily["predicted_sales"],
        label="Predicted"
    )

    plt.title(
        "Actual vs Predicted Daily Sales"
    )

    plt.xlabel("Date")
    plt.ylabel("Sales")

    plt.legend()

    plt.xticks(rotation=45)

    plt.tight_layout()

    path = os.path.join(
        FIGURE_DIR,
        "actual_vs_predicted_daily.png"
    )

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()

    print(
        f"✅ Plot saved:\n{path}"
    )


# ==============================================================
# RESIDUAL DISTRIBUTION
# ==============================================================

def plot_residuals(df):

    print("\nCreating residual distribution...")

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["error"],
        bins=100
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.title(
        "Prediction Error Distribution"
    )

    plt.xlabel(
        "Actual Sales - Predicted Sales"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    path = os.path.join(
        FIGURE_DIR,
        "residual_distribution.png"
    )

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()

    print(
        f"✅ Residual plot saved:\n{path}"
    )


# ==============================================================
# TOP ERRORS
# ==============================================================

def save_top_errors(df):

    print("\nFinding largest prediction errors...")

    columns = [
        "date",
        "store_nbr",
        "family",
        "sales",
        "prediction",
        "error",
        "absolute_error",
        "percentage_error",
    ]

    result = (
        df[
            columns
        ]
        .sort_values(
            "absolute_error",
            ascending=False
        )
        .head(1000)
    )

    path = os.path.join(
        OUTPUT_DIR,
        "top_1000_errors.csv"
    )

    result.to_csv(
        path,
        index=False
    )

    print(
        f"✅ Top errors saved:\n{path}"
    )


# ==============================================================
# MAIN
# ==============================================================

def main():

    # Load validation data
    df = load_validation_data()

    # Prepare features
    X = prepare_features(df)

    # Load model
    model = load_model()

    # Generate predictions
    df["prediction"] = generate_predictions(
        model,
        X
    )

    # Create errors
    df = create_error_columns(df)

    # Overall evaluation
    overall_evaluation(df)

    # Monthly performance
    monthly_performance(df)

    # Store performance
    store_performance(df)

    # Family performance
    family_performance(df)

    # Top errors
    save_top_errors(df)

    # Visualizations
    plot_actual_vs_predicted(df)

    plot_residuals(df)

    # Save complete predictions
    prediction_columns = [
        "id",
        "date",
        "store_nbr",
        "family",
        "sales",
        "prediction",
        "error",
        "absolute_error",
        "percentage_error",
    ]

    prediction_path = os.path.join(
        OUTPUT_DIR,
        "final_predictions.csv"
    )

    df[
        prediction_columns
    ].to_csv(
        prediction_path,
        index=False
    )

    print(
        f"\n✅ Final predictions saved:\n"
        f"{prediction_path}"
    )

    # ==========================================================
    # FINAL SUMMARY
    # ==========================================================

    print("\n" + "=" * 70)
    print("FINAL EVALUATION COMPLETE")
    print("=" * 70)

    metrics = calculate_metrics(
        df["sales"],
        df["prediction"]
    )

    print("\n🏆 FINAL TUNED LIGHTGBM MODEL")

    print(
        f"MAE   : {metrics['MAE']:.4f}"
    )

    print(
        f"RMSE  : {metrics['RMSE']:.4f}"
    )

    print(
        f"RMSLE : {metrics['RMSLE']:.4f}"
    )

    print(
        f"WAPE  : {metrics['WAPE_percent']:.2f}%"
    )

    print(
        f"MAPE  : {metrics['MAPE_percent']:.2f}%"
    )

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nFigure directory:")
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()