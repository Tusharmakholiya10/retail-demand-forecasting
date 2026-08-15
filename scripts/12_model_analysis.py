from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

FORECAST_PATH = Path("data/forecasts")
MODEL_PATH = Path("models")
REPORT_PATH = Path("reports/figures")

PREDICTIONS_FILE = (
    FORECAST_PATH /
    "lightgbm_predictions.csv"
)

MODEL_FILE = (
    MODEL_PATH /
    "lightgbm_model.txt"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LIGHTGBM MODEL ANALYSIS")
    print("=" * 70)

    print("\nLoading predictions...")

    df = pd.read_csv(
        PREDICTIONS_FILE,
        parse_dates=["date"]
    )

    print(
        f"Prediction rows: "
        f"{len(df):,}"
    )

    print("\nLoading LightGBM model...")

    model = lgb.Booster(
        model_file=str(
            MODEL_FILE
        )
    )

    print("✅ Model loaded.")

    return df, model


# ============================================================
# ERROR ANALYSIS
# ============================================================

def calculate_errors(df):

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    df = df.copy()

    df["absolute_error"] = (
        np.abs(
            df["sales"]
            -
            df["prediction"]
        )
    )

    df["squared_error"] = (
        (
            df["sales"]
            -
            df["prediction"]
        ) ** 2
    )

    print(
        "\nMean absolute error:"
    )

    print(
        f"{df['absolute_error'].mean():,.4f}"
    )

    return df


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def feature_importance(model):

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    importance = pd.DataFrame(
        {
            "feature": model.feature_name(),
            "importance": model.feature_importance(
                importance_type="gain"
            )
        }
    )

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\nTop 15 features:")

    print(
        importance.head(15)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    REPORT_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    top_features = (
        importance
        .head(15)
        .sort_values(
            "importance"
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )

    plt.xlabel(
        "Importance (Gain)"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "LightGBM Feature Importance"
    )

    plt.tight_layout()

    output_file = (
        REPORT_PATH /
        "lightgbm_feature_importance.png"
    )

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()

    print(
        f"\n✅ Feature importance plot saved:"
        f"\n{output_file}"
    )

    return importance


# ============================================================
# STORE PERFORMANCE
# ============================================================

def store_analysis(df):

    print("\n" + "=" * 70)
    print("STORE-LEVEL PERFORMANCE")
    print("=" * 70)

    store_results = (
        df
        .groupby("store_nbr")
        .agg(
            actual_sales=(
                "sales",
                "sum"
            ),
            predicted_sales=(
                "prediction",
                "sum"
            ),
            MAE=(
                "absolute_error",
                "mean"
            )
        )
        .reset_index()
    )

    store_results[
        "sales_difference"
    ] = (
        store_results[
            "predicted_sales"
        ]
        -
        store_results[
            "actual_sales"
        ]
    )

    print("\nBest stores by MAE:")

    print(
        store_results
        .sort_values("MAE")
        .head(10)
        .to_string(
            index=False
        )
    )

    print("\nWorst stores by MAE:")

    print(
        store_results
        .sort_values(
            "MAE",
            ascending=False
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    output_file = (
        FORECAST_PATH /
        "store_performance.csv"
    )

    store_results.to_csv(
        output_file,
        index=False
    )

    print(
        f"\n✅ Store analysis saved:"
        f"\n{output_file}"
    )

    return store_results


# ============================================================
# PRODUCT FAMILY PERFORMANCE
# ============================================================

def family_analysis(df):

    print("\n" + "=" * 70)
    print("PRODUCT FAMILY PERFORMANCE")
    print("=" * 70)

    family_results = (
        df
        .groupby("family")
        .agg(
            actual_sales=(
                "sales",
                "sum"
            ),
            predicted_sales=(
                "prediction",
                "sum"
            ),
            MAE=(
                "absolute_error",
                "mean"
            )
        )
        .reset_index()
    )

    print("\nBest product families by MAE:")

    print(
        family_results
        .sort_values("MAE")
        .head(10)
        .to_string(
            index=False
        )
    )

    print("\nWorst product families by MAE:")

    print(
        family_results
        .sort_values(
            "MAE",
            ascending=False
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    output_file = (
        FORECAST_PATH /
        "family_performance.csv"
    )

    family_results.to_csv(
        output_file,
        index=False
    )

    print(
        f"\n✅ Family analysis saved:"
        f"\n{output_file}"
    )

    return family_results


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

def plot_actual_vs_predicted(df):

    print("\n" + "=" * 70)
    print("ACTUAL VS PREDICTED")
    print("=" * 70)

    daily = (
        df
        .groupby("date")
        .agg(
            actual_sales=(
                "sales",
                "sum"
            ),
            predicted_sales=(
                "prediction",
                "sum"
            )
        )
        .reset_index()
    )

    plt.figure(
        figsize=(14, 6)
    )

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

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Sales"
    )

    plt.title(
        "Actual vs Predicted Sales"
    )

    plt.legend()

    plt.tight_layout()

    output_file = (
        REPORT_PATH /
        "actual_vs_predicted.png"
    )

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()

    print(
        f"✅ Actual vs predicted plot saved:"
        f"\n{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df, model = load_data()

    df = calculate_errors(
        df
    )

    feature_importance(
        model
    )

    store_analysis(
        df
    )

    family_analysis(
        df
    )

    plot_actual_vs_predicted(
        df
    )

    print("\n" + "=" * 70)
    print("✅ MODEL ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    main()