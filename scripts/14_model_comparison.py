from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

FORECAST_DIR = Path("data/forecasts")
FIGURE_DIR = Path("reports/figures")

OUTPUT_FILE = FORECAST_DIR / "model_comparison.csv"
PLOT_FILE = FIGURE_DIR / "model_comparison.png"


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():

    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    result_files = {
        "Baseline": FORECAST_DIR / "baseline_results.csv",
        "LightGBM": FORECAST_DIR / "lightgbm_results.csv",
        "XGBoost": FORECAST_DIR / "xgboost_results.csv",
    }

    results = []

    for model_name, file_path in result_files.items():

        print(f"\nLoading {model_name} results...")

        if not file_path.exists():

            print(f"⚠️ File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)

        print(f"Rows loaded: {len(df)}")

        results.append(df)

    if not results:
        raise FileNotFoundError(
            "No model result files were found."
        )

    combined = pd.concat(
        results,
        ignore_index=True
    )

    return combined


# ============================================================
# CLEAN RESULTS
# ============================================================

def clean_results(df):

    # Make sure numerical columns are numeric
    metric_columns = [
        "rows_evaluated",
        "MAE",
        "RMSE",
        "RMSLE"
    ]

    for column in metric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Remove accidental duplicate model rows
    df = df.drop_duplicates(
        subset=["model"],
        keep="last"
    )

    return df


# ============================================================
# BASELINE IMPROVEMENT
# ============================================================

def calculate_improvement(df):

    baseline_models = [
        "Naive_Lag_1",
        "Weekly_Seasonal_Lag_7",
        "Seasonal_Lag_28"
    ]

    baseline_df = df[
        df["model"].isin(baseline_models)
    ]

    if len(baseline_df) == 0:
        return df

    best_baseline_mae = baseline_df["MAE"].min()
    best_baseline_rmse = baseline_df["RMSE"].min()
    best_baseline_rmsle = baseline_df["RMSLE"].min()

    df["MAE_improvement_vs_best_baseline_%"] = (
        (best_baseline_mae - df["MAE"])
        / best_baseline_mae
        * 100
    )

    df["RMSE_improvement_vs_best_baseline_%"] = (
        (best_baseline_rmse - df["RMSE"])
        / best_baseline_rmse
        * 100
    )

    df["RMSLE_improvement_vs_best_baseline_%"] = (
        (best_baseline_rmsle - df["RMSLE"])
        / best_baseline_rmsle
        * 100
    )

    return df


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(df):

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    display_columns = [
        "model",
        "rows_evaluated",
        "MAE",
        "RMSE",
        "RMSLE"
    ]

    print(
        df[display_columns]
        .sort_values("MAE")
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Best models
    # --------------------------------------------------------

    best_mae = df.loc[
        df["MAE"].idxmin()
    ]

    best_rmse = df.loc[
        df["RMSE"].idxmin()
    ]

    best_rmsle = df.loc[
        df["RMSLE"].idxmin()
    ]

    print("\n" + "=" * 70)
    print("BEST MODELS")
    print("=" * 70)

    print(
        f"\nBest MAE   : "
        f"{best_mae['model']} "
        f"({best_mae['MAE']:.4f})"
    )

    print(
        f"Best RMSE  : "
        f"{best_rmse['model']} "
        f"({best_rmse['RMSE']:.4f})"
    )

    print(
        f"Best RMSLE : "
        f"{best_rmsle['model']} "
        f"({best_rmsle['RMSLE']:.4f})"
    )


# ============================================================
# CREATE VISUALIZATION
# ============================================================

def create_plot(df):

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_df = df.sort_values(
        "MAE"
    ).copy()

    plt.figure(
        figsize=(11, 6)
    )

    bars = plt.bar(
        plot_df["model"],
        plot_df["MAE"]
    )

    plt.title(
        "Model Comparison - MAE"
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Mean Absolute Error (MAE)"
    )

    plt.xticks(
        rotation=25,
        ha="right"
    )

    # Add values above bars
    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    plt.savefig(
        PLOT_FILE,
        dpi=150
    )

    plt.close()

    print(
        f"\n✅ Comparison plot saved:"
        f"\n{PLOT_FILE}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(df):

    FORECAST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = df.sort_values(
        "MAE"
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\n✅ Comparison results saved:"
        f"\n{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_results()

    df = clean_results(
        df
    )

    df = calculate_improvement(
        df
    )

    display_results(
        df
    )

    create_plot(
        df
    )

    save_results(
        df
    )

    print("\n" + "=" * 70)
    print("✅ MODEL COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()