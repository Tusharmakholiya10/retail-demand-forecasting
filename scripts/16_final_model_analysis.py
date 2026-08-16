import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

FINAL_DIR = "data/forecasts/final_evaluation"
FIGURE_DIR = "reports/figures/final_evaluation"
OUTPUT_DIR = "data/forecasts/final_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_plot(filename):
    path = os.path.join(FIGURE_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("FINAL MODEL ANALYSIS")
    print("=" * 70)

    print("\nLoading final predictions...")

    predictions_path = os.path.join(
        FINAL_DIR,
        "final_predictions.csv"
    )

    df = pd.read_csv(predictions_path)

    print(f"Prediction rows: {len(df):,}")
    print("\nColumns:")
    print(df.columns.tolist())

    return df


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

def identify_columns(df):

    actual_candidates = [
        "sales",
        "actual_sales",
        "actual"
    ]

    predicted_candidates = [
        "predicted_sales",
        "prediction",
        "predicted",
        "pred"
    ]

    actual_col = None
    predicted_col = None

    for col in actual_candidates:
        if col in df.columns:
            actual_col = col
            break

    for col in predicted_candidates:
        if col in df.columns:
            predicted_col = col
            break

    if actual_col is None:
        raise ValueError(
            "Could not find actual sales column."
        )

    if predicted_col is None:
        raise ValueError(
            "Could not find predicted sales column."
        )

    print(f"\nActual column    : {actual_col}")
    print(f"Prediction column: {predicted_col}")

    return actual_col, predicted_col


# ============================================================
# CREATE ERROR FEATURES
# ============================================================

def create_error_features(
    df,
    actual_col,
    predicted_col
):

    print("\n" + "=" * 70)
    print("CREATING ERROR FEATURES")
    print("=" * 70)

    df = df.copy()

    df["error"] = (
        df[predicted_col] - df[actual_col]
    )

    df["absolute_error"] = (
        df["error"].abs()
    )

    df["squared_error"] = (
        df["error"] ** 2
    )

    # Positive error = overprediction
    # Negative error = underprediction

    df["error_type"] = np.where(
        df["error"] > 0,
        "Overprediction",
        np.where(
            df["error"] < 0,
            "Underprediction",
            "Exact"
        )
    )

    # Safe percentage error
    df["percentage_error"] = np.where(
        df[actual_col] != 0,
        (
            df["absolute_error"]
            / df[actual_col]
        ) * 100,
        np.nan
    )

    if "date" in df.columns:

        df["date"] = pd.to_datetime(df["date"])

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        df["year_month"] = (
            df["date"]
            .dt.to_period("M")
            .astype(str)
        )

    return df


# ============================================================
# OVERALL ERROR ANALYSIS
# ============================================================

def overall_error_analysis(
    df,
    actual_col,
    predicted_col
):

    print("\n" + "=" * 70)
    print("OVERALL ERROR ANALYSIS")
    print("=" * 70)

    mae = df["absolute_error"].mean()

    rmse = np.sqrt(
        df["squared_error"].mean()
    )

    total_actual = df[actual_col].sum()

    wape = (
        df["absolute_error"].sum()
        / total_actual
        * 100
        if total_actual != 0
        else np.nan
    )

    overprediction = df[
        df["error"] > 0
    ]

    underprediction = df[
        df["error"] < 0
    ]

    summary = pd.DataFrame({
        "metric": [
            "MAE",
            "RMSE",
            "WAPE (%)",
            "Mean Error",
            "Median Absolute Error",
            "Maximum Absolute Error",
            "Overprediction Rows",
            "Underprediction Rows"
        ],
        "value": [
            mae,
            rmse,
            wape,
            df["error"].mean(),
            df["absolute_error"].median(),
            df["absolute_error"].max(),
            len(overprediction),
            len(underprediction)
        ]
    })

    print(summary.to_string(index=False))

    output_path = os.path.join(
        OUTPUT_DIR,
        "overall_error_analysis.csv"
    )

    summary.to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved: {output_path}")

    return summary


# ============================================================
# MONTHLY ANALYSIS
# ============================================================

def monthly_analysis(
    df,
    actual_col,
    predicted_col
):

    print("\n" + "=" * 70)
    print("MONTHLY ERROR ANALYSIS")
    print("=" * 70)

    if "year_month" not in df.columns:
        print("Date information unavailable.")
        return None

    monthly = (
        df.groupby("year_month")
        .agg(
            actual_sales=(
                actual_col,
                "sum"
            ),
            predicted_sales=(
                predicted_col,
                "sum"
            ),
            MAE=(
                "absolute_error",
                "mean"
            ),
            RMSE=(
                "squared_error",
                lambda x: np.sqrt(x.mean())
            ),
            observations=(
                actual_col,
                "count"
            )
        )
        .reset_index()
    )

    monthly["error"] = (
        monthly["predicted_sales"]
        - monthly["actual_sales"]
    )

    monthly["WAPE_percent"] = (
        monthly["error"]
        .abs()
        / monthly["actual_sales"]
        * 100
    )

    print("\nWorst months by MAE:")

    print(
        monthly
        .sort_values("MAE", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "monthly_analysis.csv"
    )

    monthly.to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved: {output_path}")

    # --------------------------------------------------------
    # Monthly MAE
    # --------------------------------------------------------

    plt.figure(figsize=(14, 6))

    plt.plot(
        monthly["year_month"],
        monthly["MAE"]
    )

    plt.xticks(
        rotation=90,
        fontsize=7
    )

    plt.xlabel("Month")
    plt.ylabel("MAE")
    plt.title("Monthly Forecast MAE")

    save_plot("monthly_mae.png")

    # --------------------------------------------------------
    # Monthly actual vs predicted
    # --------------------------------------------------------

    plt.figure(figsize=(14, 6))

    plt.plot(
        monthly["year_month"],
        monthly["actual_sales"],
        label="Actual"
    )

    plt.plot(
        monthly["year_month"],
        monthly["predicted_sales"],
        label="Predicted"
    )

    plt.xticks(
        rotation=90,
        fontsize=7
    )

    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.title("Monthly Actual vs Predicted Sales")

    plt.legend()

    save_plot("monthly_actual_vs_predicted.png")

    return monthly


# ============================================================
# STORE ANALYSIS
# ============================================================

def store_analysis(
    df,
    actual_col,
    predicted_col
):

    print("\n" + "=" * 70)
    print("STORE ERROR ANALYSIS")
    print("=" * 70)

    if "store_nbr" not in df.columns:
        print("store_nbr column unavailable.")
        return None

    store = (
        df.groupby("store_nbr")
        .agg(
            actual_sales=(
                actual_col,
                "sum"
            ),
            predicted_sales=(
                predicted_col,
                "sum"
            ),
            MAE=(
                "absolute_error",
                "mean"
            ),
            RMSE=(
                "squared_error",
                lambda x: np.sqrt(x.mean())
            ),
            observations=(
                actual_col,
                "count"
            )
        )
        .reset_index()
    )

    store["sales_difference"] = (
        store["predicted_sales"]
        - store["actual_sales"]
    )

    store["WAPE_percent"] = (
        store["sales_difference"]
        .abs()
        / store["actual_sales"]
        * 100
    )

    print("\nBest stores by MAE:")

    print(
        store
        .sort_values("MAE")
        .head(10)
        .to_string(index=False)
    )

    print("\nWorst stores by MAE:")

    print(
        store
        .sort_values(
            "MAE",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "store_analysis.csv"
    )

    store.to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved: {output_path}")

    # --------------------------------------------------------
    # Store MAE
    # --------------------------------------------------------

    worst = (
        store
        .sort_values("MAE", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(12, 6))

    plt.bar(
        worst["store_nbr"].astype(str),
        worst["MAE"]
    )

    plt.xlabel("Store")
    plt.ylabel("MAE")
    plt.title("15 Stores With Highest Forecast Error")

    save_plot("worst_store_mae.png")

    return store


# ============================================================
# PRODUCT FAMILY ANALYSIS
# ============================================================

def family_analysis(
    df,
    actual_col,
    predicted_col
):

    print("\n" + "=" * 70)
    print("PRODUCT FAMILY ERROR ANALYSIS")
    print("=" * 70)

    if "family" not in df.columns:
        print("family column unavailable.")
        return None

    family = (
        df.groupby("family")
        .agg(
            actual_sales=(
                actual_col,
                "sum"
            ),
            predicted_sales=(
                predicted_col,
                "sum"
            ),
            MAE=(
                "absolute_error",
                "mean"
            ),
            RMSE=(
                "squared_error",
                lambda x: np.sqrt(x.mean())
            ),
            observations=(
                actual_col,
                "count"
            )
        )
        .reset_index()
    )

    family["sales_difference"] = (
        family["predicted_sales"]
        - family["actual_sales"]
    )

    family["WAPE_percent"] = (
        family["sales_difference"]
        .abs()
        / family["actual_sales"]
        * 100
    )

    print("\nBest product families:")

    print(
        family
        .sort_values("MAE")
        .head(10)
        .to_string(index=False)
    )

    print("\nWorst product families:")

    print(
        family
        .sort_values(
            "MAE",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "family_analysis.csv"
    )

    family.to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved: {output_path}")

    # --------------------------------------------------------
    # Worst product families
    # --------------------------------------------------------

    worst = (
        family
        .sort_values(
            "MAE",
            ascending=False
        )
        .head(15)
    )

    plt.figure(figsize=(12, 7))

    plt.barh(
        worst["family"],
        worst["MAE"]
    )

    plt.xlabel("MAE")
    plt.ylabel("Product Family")
    plt.title(
        "15 Product Families With Highest Forecast Error"
    )

    save_plot("worst_family_mae.png")

    return family


# ============================================================
# TOP ERRORS
# ============================================================

def top_errors(df):

    print("\n" + "=" * 70)
    print("TOP PREDICTION ERRORS")
    print("=" * 70)

    columns = [
        col for col in [
            "date",
            "store_nbr",
            "family",
            "sales",
            "actual_sales",
            "predicted_sales",
            "error",
            "absolute_error",
            "percentage_error",
            "error_type"
        ]
        if col in df.columns
    ]

    top = (
        df
        .sort_values(
            "absolute_error",
            ascending=False
        )
        .head(100)
    )

    print("\nTop 20 largest errors:")

    print(
        top[columns]
        .head(20)
        .to_string(index=False)
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "top_100_errors.csv"
    )

    top.to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved: {output_path}")

    # --------------------------------------------------------
    # Error distribution
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["error"],
        bins=100
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Prediction Error Distribution")

    save_plot("error_distribution.png")

    return top


# ============================================================
# OVER / UNDER PREDICTION
# ============================================================

def over_under_analysis(df):

    print("\n" + "=" * 70)
    print("OVERPREDICTION VS UNDERPREDICTION")
    print("=" * 70)

    counts = (
        df["error_type"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "error_type",
        "count"
    ]

    print(counts.to_string(index=False))

    output_path = os.path.join(
        OUTPUT_DIR,
        "over_under_prediction.csv"
    )

    counts.to_csv(
        output_path,
        index=False
    )

    plt.figure(figsize=(8, 6))

    plt.bar(
        counts["error_type"],
        counts["count"]
    )

    plt.xlabel("Prediction Type")
    plt.ylabel("Number of Predictions")
    plt.title(
        "Overprediction vs Underprediction"
    )

    save_plot("over_under_prediction.png")

    print(f"\nSaved: {output_path}")


# ============================================================
# FINAL SUMMARY
# ============================================================

def create_final_summary(
    df,
    actual_col,
    predicted_col,
    monthly,
    store,
    family
):

    print("\n" + "=" * 70)
    print("FINAL BUSINESS SUMMARY")
    print("=" * 70)

    summary = []

    # Overall
    summary.append({
        "category": "Overall",
        "metric": "MAE",
        "value": df["absolute_error"].mean()
    })

    summary.append({
        "category": "Overall",
        "metric": "RMSE",
        "value": np.sqrt(
            df["squared_error"].mean()
        )
    })

    # Best/worst store
    if store is not None:

        best_store = (
            store
            .sort_values("MAE")
            .iloc[0]
        )

        worst_store = (
            store
            .sort_values("MAE", ascending=False)
            .iloc[0]
        )

        summary.append({
            "category": "Store",
            "metric": "Best Store",
            "value": best_store["store_nbr"]
        })

        summary.append({
            "category": "Store",
            "metric": "Best Store MAE",
            "value": best_store["MAE"]
        })

        summary.append({
            "category": "Store",
            "metric": "Worst Store",
            "value": worst_store["store_nbr"]
        })

        summary.append({
            "category": "Store",
            "metric": "Worst Store MAE",
            "value": worst_store["MAE"]
        })

    # Best/worst family
    if family is not None:

        best_family = (
            family
            .sort_values("MAE")
            .iloc[0]
        )

        worst_family = (
            family
            .sort_values("MAE", ascending=False)
            .iloc[0]
        )

        summary.append({
            "category": "Product Family",
            "metric": "Best Family",
            "value": best_family["family"]
        })

        summary.append({
            "category": "Product Family",
            "metric": "Best Family MAE",
            "value": best_family["MAE"]
        })

        summary.append({
            "category": "Product Family",
            "metric": "Worst Family",
            "value": worst_family["family"]
        })

        summary.append({
            "category": "Product Family",
            "metric": "Worst Family MAE",
            "value": worst_family["MAE"]
        })

    summary_df = pd.DataFrame(summary)

    output_path = os.path.join(
        OUTPUT_DIR,
        "final_business_summary.csv"
    )

    summary_df.to_csv(
        output_path,
        index=False
    )

    print(
        summary_df.to_string(index=False)
    )

    print(f"\nSaved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    actual_col, predicted_col = (
        identify_columns(df)
    )

    df = create_error_features(
        df,
        actual_col,
        predicted_col
    )

    overall_error_analysis(
        df,
        actual_col,
        predicted_col
    )

    monthly = monthly_analysis(
        df,
        actual_col,
        predicted_col
    )

    store = store_analysis(
        df,
        actual_col,
        predicted_col
    )

    family = family_analysis(
        df,
        actual_col,
        predicted_col
    )

    top_errors(df)

    over_under_analysis(df)

    create_final_summary(
        df,
        actual_col,
        predicted_col,
        monthly,
        store,
        family
    )

    print("\n" + "=" * 70)
    print("✅ FINAL MODEL ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nAnalysis files:")
    print(f"  {OUTPUT_DIR}")

    print("\nFigures:")
    print(f"  {FIGURE_DIR}")


if __name__ == "__main__":
    main()