import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTION_FILE = "data/forecasts/lightgbm_predictions.csv"
OUTPUT_DIR = "data/forecasts/error_analysis"
FIGURE_DIR = "reports/figures/error_analysis"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("=" * 70)
print("LIGHTGBM ERROR ANALYSIS")
print("=" * 70)

print("\nLoading predictions...")

df = pd.read_csv(PREDICTION_FILE)

print(f"Prediction rows: {len(df):,}")

# Convert date
df["date"] = pd.to_datetime(df["date"])


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "date",
    "store_nbr",
    "family",
    "sales",
    "prediction"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\nERROR: Missing required columns:")
    print(missing_columns)

    print("\nAvailable columns:")
    print(df.columns.tolist())

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# CREATE ERROR FEATURES
# ============================================================

print("\nCreating error metrics...")

df["error"] = df["prediction"] - df["sales"]

df["absolute_error"] = np.abs(df["error"])

df["squared_error"] = df["error"] ** 2

df["percentage_error"] = (
    df["absolute_error"] /
    df["sales"].replace(0, np.nan)
) * 100

df["direction"] = np.where(
    df["error"] > 0,
    "Overprediction",
    np.where(
        df["error"] < 0,
        "Underprediction",
        "Exact"
    )
)


# ============================================================
# OVERALL ERROR ANALYSIS
# ============================================================

mae = df["absolute_error"].mean()

rmse = np.sqrt(df["squared_error"].mean())

bias = df["error"].mean()

median_absolute_error = df["absolute_error"].median()

print("\n" + "=" * 70)
print("OVERALL ERROR METRICS")
print("=" * 70)

print(f"MAE                  : {mae:.4f}")
print(f"RMSE                 : {rmse:.4f}")
print(f"Mean Error / Bias    : {bias:.4f}")
print(f"Median Absolute Error: {median_absolute_error:.4f}")


# ============================================================
# OVERPREDICTION / UNDERPREDICTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION DIRECTION")
print("=" * 70)

direction_counts = df["direction"].value_counts()

print(direction_counts)

direction_percent = (
    df["direction"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nPercentage:")

print(direction_percent)

direction_table = pd.DataFrame({
    "count": direction_counts,
    "percentage": direction_percent
})

direction_table.to_csv(
    f"{OUTPUT_DIR}/prediction_direction.csv"
)


# ============================================================
# LARGEST INDIVIDUAL ERRORS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 LARGEST ABSOLUTE ERRORS")
print("=" * 70)

largest_errors = (
    df[
        [
            "date",
            "store_nbr",
            "family",
            "sales",
            "prediction",
            "error",
            "absolute_error"
        ]
    ]
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(20)
)

print(largest_errors.to_string(index=False))

largest_errors.to_csv(
    f"{OUTPUT_DIR}/largest_errors.csv",
    index=False
)


# ============================================================
# STORE ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("ERROR BY STORE")
print("=" * 70)

store_error = (
    df.groupby("store_nbr")
    .agg(
        actual_sales=("sales", "sum"),
        predicted_sales=("prediction", "sum"),
        MAE=("absolute_error", "mean"),
        RMSE=("squared_error",
              lambda x: np.sqrt(x.mean())),
        bias=("error", "mean"),
        observations=("sales", "count")
    )
    .reset_index()
)

store_error["sales_difference"] = (
    store_error["predicted_sales"]
    - store_error["actual_sales"]
)

store_error = store_error.sort_values("MAE")

print("\nBest 10 stores:")

print(
    store_error
    .head(10)
    .to_string(index=False)
)

print("\nWorst 10 stores:")

print(
    store_error
    .tail(10)
    .sort_values("MAE", ascending=False)
    .to_string(index=False)
)

store_error.to_csv(
    f"{OUTPUT_DIR}/store_error_analysis.csv",
    index=False
)


# ============================================================
# FAMILY ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("ERROR BY PRODUCT FAMILY")
print("=" * 70)

family_error = (
    df.groupby("family")
    .agg(
        actual_sales=("sales", "sum"),
        predicted_sales=("prediction", "sum"),
        MAE=("absolute_error", "mean"),
        RMSE=("squared_error",
              lambda x: np.sqrt(x.mean())),
        bias=("error", "mean"),
        observations=("sales", "count")
    )
    .reset_index()
)

family_error["sales_difference"] = (
    family_error["predicted_sales"]
    - family_error["actual_sales"]
)

family_error = family_error.sort_values("MAE")

print("\nBest 10 product families:")

print(
    family_error
    .head(10)
    .to_string(index=False)
)

print("\nWorst 10 product families:")

print(
    family_error
    .tail(10)
    .sort_values("MAE", ascending=False)
    .to_string(index=False)
)

family_error.to_csv(
    f"{OUTPUT_DIR}/family_error_analysis.csv",
    index=False
)


# ============================================================
# MONTHLY ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("ERROR BY MONTH")
print("=" * 70)

df["year"] = df["date"].dt.year

df["month"] = df["date"].dt.month

monthly_error = (
    df.groupby(["year", "month"])
    .agg(
        actual_sales=("sales", "sum"),
        predicted_sales=("prediction", "sum"),
        MAE=("absolute_error", "mean"),
        RMSE=("squared_error",
              lambda x: np.sqrt(x.mean())),
        bias=("error", "mean")
    )
    .reset_index()
)

monthly_error.to_csv(
    f"{OUTPUT_DIR}/monthly_error_analysis.csv",
    index=False
)

print(monthly_error.tail(12).to_string(index=False))


# ============================================================
# PROMOTION ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PROMOTION ERROR ANALYSIS")
print("=" * 70)

if "onpromotion" in df.columns:

    df["promotion_status"] = np.where(
        df["onpromotion"] > 0,
        "Promotion",
        "No Promotion"
    )

    promotion_error = (
        df.groupby("promotion_status")
        .agg(
            actual_sales=("sales", "sum"),
            predicted_sales=("prediction", "sum"),
            MAE=("absolute_error", "mean"),
            RMSE=("squared_error",
                  lambda x: np.sqrt(x.mean())),
            bias=("error", "mean"),
            observations=("sales", "count")
        )
        .reset_index()
    )

    print(promotion_error.to_string(index=False))

    promotion_error.to_csv(
        f"{OUTPUT_DIR}/promotion_error_analysis.csv",
        index=False
    )


# ============================================================
# HOLIDAY ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("HOLIDAY ERROR ANALYSIS")
print("=" * 70)

holiday_columns = [
    "has_national_holiday",
    "is_regional_holiday",
    "is_local_holiday"
]

available_holiday_columns = [
    col for col in holiday_columns
    if col in df.columns
]

if available_holiday_columns:

    df["any_holiday"] = (
        df[available_holiday_columns]
        .sum(axis=1) > 0
    )

    df["holiday_status"] = np.where(
        df["any_holiday"],
        "Holiday",
        "Normal Day"
    )

    holiday_error = (
        df.groupby("holiday_status")
        .agg(
            actual_sales=("sales", "sum"),
            predicted_sales=("prediction", "sum"),
            MAE=("absolute_error", "mean"),
            RMSE=("squared_error",
                  lambda x: np.sqrt(x.mean())),
            bias=("error", "mean"),
            observations=("sales", "count")
        )
        .reset_index()
    )

    print(holiday_error.to_string(index=False))

    holiday_error.to_csv(
        f"{OUTPUT_DIR}/holiday_error_analysis.csv",
        index=False
    )


# ============================================================
# ACTUAL VS PREDICTED SCATTER PLOT
# ============================================================

print("\nCreating actual vs predicted plot...")

plt.figure(figsize=(10, 7))

plt.scatter(
    df["sales"],
    df["prediction"],
    alpha=0.15,
    s=8
)

max_value = max(
    df["sales"].max(),
    df["prediction"].max()
)

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--"
)

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

plt.title(
    "LightGBM: Actual vs Predicted Sales"
)

plt.tight_layout()

plt.savefig(
    f"{FIGURE_DIR}/actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ============================================================
# ERROR DISTRIBUTION
# ============================================================

print("Creating error distribution plot...")

plt.figure(figsize=(10, 7))

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

plt.title(
    "Distribution of Prediction Errors"
)

plt.tight_layout()

plt.savefig(
    f"{FIGURE_DIR}/error_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# MAE BY MONTH
# ============================================================

print("Creating monthly MAE plot...")

monthly_plot = (
    monthly_error
    .copy()
)

monthly_plot["period"] = pd.to_datetime(
    monthly_plot["year"].astype(str)
    + "-"
    + monthly_plot["month"].astype(str)
    + "-01"
)

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_plot["period"],
    monthly_plot["MAE"]
)

plt.xlabel("Month")

plt.ylabel("MAE")

plt.title(
    "Monthly Forecasting Error"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    f"{FIGURE_DIR}/monthly_mae.png",
    dpi=300
)

plt.close()


# ============================================================
# STORE MAE
# ============================================================

print("Creating store MAE plot...")

store_plot = (
    store_error
    .sort_values("MAE", ascending=False)
)

plt.figure(figsize=(12, 7))

plt.bar(
    store_plot["store_nbr"].astype(str),
    store_plot["MAE"]
)

plt.xlabel("Store Number")

plt.ylabel("MAE")

plt.title(
    "Forecasting Error by Store"
)

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    f"{FIGURE_DIR}/store_mae.png",
    dpi=300
)

plt.close()


# ============================================================
# FAMILY MAE
# ============================================================

print("Creating product family MAE plot...")

family_plot = (
    family_error
    .sort_values("MAE", ascending=False)
)

plt.figure(figsize=(14, 8))

plt.barh(
    family_plot["family"],
    family_plot["MAE"]
)

plt.xlabel("MAE")

plt.ylabel("Product Family")

plt.title(
    "Forecasting Error by Product Family"
)

plt.tight_layout()

plt.savefig(
    f"{FIGURE_DIR}/family_mae.png",
    dpi=300
)

plt.close()


# ============================================================
# SAVE COMPLETE ERROR DATASET
# ============================================================

print("\nSaving detailed error dataset...")

df.to_csv(
    f"{OUTPUT_DIR}/detailed_error_analysis.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)

print(f"""
Overall MAE     : {mae:.4f}
Overall RMSE    : {rmse:.4f}
Model Bias      : {bias:.4f}

Output directory:
{OUTPUT_DIR}

Figures:
{FIGURE_DIR}

Generated files:

1. prediction_direction.csv
2. largest_errors.csv
3. store_error_analysis.csv
4. family_error_analysis.csv
5. monthly_error_analysis.csv
6. promotion_error_analysis.csv
7. holiday_error_analysis.csv
8. detailed_error_analysis.csv

Generated figures:

1. actual_vs_predicted.png
2. error_distribution.png
3. monthly_mae.png
4. store_mae.png
5. family_mae.png

======================================================================
""")