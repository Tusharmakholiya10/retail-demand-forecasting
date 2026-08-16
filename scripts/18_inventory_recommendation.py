"""
18_inventory_recommendation.py

INVENTORY & REORDER RECOMMENDATION

Uses the 7-day future demand forecast to generate
simple inventory planning recommendations.

Important:
This is a planning estimate, not a real inventory system,
because actual stock levels and supplier lead times are
not available in the dataset.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path(
    "data/forecasts/future/future_predictions.csv"
)

OUTPUT_DIR = Path(
    "data/forecasts/inventory"
)

OUTPUT_PATH = (
    OUTPUT_DIR /
    "inventory_recommendations.csv"
)

# Business assumptions
LEAD_TIME_DAYS = 2

SAFETY_STOCK_PERCENT = 0.20


# ============================================================
# LOAD FORECAST
# ============================================================

def load_forecast():

    print("=" * 70)
    print("INVENTORY & REORDER RECOMMENDATION")
    print("=" * 70)

    print("\nLoading future forecast...")

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Forecast file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["date"]
    )

    print(
        f"Forecast shape: {df.shape}"
    )

    print(
        f"Forecast period: "
        f"{df['date'].min().date()} -> "
        f"{df['date'].max().date()}"
    )

    return df


# ============================================================
# CALCULATE INVENTORY METRICS
# ============================================================

def calculate_inventory_metrics(df):

    print("\nCalculating inventory metrics...")

    grouped = (
        df.groupby(
            [
                "store_nbr",
                "family",
                "city",
                "state",
                "type",
                "cluster"
            ],
            as_index=False
        )
        .agg(
            forecast_7_day=(
                "predicted_sales",
                "sum"
            )
        )
    )

    # --------------------------------------------------------
    # Average daily demand
    # --------------------------------------------------------

    grouped["average_daily_demand"] = (
        grouped["forecast_7_day"] / 7
    )

    # --------------------------------------------------------
    # Safety stock
    # --------------------------------------------------------

    grouped["safety_stock"] = (
        grouped["forecast_7_day"]
        * SAFETY_STOCK_PERCENT
    )

    # --------------------------------------------------------
    # Reorder point
    # --------------------------------------------------------

    grouped["reorder_point"] = (
        grouped["average_daily_demand"]
        * LEAD_TIME_DAYS
        +
        grouped["safety_stock"]
    )

    # --------------------------------------------------------
    # Recommended stock
    # --------------------------------------------------------

    grouped["recommended_stock"] = (
        grouped["forecast_7_day"]
        +
        grouped["safety_stock"]
    )

    # --------------------------------------------------------
    # Inventory status
    #
    # Since actual inventory is unavailable, this status
    # represents planning priority based on forecast demand.
    # --------------------------------------------------------

    def classify_priority(value):

        if value >= grouped[
            "recommended_stock"
        ].quantile(0.75):

            return "HIGH PRIORITY"

        elif value >= grouped[
            "recommended_stock"
        ].quantile(0.40):

            return "MEDIUM PRIORITY"

        else:

            return "LOW PRIORITY"

    grouped["inventory_priority"] = (
        grouped["recommended_stock"]
        .apply(classify_priority)
    )

    return grouped


# ============================================================
# ROUND VALUES
# ============================================================

def clean_output(df):

    numeric_columns = [
        "forecast_7_day",
        "average_daily_demand",
        "safety_stock",
        "reorder_point",
        "recommended_stock"
    ]

    for column in numeric_columns:

        df[column] = (
            df[column]
            .round(2)
        )

    return df


# ============================================================
# SAVE
# ============================================================

def save_output(df):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\n✅ Inventory recommendations saved:"
    )

    print(
        f"File: {OUTPUT_PATH}"
    )

    print(
        f"Rows: {len(df):,}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(df):

    print("\n" + "=" * 70)
    print("INVENTORY PLANNING SUMMARY")
    print("=" * 70)

    print(
        f"\nStore-family combinations: "
        f"{len(df):,}"
    )

    print(
        f"Total 7-day forecast: "
        f"{df['forecast_7_day'].sum():,.2f}"
    )

    print(
        f"Total safety stock: "
        f"{df['safety_stock'].sum():,.2f}"
    )

    print(
        f"Total recommended stock: "
        f"{df['recommended_stock'].sum():,.2f}"
    )

    print("\nPriority distribution:")

    print(
        df["inventory_priority"]
        .value_counts()
    )

    print("\nTop 10 inventory priorities:")

    top = (
        df.sort_values(
            "recommended_stock",
            ascending=False
        )
        .head(10)
    )

    print(
        top[
            [
                "store_nbr",
                "family",
                "forecast_7_day",
                "safety_stock",
                "reorder_point",
                "recommended_stock",
                "inventory_priority"
            ]
        ]
        .to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_forecast()

    inventory_df = (
        calculate_inventory_metrics(df)
    )

    inventory_df = clean_output(
        inventory_df
    )

    save_output(
        inventory_df
    )

    print_summary(
        inventory_df
    )

    print("\n" + "=" * 70)
    print("✅ INVENTORY RECOMMENDATION PIPELINE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()