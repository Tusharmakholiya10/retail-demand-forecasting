"""
17_future_predict.py

TRUE FUTURE DEMAND FORECASTING PIPELINE

Uses the trained tuned LightGBM model to recursively
forecast future demand after the last available date.

Important:
- No actual future sales are used.
- Future lag/rolling features are generated recursively.
- Future promotions default to 0 unless supplied.
- Future oil price uses the latest known oil price.
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import lightgbm as lgb


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    "models/lightgbm/tuned_lightgbm_model.txt"
)

HISTORY_PATH = Path(
    "data/processed/modeling_data.csv"
)

NATIONAL_HOLIDAYS_PATH = Path(
    "data/processed/national_holidays.csv"
)

REGIONAL_HOLIDAYS_PATH = Path(
    "data/processed/regional_holidays.csv"
)

LOCAL_HOLIDAYS_PATH = Path(
    "data/processed/local_holidays.csv"
)

OIL_PATH = Path(
    "data/processed/daily_oil.csv"
)

OUTPUT_DIR = Path(
    "data/forecasts/future"
)

OUTPUT_PATH = (
    OUTPUT_DIR / "future_predictions.csv"
)


# Number of days to forecast
FORECAST_DAYS = 7


# ============================================================
# MODEL FEATURES
# ============================================================

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


CATEGORICAL_COLUMNS = [
    "family",
    "city",
    "state",
    "type",
]


# ============================================================
# LOAD HISTORY
# ============================================================

def load_history():

    print("=" * 70)
    print("TRUE FUTURE DEMAND FORECASTING")
    print("=" * 70)

    print("\nLoading modeling data...")

    df = pd.read_csv(
        HISTORY_PATH,
        parse_dates=["date"]
    )

    print(
        f"Historical data shape: {df.shape}"
    )

    print(
        f"Historical date range: "
        f"{df['date'].min().date()} -> "
        f"{df['date'].max().date()}"
    )

    return df


# ============================================================
# LOAD HOLIDAYS
# ============================================================

def load_holidays():

    print("\nLoading holiday data...")

    national = pd.read_csv(
        NATIONAL_HOLIDAYS_PATH,
        parse_dates=["date"]
    )

    regional = pd.read_csv(
        REGIONAL_HOLIDAYS_PATH,
        parse_dates=["date"]
    )

    local = pd.read_csv(
        LOCAL_HOLIDAYS_PATH,
        parse_dates=["date"]
    )

    print(
        f"National holidays: {len(national):,}"
    )

    print(
        f"Regional holidays: {len(regional):,}"
    )

    print(
        f"Local holidays: {len(local):,}"
    )

    return national, regional, local


# ============================================================
# LOAD OIL
# ============================================================

def load_oil():

    print("\nLoading oil price data...")

    oil = pd.read_csv(
        OIL_PATH,
        parse_dates=["date"]
    )

    oil = oil.sort_values("date")

    last_oil_price = oil[
        "oil_price"
    ].dropna().iloc[-1]

    print(
        f"Latest known oil price: "
        f"{last_oil_price:.4f}"
    )

    return oil, last_oil_price


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading tuned LightGBM model...")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = lgb.Booster(
        model_file=str(MODEL_PATH)
    )

    print(
        "Model loaded successfully."
    )

    print(
        f"Number of model features: "
        f"{model.num_feature()}"
    )

    return model


# ============================================================
# CREATE FUTURE ROWS
# ============================================================

def create_future_rows(
    history,
    forecast_dates,
    national,
    regional,
    local,
    oil_price
):

    print("\nCreating future rows...")

    # --------------------------------------------------------
    # Get unique store/family combinations
    # --------------------------------------------------------

    combinations = history[
        [
            "store_nbr",
            "family",
            "city",
            "state",
            "type",
            "cluster",
        ]
    ].drop_duplicates()

    print(
        f"Store-family combinations: "
        f"{len(combinations):,}"
    )

    future = combinations.merge(
        pd.DataFrame(
            {"date": forecast_dates}
        ),
        how="cross"
    )

    # --------------------------------------------------------
    # Promotion
    # --------------------------------------------------------

    # We don't know future promotion schedules.
    # Therefore default to no promotion.

    future["onpromotion"] = 0

    future["log_onpromotion"] = np.log1p(
        future["onpromotion"]
    )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    future["year"] = (
        future["date"].dt.year
    )

    future["month"] = (
        future["date"].dt.month
    )

    future["quarter"] = (
        future["date"].dt.quarter
    )

    future["week_of_year"] = (
        future["date"]
        .dt.isocalendar()
        .week
        .astype("int16")
    )

    future["day_of_week"] = (
        future["date"].dt.dayofweek
    )

    future["day_of_month"] = (
        future["date"].dt.day
    )

    future["is_weekend"] = (
        future["day_of_week"] >= 5
    ).astype("int8")

    # --------------------------------------------------------
    # Oil price
    # --------------------------------------------------------

    future["oil_price"] = oil_price

    # --------------------------------------------------------
    # National holidays
    # --------------------------------------------------------

    national_temp = national[
        ["date", "has_national_holiday"]
    ].drop_duplicates(
        subset=["date"]
    )

    future = future.merge(
        national_temp,
        on="date",
        how="left"
    )

    future["has_national_holiday"] = (
        future[
            "has_national_holiday"
        ]
        .fillna(0)
        .astype("int8")
    )

    # --------------------------------------------------------
    # Regional holidays
    # --------------------------------------------------------

    regional_temp = regional[
        [
            "date",
            "state",
            "is_regional_holiday"
        ]
    ].drop_duplicates(
        subset=["date", "state"]
    )

    future = future.merge(
        regional_temp,
        on=["date", "state"],
        how="left"
    )

    future["is_regional_holiday"] = (
        future[
            "is_regional_holiday"
        ]
        .fillna(0)
        .astype("int8")
    )

    # --------------------------------------------------------
    # Local holidays
    # --------------------------------------------------------

    local_temp = local[
        [
            "date",
            "city",
            "is_local_holiday"
        ]
    ].drop_duplicates(
        subset=["date", "city"]
    )

    future = future.merge(
        local_temp,
        on=["date", "city"],
        how="left"
    )

    future["is_local_holiday"] = (
        future[
            "is_local_holiday"
        ]
        .fillna(0)
        .astype("int8")
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    future = future.sort_values(
        [
            "store_nbr",
            "family",
            "date"
        ]
    ).reset_index(drop=True)

    print(
        f"Future rows created: "
        f"{len(future):,}"
    )

    return future


# ============================================================
# CREATE RECURSIVE FEATURES
# ============================================================

def calculate_features(
    future_row,
    history_group
):

    sales_history = list(
        history_group["sales"].values
    )

    # --------------------------------------------------------
    # Required lag values
    # --------------------------------------------------------

    if len(sales_history) < 28:

        raise ValueError(
            "Not enough historical observations "
            "to calculate lag_28."
        )

    future_row["lag_1"] = (
        sales_history[-1]
    )

    future_row["lag_7"] = (
        sales_history[-7]
    )

    future_row["lag_14"] = (
        sales_history[-14]
    )

    future_row["lag_28"] = (
        sales_history[-28]
    )

    # --------------------------------------------------------
    # Rolling features
    #
    # IMPORTANT:
    # These use only observations BEFORE
    # the current prediction.
    # --------------------------------------------------------

    last_7 = np.array(
        sales_history[-7:]
    )

    last_14 = np.array(
        sales_history[-14:]
    )

    last_28 = np.array(
        sales_history[-28:]
    )

    future_row["rolling_mean_7"] = (
        last_7.mean()
    )

    future_row["rolling_mean_14"] = (
        last_14.mean()
    )

    future_row["rolling_mean_28"] = (
        last_28.mean()
    )

    future_row["rolling_std_7"] = (
        last_7.std(ddof=1)
        if len(last_7) > 1
        else 0
    )

    future_row["rolling_std_28"] = (
        last_28.std(ddof=1)
        if len(last_28) > 1
        else 0
    )

    return future_row


# ============================================================
# FORECAST
# ============================================================

def forecast(
    history,
    future,
    model
):

    print("\n" + "=" * 70)
    print("GENERATING FUTURE FORECAST")
    print("=" * 70)

    # --------------------------------------------------------
    # History dictionary
    # --------------------------------------------------------

    history_groups = {}

    for (
        store,
        family
    ), group in history.groupby(
        ["store_nbr", "family"]
    ):

        group = group.sort_values(
            "date"
        )

        history_groups[
            (store, family)
        ] = group.copy()

    predictions = []

    # --------------------------------------------------------
    # Predict one date at a time
    # --------------------------------------------------------

    forecast_dates = sorted(
        future["date"].unique()
    )

    for current_date in forecast_dates:

        print(
            f"\nForecasting "
            f"{pd.Timestamp(current_date).date()}..."
        )

        current_rows = future[
            future["date"] == current_date
        ].copy()

        for index, row in current_rows.iterrows():

            key = (
                row["store_nbr"],
                row["family"]
            )

            group = history_groups[key]

            row_dict = row.to_dict()

            # ----------------------------------------------
            # Create lag and rolling features
            # ----------------------------------------------

            row_dict = calculate_features(
                row_dict,
                group
            )

            # ----------------------------------------------
            # Prepare model input
            # ----------------------------------------------

            X = pd.DataFrame(
                [row_dict]
            )

            X = X[FEATURES].copy()

            # categorical columns
            for col in CATEGORICAL_COLUMNS:

                X[col] = X[col].astype(
                    "category"
                )

            # numeric columns
            numeric_columns = [
                col
                for col in FEATURES
                if col not in CATEGORICAL_COLUMNS
            ]

            for col in numeric_columns:

                X[col] = pd.to_numeric(
                    X[col],
                    errors="coerce"
                )

            # ----------------------------------------------
            # Predict
            # ----------------------------------------------

            prediction = model.predict(X)[0]

            # Demand cannot be negative
            prediction = max(
                0,
                float(prediction)
            )

            # ----------------------------------------------
            # Save prediction
            # ----------------------------------------------

            row_dict[
                "predicted_sales"
            ] = prediction

            predictions.append(
                row_dict
            )

            # ----------------------------------------------
            # IMPORTANT:
            # Add prediction to history.
            #
            # This allows the next future day to use
            # today's predicted sales as lag_1.
            # ----------------------------------------------

            new_history_row = {
                "date": row_dict["date"],
                "sales": prediction
            }

            history_groups[key] = pd.concat(
                [
                    group,
                    pd.DataFrame(
                        [new_history_row]
                    )
                ],
                ignore_index=True
            )

    result = pd.DataFrame(
        predictions
    )

    return result


# ============================================================
# SAVE
# ============================================================

def save_predictions(
    result
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_columns = [
        "date",
        "store_nbr",
        "family",
        "city",
        "state",
        "type",
        "cluster",
        "onpromotion",
        "predicted_sales",
    ]

    result = result[
        output_columns
    ].copy()

    result = result.sort_values(
        [
            "date",
            "store_nbr",
            "family"
        ]
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 70)

    print(
        "FUTURE FORECAST SAVED"
    )

    print(
        f"File: {OUTPUT_PATH}"
    )

    print(
        f"Rows: {len(result):,}"
    )

    print(
        f"Dates: "
        f"{result['date'].min().date()} -> "
        f"{result['date'].max().date()}"
    )

    return result


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    result
):

    print("\n" + "=" * 70)
    print("FORECAST SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal predicted demand: "
        f"{result['predicted_sales'].sum():,.2f}"
    )

    print(
        f"Average prediction: "
        f"{result['predicted_sales'].mean():,.2f}"
    )

    print("\nDaily forecast:")

    daily = (
        result
        .groupby("date")[
            "predicted_sales"
        ]
        .sum()
        .reset_index()
    )

    daily["date"] = daily[
        "date"
    ].dt.strftime("%Y-%m-%d")

    print(
        daily.to_string(index=False)
    )

    print("\nTop 10 predicted store-family combinations:")

    top = (
        result
        .sort_values(
            "predicted_sales",
            ascending=False
        )
        .head(10)
    )

    print(
        top[
            [
                "date",
                "store_nbr",
                "family",
                "predicted_sales"
            ]
        ].to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # 1. Load historical data
    history = load_history()

    # 2. Load holidays
    (
        national,
        regional,
        local
    ) = load_holidays()

    # 3. Load oil
    (
        oil,
        last_oil_price
    ) = load_oil()

    # 4. Load model
    model = load_model()

    # --------------------------------------------------------
    # Last historical date
    # --------------------------------------------------------

    last_date = history[
        "date"
    ].max()

    print(
        f"\nLast historical date: "
        f"{last_date.date()}"
    )

    # --------------------------------------------------------
    # Future dates
    # --------------------------------------------------------

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=FORECAST_DAYS,
        freq="D"
    )

    print(
        f"Forecasting next "
        f"{FORECAST_DAYS} days."
    )

    print(
        f"Forecast period: "
        f"{future_dates.min().date()} -> "
        f"{future_dates.max().date()}"
    )

    # --------------------------------------------------------
    # Create future rows
    # --------------------------------------------------------

    future = create_future_rows(
        history,
        future_dates,
        national,
        regional,
        local,
        last_oil_price
    )

    # --------------------------------------------------------
    # Generate recursive forecast
    # --------------------------------------------------------

    result = forecast(
        history,
        future,
        model
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result = save_predictions(
        result
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        result
    )

    print("\n" + "=" * 70)
    print(
        "FUTURE FORECASTING COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()