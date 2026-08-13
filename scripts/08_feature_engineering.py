from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DATA_PATH = Path("data/processed")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    print("\nLoading processed sales data...")

    df = pd.read_csv(
        PROCESSED_DATA_PATH / "train_processed.csv",
        parse_dates=["date"]
    )

    print(f"Shape: {df.shape}")

    print("\nLoading holiday features...")

    national = pd.read_csv(
        PROCESSED_DATA_PATH / "national_holidays.csv",
        parse_dates=["date"]
    )

    regional = pd.read_csv(
        PROCESSED_DATA_PATH / "regional_holidays.csv",
        parse_dates=["date"]
    )

    local = pd.read_csv(
        PROCESSED_DATA_PATH / "local_holidays.csv",
        parse_dates=["date"]
    )

    print(f"National holidays: {national.shape}")
    print(f"Regional holidays: {regional.shape}")
    print(f"Local holidays: {local.shape}")

    return df, national, regional, local


# ============================================================
# CALENDAR FEATURES
# ============================================================

def create_calendar_features(df):

    print("\n" + "=" * 70)
    print("CREATING CALENDAR FEATURES")
    print("=" * 70)

    df = df.copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    df["week_of_year"] = (
        df["date"].dt.isocalendar().week.astype("int16")
    )

    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype("int8")

    print("Calendar features created.")

    return df


# ============================================================
# OIL PRICE
# ============================================================

def merge_oil(df):

    print("\n" + "=" * 70)
    print("CHECKING OIL PRICE")
    print("=" * 70)

    df = df.copy()

    # oil_price was already added during 02_data_cleaning.py.
    # Therefore, we do not merge it again.

    if "oil_price" not in df.columns:
        raise ValueError(
            "oil_price column is missing from train_processed.csv"
        )

    missing = df["oil_price"].isna().sum()

    print(
        f"Missing oil prices currently: {missing:,}"
    )

    print(
        "✅ Existing oil_price feature will be used."
    )

    return df


# ============================================================
# NATIONAL HOLIDAYS
# ============================================================

def merge_national_holidays(
    df,
    national
):

    print("\n" + "=" * 70)
    print("ADDING NATIONAL HOLIDAY FEATURES")
    print("=" * 70)

    # Keep only the columns we need.
    national = national[
        ["date", "has_national_holiday"]
    ].copy()

    national = national.drop_duplicates(
        subset=["date"]
    )

    df = df.merge(
        national,
        on="date",
        how="left",
        validate="many_to_one"
    )

    df["has_national_holiday"] = (
        df["has_national_holiday"]
        .fillna(0)
        .astype("int8")
    )

    print(
        "National holiday feature added."
    )

    return df


# ============================================================
# REGIONAL HOLIDAYS
# ============================================================

def merge_regional_holidays(
    df,
    regional    
):

    print("\n" + "=" * 70)
    print("ADDING REGIONAL HOLIDAY FEATURES")
    print("=" * 70)

    regional = regional[
        [
            "date",
            "state",
            "is_regional_holiday"
        ]
    ].copy()

    regional = regional.drop_duplicates(
        subset=["date", "state"]
    )

    df = df.merge(
        regional,
        on=["date", "state"],
        how="left",
        validate="many_to_one"
    )

    df["is_regional_holiday"] = (
        df["is_regional_holiday"]
        .fillna(0)
        .astype("int8")
    )

    print(
        "Regional holiday feature added."
    )

    return df


# ============================================================
# LOCAL HOLIDAYS
# ============================================================

def merge_local_holidays(
    df,
    local
):

    print("\n" + "=" * 70)
    print("ADDING LOCAL HOLIDAY FEATURES")
    print("=" * 70)

    local = local[
        [
            "date",
            "city",
            "is_local_holiday"
        ]
    ].copy()

    local = local.drop_duplicates(
        subset=["date", "city"]
    )

    df = df.merge(
        local,
        on=["date", "city"],
        how="left",
        validate="many_to_one"
    )

    df["is_local_holiday"] = (
        df["is_local_holiday"]
        .fillna(0)
        .astype("int8")
    )

    print(
        "Local holiday feature added."
    )

    return df


# ============================================================
# PROMOTION FEATURES
# ============================================================

def create_promotion_features(df):

    print("\n" + "=" * 70)
    print("CREATING PROMOTION FEATURES")
    print("=" * 70)

    df = df.copy()

    df["onpromotion"] = (
        df["onpromotion"]
        .fillna(0)
    )

    # Log transformation helps reduce extreme values.
    df["log_onpromotion"] = np.log1p(
        df["onpromotion"]
    )

    print(
        "Promotion features created."
    )

    return df


# ============================================================
# SORT FOR TIME-SERIES FEATURES
# ============================================================

def sort_for_time_series(df):

    print("\nSorting data for time-series features...")

    df = df.sort_values(
        [
            "store_nbr",
            "family",
            "date"
        ]
    ).reset_index(drop=True)

    return df


# ============================================================
# LAG FEATURES
# ============================================================

def create_lag_features(df):

    print("\n" + "=" * 70)
    print("CREATING LAG FEATURES")
    print("=" * 70)

    group = df.groupby(
        ["store_nbr", "family"],
        sort=False
    )["sales"]

    print("Creating lag_1...")

    df["lag_1"] = group.shift(1)

    print("Creating lag_7...")

    df["lag_7"] = group.shift(7)

    print("Creating lag_14...")

    df["lag_14"] = group.shift(14)

    print("Creating lag_28...")

    df["lag_28"] = group.shift(28)

    print("Lag features created.")

    return df


# ============================================================
# ROLLING FEATURES
# ============================================================

def create_rolling_features(df):

    print("\n" + "=" * 70)
    print("CREATING ROLLING FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # IMPORTANT:
    # We shift by 1 before calculating rolling statistics.
    #
    # This prevents the current day's sales from being used
    # to predict the current day's sales.
    #
    # Example:
    #
    # sales on Aug 15
    #     ↓
    # NOT included in rolling_mean_7 for Aug 15
    #
    # Instead we use:
    # Aug 8 -> Aug 14
    # --------------------------------------------------------

    grouped = df.groupby(
        ["store_nbr", "family"],
        sort=False
    )["sales"]

    print("Creating rolling_mean_7...")

    df["rolling_mean_7"] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
             .rolling(
                 window=7,
                 min_periods=7
             )
             .mean()
        )
    )

    print("Creating rolling_mean_14...")

    df["rolling_mean_14"] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
             .rolling(
                 window=14,
                 min_periods=14
             )
             .mean()
        )
    )

    print("Creating rolling_mean_28...")

    df["rolling_mean_28"] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
             .rolling(
                 window=28,
                 min_periods=28
             )
             .mean()
        )
    )

    print("Creating rolling_std_7...")

    df["rolling_std_7"] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
             .rolling(
                 window=7,
                 min_periods=7
             )
             .std()
        )
    )

    print("Creating rolling_std_28...")

    df["rolling_std_28"] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
             .rolling(
                 window=28,
                 min_periods=28
             )
             .std()
        )
    )

    print("Rolling features created.")

    return df

    print("\n" + "=" * 70)
    print("CREATING ROLLING FEATURES")
    print("=" * 70)

    group = df.groupby(
        ["store_nbr", "family"],
        sort=False
    )["sales"]

    print("Creating rolling_mean_7...")

    df["rolling_mean_7"] = (
        group
        .shift(1)
        .rolling(7)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    print("Creating rolling_mean_14...")

    df["rolling_mean_14"] = (
        group
        .shift(1)
        .rolling(14)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    print("Creating rolling_mean_28...")

    df["rolling_mean_28"] = (
        group
        .shift(1)
        .rolling(28)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    print("Creating rolling_std_7...")

    df["rolling_std_7"] = (
        group
        .shift(1)
        .rolling(7)
        .std()
        .reset_index(level=[0, 1], drop=True)
    )

    print("Creating rolling_std_28...")

    df["rolling_std_28"] = (
        group
        .shift(1)
        .rolling(28)
        .std()
        .reset_index(level=[0, 1], drop=True)
    )

    print("Rolling features created.")

    return df


# ============================================================
# CLEAN FEATURE DATA
# ============================================================

def clean_features(df):

    print("\n" + "=" * 70)
    print("CLEANING FEATURE DATA")
    print("=" * 70)

    # Rows without enough historical information cannot
    # have all lag/rolling features.
    before = len(df)

    df = df.dropna(
        subset=[
            "lag_1",
            "lag_7",
            "lag_14",
            "lag_28",
            "rolling_mean_7",
            "rolling_mean_14",
            "rolling_mean_28"
        ]
    )

    after = len(df)

    print(
        f"Rows removed because of insufficient history: "
        f"{before - after:,}"
    )

    print(
        f"Rows remaining: {after:,}"
    )

    # Rolling standard deviation can occasionally be NaN.
    df["rolling_std_7"] = (
        df["rolling_std_7"].fillna(0)
    )

    df["rolling_std_28"] = (
        df["rolling_std_28"].fillna(0)
    )

    return df


# ============================================================
# SAVE DATA
# ============================================================

def save_features(df):

    print("\n" + "=" * 70)
    print("SAVING FEATURE DATASET")
    print("=" * 70)

    output_file = (
        PROCESSED_DATA_PATH /
        "modeling_data.csv"
    )

    print(
        f"\nFinal shape: {df.shape}"
    )

    print(
        f"Saving to: {output_file}"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        "\n✅ Feature dataset saved successfully."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df, national, regional, local = load_data()

    df = create_calendar_features(df)

    df = merge_oil(df)

    df = merge_national_holidays(
        df,
        national
    )

    df = merge_regional_holidays(
        df,
        regional
    )

    df = merge_local_holidays(
        df,
        local
    )

    df = create_promotion_features(df)

    df = sort_for_time_series(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = clean_features(df)

    save_features(df)

    print("\n" + "=" * 70)
    print("✅ FEATURE ENGINEERING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()