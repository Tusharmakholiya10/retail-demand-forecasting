from pathlib import Path
import pandas as pd


RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")


def main():

    print("=" * 70)
    print("PREPARING HOLIDAY FEATURES")
    print("=" * 70)

    holidays = pd.read_csv(
        RAW_DATA_PATH / "holidays_events.csv",
        parse_dates=["date"]
    )

    # --------------------------------------------------------
    # Keep only the forecasting period
    # --------------------------------------------------------

    train = pd.read_csv(
        RAW_DATA_PATH / "train.csv",
        usecols=["date"],
        parse_dates=["date"]
    )

    start_date = train["date"].min()
    end_date = train["date"].max()

    holidays = holidays[
        (holidays["date"] >= start_date) &
        (holidays["date"] <= end_date)
    ].copy()

    print(
        f"\nForecasting period: "
        f"{start_date.date()} -> {end_date.date()}"
    )

    print(
        f"Holiday records in forecasting period: "
        f"{len(holidays)}"
    )

    # --------------------------------------------------------
    # National holiday features
    # --------------------------------------------------------

    national = holidays[
        holidays["locale"] == "National"
    ].copy()

    national_features = (
        national.groupby("date")
        .agg(
            is_national_holiday=("date", "size"),
            has_national_holiday=("date", "size")
        )
        .reset_index()
    )

    national_features["has_national_holiday"] = 1

    # --------------------------------------------------------
    # Regional holiday features
    # --------------------------------------------------------

    regional = holidays[
        holidays["locale"] == "Regional"
    ].copy()

    regional_features = (
        regional[
            ["date", "locale_name", "type", "transferred"]
        ]
        .rename(
            columns={
                "locale_name": "state",
                "type": "holiday_type",
                "transferred": "holiday_transferred"
            }
        )
    )

    regional_features["is_regional_holiday"] = 1

    # --------------------------------------------------------
    # Local holiday features
    # --------------------------------------------------------

    local = holidays[
        holidays["locale"] == "Local"
    ].copy()

    local_features = (
        local[
            ["date", "locale_name", "type", "transferred"]
        ]
        .rename(
            columns={
                "locale_name": "city",
                "type": "holiday_type",
                "transferred": "holiday_transferred"
            }
        )
    )

    local_features["is_local_holiday"] = 1

    # --------------------------------------------------------
    # Save national features
    # --------------------------------------------------------

    national_file = (
        PROCESSED_DATA_PATH /
        "national_holidays.csv"
    )

    national_features.to_csv(
        national_file,
        index=False
    )

    # --------------------------------------------------------
    # Save regional features
    # --------------------------------------------------------

    regional_file = (
        PROCESSED_DATA_PATH /
        "regional_holidays.csv"
    )

    regional_features.to_csv(
        regional_file,
        index=False
    )

    # --------------------------------------------------------
    # Save local features
    # --------------------------------------------------------

    local_file = (
        PROCESSED_DATA_PATH /
        "local_holidays.csv"
    )

    local_features.to_csv(
        local_file,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nNational holiday records:")
    print(len(national_features))

    print("\nRegional holiday records:")
    print(len(regional_features))

    print("\nLocal holiday records:")
    print(len(local_features))

    print("\nSaved files:")

    print(f"  {national_file}")
    print(f"  {regional_file}")
    print(f"  {local_file}")

    print("\n" + "=" * 70)
    print("✅ HOLIDAY FEATURE PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()