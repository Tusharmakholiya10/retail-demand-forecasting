from pathlib import Path
import pandas as pd


RAW_DATA_PATH = Path("data/raw")


def main():

    print("=" * 70)
    print("HOLIDAY DATA ANALYSIS")
    print("=" * 70)

    holidays = pd.read_csv(
        RAW_DATA_PATH / "holidays_events.csv",
        parse_dates=["date"]
    )

    print("\nShape:")
    print(holidays.shape)

    print("\nColumns:")
    print(holidays.columns.tolist())

    print("\nData types:")
    print(holidays.dtypes)

    print("\nFirst 10 rows:")
    print(holidays.head(10).to_string(index=False))

    print("\nMissing values:")
    print(holidays.isna().sum())

    print("\nDuplicate rows:")
    print(holidays.duplicated().sum())

    print("\nHoliday types:")
    print(holidays["type"].value_counts())

    print("\nHoliday locales:")
    print(holidays["locale"].value_counts())

    print("\nTransferred:")
    print(holidays["transferred"].value_counts())

    print("\nDate range:")
    print(holidays["date"].min())
    print(holidays["date"].max())

    print("\nHoliday counts by year:")

    holidays["year"] = holidays["date"].dt.year

    print(
        holidays.groupby("year")
        .size()
    )

    print("\nLocale + type combinations:")

    print(
        holidays.groupby(
            ["locale", "type"]
        ).size()
    )

    print("\nSample local holidays:")

    print(
        holidays[
            holidays["locale"] == "Local"
        ].head(10).to_string(index=False)
    )

    print("\nSample regional holidays:")

    print(
        holidays[
            holidays["locale"] == "Regional"
        ].head(10).to_string(index=False)
    )

    print("\nSample national holidays:")

    print(
        holidays[
            holidays["locale"] == "National"
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()