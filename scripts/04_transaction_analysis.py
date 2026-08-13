from pathlib import Path
import pandas as pd


RAW_DATA_PATH = Path("data/raw")


def main():

    print("=" * 70)
    print("TRANSACTION DATA ANALYSIS")
    print("=" * 70)

    transactions = pd.read_csv(
        RAW_DATA_PATH / "transactions.csv",
        parse_dates=["date"]
    )

    train = pd.read_csv(
        RAW_DATA_PATH / "train.csv",
        usecols=["date", "store_nbr"],
        parse_dates=["date"]
    )

    print("\nTransaction dataset:")
    print(f"Rows: {len(transactions):,}")
    print(f"Date range: {transactions['date'].min()} -> "
          f"{transactions['date'].max()}")
    print(f"Unique dates: {transactions['date'].nunique():,}")
    print(f"Unique stores: {transactions['store_nbr'].nunique()}")

    print("\nTransaction date coverage:")

    print(
        transactions.groupby("date")
        .size()
        .describe()
    )

    # --------------------------------------------------------
    # Find missing date/store combinations
    # --------------------------------------------------------

    expected = (
        train[["date", "store_nbr"]]
        .drop_duplicates()
    )

    actual = (
        transactions[["date", "store_nbr"]]
        .drop_duplicates()
    )

    comparison = expected.merge(
        actual,
        on=["date", "store_nbr"],
        how="left",
        indicator=True
    )

    missing = comparison[
        comparison["_merge"] == "left_only"
    ]

    print("\nMissing date/store combinations:")
    print(len(missing))

    print("\nMissing combinations by year:")

    missing["year"] = missing["date"].dt.year

    print(
        missing.groupby("year")
        .size()
    )

    print("\nMissing combinations by store:")

    print(
        missing.groupby("store_nbr")
        .size()
        .sort_values(ascending=False)
        .head(20)
    )

    print("\nFirst 30 missing combinations:")

    print(
        missing[
            ["date", "store_nbr"]
        ].head(30).to_string(index=False)
    )


if __name__ == "__main__":
    main()