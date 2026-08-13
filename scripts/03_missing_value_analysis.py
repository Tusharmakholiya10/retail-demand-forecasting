from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/processed/train_processed.csv")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MISSING VALUE ANALYSIS")
    print("=" * 70)

    print("\nLoading processed dataset...")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"]
    )

    print(f"Shape: {df.shape}")

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values:")

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    print(missing)

    # --------------------------------------------------------
    # Missing percentage
    # --------------------------------------------------------

    print("\nMissing percentage:")

    missing_percentage = (
        df.isnull().mean() * 100
    )

    missing_percentage = (
        missing_percentage[
            missing_percentage > 0
        ]
        .sort_values(ascending=False)
    )

    print(
        missing_percentage.round(2)
    )

    # --------------------------------------------------------
    # Missing values by date
    # --------------------------------------------------------

    print("\nMissing values by date:")

    date_missing = (
        df.groupby("date")
        [["oil_price", "transactions"]]
        .apply(lambda x: x.isnull().sum())
    )

    print(date_missing.head(20))

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    print("\nDate range:")

    print("Start:", df["date"].min())
    print("End  :", df["date"].max())

    # --------------------------------------------------------
    # Oil missing dates
    # --------------------------------------------------------

    oil_missing_dates = (
        df.loc[
            df["oil_price"].isnull(),
            "date"
        ]
        .drop_duplicates()
        .sort_values()
    )

    print(
        "\nNumber of dates with missing oil price:",
        len(oil_missing_dates)
    )

    print("\nFirst missing oil dates:")

    print(
        oil_missing_dates.head(20).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Transaction missing dates
    # --------------------------------------------------------

    transaction_missing_dates = (
        df.loc[
            df["transactions"].isnull(),
            "date"
        ]
        .drop_duplicates()
        .sort_values()
    )

    print(
        "\nNumber of dates with missing transactions:",
        len(transaction_missing_dates)
    )

    print("\nFirst missing transaction dates:")

    print(
        transaction_missing_dates.head(20).to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()