from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")


# ============================================================
# PREPARE OIL DATA
# ============================================================

def prepare_oil():

    print("=" * 70)
    print("PREPARING DAILY OIL PRICE DATA")
    print("=" * 70)

    # Load training dates
    train = pd.read_csv(
        RAW_DATA_PATH / "train.csv",
        usecols=["date"],
        parse_dates=["date"]
    )

    # Get complete date range
    start_date = train["date"].min()
    end_date = train["date"].max()

    print(f"\nTraining period:")
    print(f"{start_date.date()} -> {end_date.date()}")

    # Load oil data
    oil = pd.read_csv(
        RAW_DATA_PATH / "oil.csv",
        parse_dates=["date"]
    )

    oil = oil.rename(
        columns={
            "dcoilwtico": "oil_price"
        }
    )

    oil = oil.sort_values("date")

    print(f"\nOriginal oil rows: {len(oil):,}")
    print(
        f"Original oil date range: "
        f"{oil['date'].min().date()} -> "
        f"{oil['date'].max().date()}"
    )

    print(
        f"Missing prices before filling: "
        f"{oil['oil_price'].isna().sum()}"
    )

    # --------------------------------------------------------
    # Create complete daily date range
    # --------------------------------------------------------

    daily_dates = pd.DataFrame(
        {
            "date": pd.date_range(
                start=start_date,
                end=end_date,
                freq="D"
            )
        }
    )

    # --------------------------------------------------------
    # Merge oil observations onto complete date range
    # --------------------------------------------------------

    daily_oil = daily_dates.merge(
        oil[["date", "oil_price"]],
        on="date",
        how="left"
    )

    # --------------------------------------------------------
    # Fill missing prices
    # --------------------------------------------------------

    daily_oil["oil_price"] = (
        daily_oil["oil_price"]
        .ffill()
        .bfill()
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print(
        f"\nDaily oil rows: {len(daily_oil):,}"
    )

    print(
        f"Missing prices after filling: "
        f"{daily_oil['oil_price'].isna().sum()}"
    )

    print(
        f"Date range: "
        f"{daily_oil['date'].min().date()} -> "
        f"{daily_oil['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    PROCESSED_DATA_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        PROCESSED_DATA_PATH /
        "daily_oil.csv"
    )

    daily_oil.to_csv(
        output_file,
        index=False
    )

    print(
        f"\n✅ Saved: {output_file}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    prepare_oil()