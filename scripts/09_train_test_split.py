from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DATA_PATH = Path("data/processed")

INPUT_FILE = (
    PROCESSED_DATA_PATH /
    "modeling_data.csv"
)

TRAIN_FILE = (
    PROCESSED_DATA_PATH /
    "train_data.csv"
)

VALIDATION_FILE = (
    PROCESSED_DATA_PATH /
    "validation_data.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("TIME-SERIES TRAIN / VALIDATION SPLIT")
    print("=" * 70)

    print("\nLoading modeling data...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"]
    )

    print(
        f"Shape: {df.shape}"
    )

    return df


# ============================================================
# DATA AUDIT
# ============================================================

def audit_data(df):

    print("\n" + "=" * 70)
    print("DATA AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    print(
        f"\nDate range:"
        f"\nStart: {df['date'].min()}"
        f"\nEnd  : {df['date'].max()}"
    )

    print(
        f"\nUnique dates: "
        f"{df['date'].nunique():,}"
    )

    print(
        f"Unique stores: "
        f"{df['store_nbr'].nunique():,}"
    )

    print(
        f"Unique product families: "
        f"{df['family'].nunique():,}"
    )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicates = df.duplicated(
        subset=[
            "date",
            "store_nbr",
            "family"
        ]
    ).sum()

    print(
        f"\nDuplicate date/store/family rows: "
        f"{duplicates:,}"
    )

    if duplicates > 0:

        raise ValueError(
            "Duplicate date/store/family combinations found."
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing = df.isna().sum()

    missing = missing[
        missing > 0
    ]

    print("\nMissing values:")

    if len(missing) == 0:

        print("None")

    else:

        print(missing)

    # --------------------------------------------------------
    # Target statistics
    # --------------------------------------------------------

    print("\nSales statistics:")

    print(
        df["sales"].describe()
    )


# ============================================================
# TIME-BASED SPLIT
# ============================================================

def split_data(df):

    print("\n" + "=" * 70)
    print("CREATING TIME-BASED SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "date",
            "store_nbr",
            "family"
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Determine validation period
    #
    # Last 90 days are reserved for validation.
    # --------------------------------------------------------

    max_date = df["date"].max()

    validation_start = (
        max_date -
        pd.Timedelta(days=89)
    )

    print(
        f"\nValidation period:"
        f"\n{validation_start.date()} "
        f"-> "
        f"{max_date.date()}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train = df[
        df["date"] < validation_start
    ].copy()

    validation = df[
        df["date"] >= validation_start
    ].copy()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\nTraining data:")

    print(
        f"Rows: {len(train):,}"
    )

    print(
        f"Dates: "
        f"{train['date'].min().date()} "
        f"-> "
        f"{train['date'].max().date()}"
    )

    print("\nValidation data:")

    print(
        f"Rows: {len(validation):,}"
    )

    print(
        f"Dates: "
        f"{validation['date'].min().date()} "
        f"-> "
        f"{validation['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if train["date"].max() >= validation["date"].min():

        raise ValueError(
            "Time leakage detected between "
            "training and validation data."
        )

    if len(train) + len(validation) != len(df):

        raise ValueError(
            "Train + validation rows do not match "
            "original dataset."
        )

    print(
        "\n✅ Time-based split created successfully."
    )

    return train, validation


# ============================================================
# SAVE DATA
# ============================================================

def save_data(train, validation):

    print("\n" + "=" * 70)
    print("SAVING SPLIT DATA")
    print("=" * 70)

    train.to_csv(
        TRAIN_FILE,
        index=False
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False
    )

    print(
        f"\n✅ Training data saved:"
        f"\n{TRAIN_FILE}"
    )

    print(
        f"\n✅ Validation data saved:"
        f"\n{VALIDATION_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    audit_data(df)

    train, validation = split_data(
        df
    )

    save_data(
        train,
        validation
    )

    print("\n" + "=" * 70)
    print("✅ TRAIN / VALIDATION PREPARATION COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()