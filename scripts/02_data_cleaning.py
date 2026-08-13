from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("DATA CLEANING AND MERGING")
    print("=" * 70)

    print("\nLoading train.csv...")

    train = pd.read_csv(
        RAW_DATA_PATH / "train.csv",
        parse_dates=["date"]
    )

    print(f"Train shape: {train.shape}")
    print(f"Train dates: {train['date'].min()} -> {train['date'].max()}")

    print("\nLoading stores.csv...")

    stores = pd.read_csv(RAW_DATA_PATH / "stores.csv")

    print(f"Stores shape: {stores.shape}")

    print("\nLoading oil.csv...")

    oil = pd.read_csv(
        RAW_DATA_PATH / "oil.csv",
        parse_dates=["date"]
    )

    print(f"Oil shape: {oil.shape}")

    print("\nLoading transactions.csv...")

    transactions = pd.read_csv(
        RAW_DATA_PATH / "transactions.csv",
        parse_dates=["date"]
    )

    print(f"Transactions shape: {transactions.shape}")

    return train, stores, oil, transactions


# ============================================================
# CLEAN TRAIN DATA
# ============================================================

def clean_train(train):

    print("\n" + "=" * 70)
    print("CLEANING TRAIN DATA")
    print("=" * 70)

    train = train.copy()

    n_dupes = train.duplicated().sum()
    print(f"\nDuplicate rows: {n_dupes}")

    train = train.drop_duplicates()

    print(f"Shape after cleaning: {train.shape}")
    print(f"Date range: {train['date'].min()} -> {train['date'].max()}")

    return train


# ============================================================
# CLEAN STORE DATA
# ============================================================

def clean_stores(stores):

    print("\n" + "=" * 70)
    print("CLEANING STORE DATA")
    print("=" * 70)

    stores = stores.copy()

    n_dupe_ids = stores["store_nbr"].duplicated().sum()
    print(f"\nDuplicate store IDs: {n_dupe_ids}")
    print(f"Stores: {stores['store_nbr'].nunique()}")

    return stores


# ============================================================
# PREPARE DAILY OIL DATA
# ============================================================

def clean_oil(oil, start_date, end_date):

    print("\n" + "=" * 70)
    print("PREPARING DAILY OIL DATA")
    print("=" * 70)

    oil = oil.copy()

    oil["date"] = pd.to_datetime(oil["date"])
    oil = oil.sort_values("date")

    print(f"\nOriginal oil rows: {len(oil):,}")
    print(
        f"Original oil date range: "
        f"{oil['date'].min().date()} -> {oil['date'].max().date()}"
    )

    n_missing_before = oil["dcoilwtico"].isna().sum()
    print(f"Missing prices before filling: {n_missing_before:,}")

    oil = oil.rename(columns={"dcoilwtico": "oil_price"})

    # Create COMPLETE daily date range.
    # oil.csv does not contain every calendar day (e.g. weekends).
    daily_dates = pd.DataFrame(
        {"date": pd.date_range(start=start_date, end=end_date, freq="D")}
    )

    print(f"\nTraining period: {start_date.date()} -> {end_date.date()}")

    daily_oil = daily_dates.merge(
        oil[["date", "oil_price"]],
        on="date",
        how="left",
        validate="one_to_one"
    )

    print(f"Daily oil rows: {len(daily_oil):,}")

    # Forward fill uses the previous known oil price.
    # Backward fill handles any missing value at the start of the period.
    daily_oil["oil_price"] = daily_oil["oil_price"].ffill().bfill()

    n_missing_after = daily_oil["oil_price"].isna().sum()
    print(f"Missing prices after filling: {n_missing_after:,}")

    if daily_oil["oil_price"].isna().any():
        raise ValueError("Oil price still contains missing values after filling.")

    # BUG FIX: PROCESSED_DATA_PATH may not exist yet — create it before writing.
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    oil_output = PROCESSED_DATA_PATH / "daily_oil.csv"
    daily_oil.to_csv(oil_output, index=False)

    print(f"\n✅ Saved daily oil data: {oil_output}")

    return daily_oil


# ============================================================
# CLEAN TRANSACTION DATA
# ============================================================

def clean_transactions(transactions):

    print("\n" + "=" * 70)
    print("CLEANING TRANSACTION DATA")
    print("=" * 70)

    transactions = transactions.copy()

    # BUG FIX: precompute the duplicate count instead of embedding a
    # multi-line method call inside an f-string (SyntaxError on Python < 3.12).
    dupe_mask = transactions.duplicated(subset=["date", "store_nbr"])
    n_dupes = dupe_mask.sum()
    print(f"\nDuplicate date/store combinations: {n_dupes}")

    # BUG FIX: train dropped its duplicates but transactions never did.
    # Left as-is, duplicate (date, store_nbr) rows would violate the
    # many_to_one merge validation later (or silently duplicate rows
    # if validation were removed). Drop them here for consistency.
    if n_dupes > 0:
        transactions = transactions.drop_duplicates(subset=["date", "store_nbr"])
        print(f"Dropped {n_dupes} duplicate rows.")

    print(f"Transaction rows: {len(transactions):,}")

    return transactions


# ============================================================
# MERGE STORES
# ============================================================

def merge_stores(train, stores):

    print("\n[1/3] Merging stores...")

    before = len(train)

    train = train.merge(
        stores,
        on="store_nbr",
        how="left",
        validate="many_to_one"
    )

    after = len(train)

    print(f"Shape: {train.shape}")

    if before != after:
        raise ValueError("Store merge changed the number of rows.")

    return train


# ============================================================
# MERGE OIL
# ============================================================

def merge_oil(train, daily_oil):

    print("\n[2/3] Merging daily oil prices...")

    before = len(train)

    train = train.merge(
        daily_oil,
        on="date",
        how="left",
        validate="many_to_one"
    )

    after = len(train)

    print(f"Shape: {train.shape}")

    if before != after:
        raise ValueError("Oil merge changed the number of rows.")

    missing_oil = train["oil_price"].isna().sum()
    print(f"Missing oil prices after merge: {missing_oil:,}")

    if missing_oil > 0:
        raise ValueError("Oil price contains missing values after merging.")

    return train


# ============================================================
# MERGE TRANSACTIONS
# ============================================================

def merge_transactions(train, transactions):

    print("\n[3/3] Merging transactions...")

    before = len(train)

    train = train.merge(
        transactions,
        on=["date", "store_nbr"],
        how="left",
        validate="many_to_one"
    )

    after = len(train)

    print(f"Shape: {train.shape}")

    if before != after:
        raise ValueError("Transaction merge changed the number of rows.")

    # BUG FIX: unlike oil, transactions legitimately has gaps (e.g. closed
    # stores), so we don't raise — but we should still surface how many
    # rows are missing instead of silently dropping the info.
    missing_transactions = train["transactions"].isna().sum()
    print(f"Missing transactions after merge: {missing_transactions:,}")

    return train


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_processed_data(train, original_rows):

    print("\n" + "-" * 70)
    print(f"Original train rows : {original_rows:,}")
    print(f"Final merged rows   : {len(train):,}")

    if len(train) == original_rows:
        print("✅ Row count preserved.")
    else:
        raise ValueError("Row count changed during merging.")

    print("\nFinal columns:")
    print(train.columns.tolist())

    print("\nFinal shape:")
    print(train.shape)

    print("\nFinal date range:")
    print(train["date"].min())
    print(train["date"].max())

    print("\nUnique dates:")
    print(train["date"].nunique())

    print("\nMissing values:")
    missing = train.isna().sum()
    print(missing[missing > 0])

    # BUG FIX: ensure the output directory exists before writing
    # (in case clean_oil() was skipped or reordered in the future).
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    output_file = PROCESSED_DATA_PATH / "train_processed.csv"

    print("\nSaving processed dataset...")
    train.to_csv(output_file, index=False)

    print(f"\n✅ Saved to: {output_file}")


# ============================================================
# MAIN
# ============================================================

def main():

    train, stores, oil, transactions = load_data()

    original_rows = len(train)

    start_date = train["date"].min()
    end_date = train["date"].max()

    train = clean_train(train)
    stores = clean_stores(stores)
    daily_oil = clean_oil(oil, start_date, end_date)
    transactions = clean_transactions(transactions)

    train = merge_stores(train, stores)
    train = merge_oil(train, daily_oil)
    train = merge_transactions(train, transactions)

    save_processed_data(train, original_rows)

    print("\n" + "=" * 70)
    print("✅ DATA CLEANING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()