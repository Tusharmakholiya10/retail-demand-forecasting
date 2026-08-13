from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DATA_PATH = Path("data/raw")


# ============================================================
# FUNCTION: LOAD CSV
# ============================================================

def load_csv(filename):
    """Load a CSV file from the raw data directory."""

    file_path = RAW_DATA_PATH / filename

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)

    return df


# ============================================================
# FUNCTION: EXPLORE DATAFRAME
# ============================================================

def explore_dataframe(df, filename):

    print("\n")
    print("=" * 70)
    print(f"DATASET: {filename}")
    print("=" * 70)

    # Shape
    print("\n📐 Shape:")
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]:,}")

    # Columns
    print("\n📋 Columns:")
    for column in df.columns:
        print(f"  - {column}")

    # Data types
    print("\n🔤 Data Types:")
    print(df.dtypes)

    # First rows
    print("\n👀 First 5 Rows:")
    print(df.head())

    # Missing values
    print("\n❓ Missing Values:")

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        print("  No missing values.")
    else:
        print(missing)

    # Duplicate rows
    print("\n♻️ Duplicate Rows:")
    print(df.duplicated().sum())

    # Basic statistics
    print("\n📊 Basic Statistics:")

    print(df.describe(include="all").T)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RETAIL DEMAND FORECASTING")
    print("DATA EXPLORATION")
    print("=" * 70)

    csv_files = list(RAW_DATA_PATH.glob("*.csv"))

    if not csv_files:

        print("\n❌ No CSV files found.")
        print("Run 00_download_dataset.py first.")

        return

    print(f"\nFound {len(csv_files)} CSV files.")

    for file_path in csv_files:

        try:

            df = load_csv(file_path.name)

            if df is not None:
                explore_dataframe(df, file_path.name)

        except Exception as error:

            print(f"\n❌ Error reading {file_path.name}")
            print(error)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()