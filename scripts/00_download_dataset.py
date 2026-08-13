
from pathlib import Path
import subprocess
import sys


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "competitions/store-sales-time-series-forecasting"
RAW_DATA_PATH = Path("data/raw")


# ============================================================
# DOWNLOAD DATASET
# ============================================================

def download_dataset():

    print("=" * 60)
    print("RETAIL DEMAND FORECASTING")
    print("KAGGLE DATASET DOWNLOAD")
    print("=" * 60)

    # Create raw data directory
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

    print("\n📦 Dataset:")
    print(DATASET)

    print("\n📁 Download location:")
    print(RAW_DATA_PATH.resolve())

    print("\n⬇️ Downloading dataset from Kaggle...")

    command = [
        sys.executable,
        "-m",
        "kaggle",
        "competitions",
        "download",
        "-c",
        "store-sales-time-series-forecasting",
        "-p",
        str(RAW_DATA_PATH)
    ]

    try:

        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )

        print(result.stdout)

        print("\n✅ Dataset downloaded successfully.")

    except subprocess.CalledProcessError as error:

        print("\n❌ Dataset download failed.")

        if error.stdout:
            print("\nKaggle output:")
            print(error.stdout)

        if error.stderr:
            print("\nError:")
            print(error.stderr)

        return False

    return True


# ============================================================
# EXTRACT ZIP FILE
# ============================================================

def extract_dataset():

    import zipfile

    zip_files = list(RAW_DATA_PATH.glob("*.zip"))

    if not zip_files:

        print("\n⚠️ No ZIP file found.")
        return False

    for zip_file in zip_files:

        print(f"\n📦 Extracting: {zip_file.name}")

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(RAW_DATA_PATH)

        print("✅ Extraction complete.")

    return True


# ============================================================
# SHOW DOWNLOADED FILES
# ============================================================

def show_files():

    print("\n" + "=" * 60)
    print("FILES IN RAW DATA DIRECTORY")
    print("=" * 60)

    files = list(RAW_DATA_PATH.iterdir())

    if not files:

        print("\n❌ No files found.")
        return

    for file in files:

        if file.is_file():

            size_mb = file.stat().st_size / (1024 * 1024)

            print(f"{file.name:<30} {size_mb:.2f} MB")


# ============================================================
# MAIN
# ============================================================

def main():

    success = download_dataset()

    if not success:
        return

    extract_dataset()

    show_files()

    print("\n" + "=" * 60)
    print("✅ DATASET SETUP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
    