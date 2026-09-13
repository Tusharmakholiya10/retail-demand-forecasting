from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Retail Demand Forecasting API",
    description=(
        "REST API for retail demand forecasts, "
        "inventory recommendations, and model metrics."
    ),
    version="1.0.0",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FORECAST_PATH = (
    BASE_DIR
    / "data"
    / "forecasts"
    / "future"
    / "future_predictions.csv"
)

INVENTORY_PATH = (
    BASE_DIR
    / "data"
    / "forecasts"
    / "inventory"
    / "inventory_recommendations.csv"
)

METRICS_PATH = (
    BASE_DIR
    / "data"
    / "forecasts"
    / "final_evaluation"
    / "overall_metrics.csv"
)


# ============================================================
# HELPERS
# ============================================================

def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file or raise a clear API error."""

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Data file not found: {path}",
        )

    try:
        return pd.read_csv(path)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read data file: {exc}",
        )


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame into JSON-safe records."""

    df = df.copy()

    # Convert datetime columns to strings.
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column] = df[column].dt.strftime("%Y-%m-%d")

    # Replace NaN/inf values with None.
    df = df.replace([float("inf"), float("-inf")], pd.NA)

    return df.astype(object).where(df.notna(), None).to_dict(
        orient="records"
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Retail Demand Forecasting API is running",
        "version": "1.0.0",
        "endpoints": [
            "/forecast",
            "/inventory",
            "/metrics",
            "/health",
        ],
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "retail-demand-forecasting-api",
    }


# ============================================================
# FORECAST
# ============================================================

@app.get("/forecast")
def get_forecast():
    """
    Return future demand forecasts.
    """

    df = load_csv(FORECAST_PATH)

    return {
        "count": len(df),
        "data": dataframe_to_records(df),
    }


# ============================================================
# INVENTORY
# ============================================================

@app.get("/inventory")
def get_inventory():
    """
    Return inventory and reorder recommendations.
    """

    df = load_csv(INVENTORY_PATH)

    return {
        "count": len(df),
        "data": dataframe_to_records(df),
    }


# ============================================================
# METRICS
# ============================================================

@app.get("/metrics")
def get_metrics():
    """
    Return final model evaluation metrics.
    """

    df = load_csv(METRICS_PATH)

    return {
        "count": len(df),
        "data": dataframe_to_records(df),
    }