from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Retail Demand Forecasting API",
    description=(
        "REST API for retail demand forecasts, "
        "inventory recommendations, and model metrics."
    ),
    version="1.1.0",
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
    """Load a CSV file and return it as a DataFrame."""

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
    """Convert DataFrame rows into JSON-safe dictionaries."""

    df = df.copy()

    # Convert datetime columns to readable strings.
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column] = df[column].dt.strftime("%Y-%m-%d")

    # Replace infinite and missing values.
    df = df.replace([float("inf"), float("-inf")], pd.NA)

    return (
        df.astype(object)
        .where(df.notna(), None)
        .to_dict(orient="records")
    )


def filter_dataframe(
    df: pd.DataFrame,
    store: Optional[int] = None,
    family: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply common store and product-family filters.
    """

    filtered = df.copy()

    if store is not None:
        if "store_nbr" not in filtered.columns:
            raise HTTPException(
                status_code=500,
                detail="The dataset does not contain 'store_nbr'.",
            )

        filtered = filtered[
            filtered["store_nbr"] == store
        ]

    if family is not None:
        if "family" not in filtered.columns:
            raise HTTPException(
                status_code=500,
                detail="The dataset does not contain 'family'.",
            )

        filtered = filtered[
            filtered["family"].astype(str).str.upper()
            == family.upper()
        ]

    return filtered


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Retail Demand Forecasting API is running",
        "version": "1.1.0",
        "endpoints": [
            "/forecast",
            "/inventory",
            "/metrics",
            "/health",
            "/docs",
        ],
    }


# ============================================================
# HEALTH
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
def get_forecast(
    store: Optional[int] = Query(
        default=None,
        description="Filter by store number.",
        ge=1,
    ),
    family: Optional[str] = Query(
        default=None,
        description="Filter by product family.",
    ),
):
    """
    Return future demand forecasts.

    Optional filters:
    - store
    - family
    """

    df = load_csv(FORECAST_PATH)

    filtered = filter_dataframe(
        df,
        store=store,
        family=family,
    )

    return {
        "count": len(filtered),
        "filters": {
            "store": store,
            "family": family,
        },
        "data": dataframe_to_records(filtered),
    }


# ============================================================
# INVENTORY
# ============================================================

@app.get("/inventory")
def get_inventory(
    store: Optional[int] = Query(
        default=None,
        description="Filter by store number.",
        ge=1,
    ),
    family: Optional[str] = Query(
        default=None,
        description="Filter by product family.",
    ),
):
    """
    Return inventory and reorder recommendations.

    Optional filters:
    - store
    - family
    """

    df = load_csv(INVENTORY_PATH)

    filtered = filter_dataframe(
        df,
        store=store,
        family=family,
    )

    return {
        "count": len(filtered),
        "filters": {
            "store": store,
            "family": family,
        },
        "data": dataframe_to_records(filtered),
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