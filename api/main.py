from datetime import date
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
    """Load a CSV file and provide a clear API error if unavailable."""

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
    """Convert a DataFrame into JSON-safe records."""

    df = df.copy()

    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column] = df[column].dt.strftime("%Y-%m-%d")

    df = df.replace([float("inf"), float("-inf")], pd.NA)

    return (
        df.astype(object)
        .where(df.notna(), None)
        .to_dict(orient="records")
    )


def apply_filters(
    df: pd.DataFrame,
    store: Optional[int] = None,
    family: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> pd.DataFrame:
    """
    Apply common store, family and date filters.

    The function checks which columns actually exist in the
    dataset so the API remains robust to small schema differences.
    """

    result = df.copy()

    # --------------------------------------------------------
    # Store filter
    # --------------------------------------------------------

    if store is not None:

        if "store_nbr" not in result.columns:
            raise HTTPException(
                status_code=400,
                detail="Store filtering is not available for this dataset.",
            )

        result = result[result["store_nbr"] == store]

    # --------------------------------------------------------
    # Product family filter
    # --------------------------------------------------------

    if family is not None:

        if "family" not in result.columns:
            raise HTTPException(
                status_code=400,
                detail="Family filtering is not available for this dataset.",
            )

        result = result[
            result["family"].astype(str).str.upper()
            == family.upper()
        ]

    # --------------------------------------------------------
    # Date filters
    # --------------------------------------------------------

    if date_from is not None or date_to is not None:

        if "date" not in result.columns:
            raise HTTPException(
                status_code=400,
                detail="Date filtering is not available for this dataset.",
            )

        result["date"] = pd.to_datetime(
            result["date"],
            errors="coerce",
        )

        if date_from is not None:
            result = result[
                result["date"] >= pd.Timestamp(date_from)
            ]

        if date_to is not None:
            result = result[
                result["date"] <= pd.Timestamp(date_to)
            ]

    return result


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Retail Demand Forecasting API is running",
        "version": app.version,
        "endpoints": [
            "/health",
            "/forecast",
            "/inventory",
            "/metrics",
        ],
        "documentation": "/docs",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "retail-demand-forecasting-api",
        "version": app.version,
    }


# ============================================================
# FORECAST
# ============================================================

@app.get("/forecast")
def get_forecast(
    store: Optional[int] = Query(
        default=None,
        description="Store number",
    ),
    family: Optional[str] = Query(
        default=None,
        description="Product family",
    ),
    date_from: Optional[date] = Query(
        default=None,
        description="Start date (YYYY-MM-DD)",
    ),
    date_to: Optional[date] = Query(
        default=None,
        description="End date (YYYY-MM-DD)",
    ),
    limit: int = Query(
        default=1000,
        ge=1,
        le=50000,
        description="Maximum number of records to return",
    ),
):
    """
    Return future demand forecasts.

    Optional filters:
    - store
    - family
    - date_from
    - date_to
    """

    df = load_csv(FORECAST_PATH)

    df = apply_filters(
        df=df,
        store=store,
        family=family,
        date_from=date_from,
        date_to=date_to,
    )

    total_matches = len(df)

    # Sort by date when available.
    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )
        df = df.sort_values("date")

    df = df.head(limit)

    return {
        "count": len(df),
        "total_matches": total_matches,
        "limit": limit,
        "filters": {
            "store": store,
            "family": family,
            "date_from": date_from,
            "date_to": date_to,
        },
        "data": dataframe_to_records(df),
    }


# ============================================================
# INVENTORY
# ============================================================

@app.get("/inventory")
def get_inventory(
    store: Optional[int] = Query(
        default=None,
        description="Store number",
    ),
    family: Optional[str] = Query(
        default=None,
        description="Product family",
    ),
    date_from: Optional[date] = Query(
        default=None,
        description="Start date (YYYY-MM-DD)",
    ),
    date_to: Optional[date] = Query(
        default=None,
        description="End date (YYYY-MM-DD)",
    ),
    limit: int = Query(
        default=1000,
        ge=1,
        le=50000,
        description="Maximum number of records to return",
    ),
):
    """
    Return inventory and reorder recommendations.

    Optional filters:
    - store
    - family
    - date_from
    - date_to
    """

    df = load_csv(INVENTORY_PATH)

    df = apply_filters(
        df=df,
        store=store,
        family=family,
        date_from=date_from,
        date_to=date_to,
    )

    total_matches = len(df)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )
        df = df.sort_values("date")

    df = df.head(limit)

    return {
        "count": len(df),
        "total_matches": total_matches,
        "limit": limit,
        "filters": {
            "store": store,
            "family": family,
            "date_from": date_from,
            "date_to": date_to,
        },
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