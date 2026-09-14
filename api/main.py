from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="Retail Demand Forecasting API",
    description=(
        "FastAPI backend for retail demand forecasting, "
        "inventory recommendations, model metrics, and metadata."
    ),
    version="1.2.0",
)

app = FastAPI(
    title="Retail Demand Forecasting API",
    description=(
        "FastAPI backend for retail demand forecasting, "
        "inventory recommendations, model metrics, and metadata."
    ),
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PREDICTIONS_PATH = (
    BASE_DIR
    / "data"
    / "forecasts"
    / "final_evaluation"
    / "final_predictions.csv"
)

FUTURE_FORECAST_PATH = (
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

MODEL_COMPARISON_PATH = (
    BASE_DIR
    / "data"
    / "forecasts"
    / "model_comparison.csv"
)

STORE_PERFORMANCE_PATH = (
    BASE_DIR
    / "data"
    / "forecasts"
    / "final_evaluation"
    / "store_performance.csv"
)

FAMILY_PERFORMANCE_PATH = (
    BASE_DIR
    / "data"
    / "forecasts"
    / "final_evaluation"
    / "family_performance.csv"
)

MONTHLY_PERFORMANCE_PATH = (
    BASE_DIR
    / "data"
    / "forecasts"
    / "final_evaluation"
    / "monthly_performance.csv"
)

OVERALL_METRICS_PATH = (
    BASE_DIR
    / "data"
    / "forecasts"
    / "final_evaluation"
    / "overall_metrics.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(path: Path) -> pd.DataFrame:
    """
    Load a CSV file and return it as a pandas DataFrame.
    Raise a clear API error if the file does not exist or cannot
    be read.
    """

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Data file not found: {path}"
        )

    try:
        return pd.read_csv(path)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read data file: {exc}"
        )


def apply_filters(
    df: pd.DataFrame,
    store: Optional[int] = None,
    family: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> pd.DataFrame:
    """
    Apply optional store, family, and date filters.
    """

    result = df.copy()

    # --------------------------------------------------------
    # Store filter
    # --------------------------------------------------------

    if store is not None:

        if "store_nbr" not in result.columns:
            raise HTTPException(
                status_code=400,
                detail="store_nbr column is not available in this dataset."
            )

        result["store_nbr"] = pd.to_numeric(
            result["store_nbr"],
            errors="coerce"
        )

        result = result[result["store_nbr"] == store]

    # --------------------------------------------------------
    # Family filter
    # --------------------------------------------------------

    if family is not None:

        if "family" not in result.columns:
            raise HTTPException(
                status_code=400,
                detail="family column is not available in this dataset."
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
                detail="date column is not available in this dataset."
            )

        result["date"] = pd.to_datetime(
            result["date"],
            errors="coerce"
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


def prepare_dataframe_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pandas-specific values into JSON-safe values.
    """

    result = df.copy()

    # Convert datetime columns to YYYY-MM-DD strings
    for column in result.columns:

        if pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = result[column].dt.strftime("%Y-%m-%d")

    # Replace NaN / inf values with None
    result = result.astype(object).where(
        pd.notna(result),
        None
    )

    return result


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    """
    Basic API information.
    """

    return {
        "name": "Retail Demand Forecasting API",
        "version": "1.2.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/forecast",
            "/inventory",
            "/stores",
            "/families",
            "/models",
            "/metrics",
        ],
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    """
    Check whether the API is running.
    """

    return {
        "status": "healthy",
        "service": "Retail Demand Forecasting API",
        "version": "1.2.0",
    }


# ============================================================
# FORECAST ENDPOINT
# ============================================================

@app.get("/forecast")
def get_forecast(
    store: Optional[int] = Query(
        default=None,
        description="Filter by store number"
    ),

    family: Optional[str] = Query(
        default=None,
        description="Filter by product family"
    ),

    date_from: Optional[date] = Query(
        default=None,
        description="Start date, inclusive"
    ),

    date_to: Optional[date] = Query(
        default=None,
        description="End date, inclusive"
    ),

    limit: int = Query(
        default=1000,
        ge=1,
        le=50000,
        description="Maximum number of records to return"
    ),
):
    """
    Return historical forecast predictions with optional filters.
    """

    df = load_csv(PREDICTIONS_PATH)

    total_matches_before_filtering = len(df)

    df = apply_filters(
        df=df,
        store=store,
        family=family,
        date_from=date_from,
        date_to=date_to,
    )

    total_matches = len(df)

    # Sort by date if available
    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        sort_columns = ["date"]

        if "store_nbr" in df.columns:
            sort_columns.append("store_nbr")

        if "family" in df.columns:
            sort_columns.append("family")

        df = df.sort_values(sort_columns)

    # Apply limit AFTER filtering
    df = df.head(limit)

    df = prepare_dataframe_for_json(df)

    return {
        "count": len(df),
        "total_matches": total_matches,
        "limit": limit,
        "filters": {
            "store": store,
            "family": family,
            "date_from": (
                date_from.isoformat()
                if date_from
                else None
            ),
            "date_to": (
                date_to.isoformat()
                if date_to
                else None
            ),
        },
        "data": df.to_dict(orient="records"),
    }


# ============================================================
# INVENTORY ENDPOINT
# ============================================================

@app.get("/inventory")
def get_inventory(
    store: Optional[int] = Query(
        default=None,
        description="Filter by store number"
    ),

    family: Optional[str] = Query(
        default=None,
        description="Filter by product family"
    ),

    date_from: Optional[date] = Query(
        default=None,
        description="Start date, inclusive"
    ),

    date_to: Optional[date] = Query(
        default=None,
        description="End date, inclusive"
    ),

    limit: int = Query(
        default=1000,
        ge=1,
        le=50000,
        description="Maximum number of records to return"
    ),
):
    """
    Return inventory recommendations with optional filters.
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

    # Sort by date if available
    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        sort_columns = ["date"]

        if "store_nbr" in df.columns:
            sort_columns.append("store_nbr")

        if "family" in df.columns:
            sort_columns.append("family")

        df = df.sort_values(sort_columns)

    # Apply limit
    df = df.head(limit)

    df = prepare_dataframe_for_json(df)

    return {
        "count": len(df),
        "total_matches": total_matches,
        "limit": limit,
        "filters": {
            "store": store,
            "family": family,
            "date_from": (
                date_from.isoformat()
                if date_from
                else None
            ),
            "date_to": (
                date_to.isoformat()
                if date_to
                else None
            ),
        },
        "data": df.to_dict(orient="records"),
    }


# ============================================================
# STORES ENDPOINT
# ============================================================

@app.get("/stores")
def get_stores():
    """
    Return all available store numbers.
    """

    df = load_csv(PREDICTIONS_PATH)

    if "store_nbr" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="store_nbr column not found in forecast data."
        )

    stores = (
        pd.to_numeric(
            df["store_nbr"],
            errors="coerce"
        )
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return {
        "count": len(stores),
        "stores": stores,
    }


# ============================================================
# FAMILIES ENDPOINT
# ============================================================

@app.get("/families")
def get_families():
    """
    Return all available product families.
    """

    df = load_csv(PREDICTIONS_PATH)

    if "family" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="family column not found in forecast data."
        )

    families = (
        df["family"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return {
        "count": len(families),
        "families": families,
    }


# ============================================================
# MODELS ENDPOINT
# ============================================================

@app.get("/models")
def get_models():
    """
    Return model comparison results.
    """

    df = load_csv(MODEL_COMPARISON_PATH)

    df = prepare_dataframe_for_json(df)

    return {
        "count": len(df),
        "data": df.to_dict(orient="records"),
    }


# ============================================================
# METRICS ENDPOINT
# ============================================================

@app.get("/metrics")
def get_metrics():
    """
    Return overall model evaluation metrics.
    """

    df = load_csv(OVERALL_METRICS_PATH)

    df = prepare_dataframe_for_json(df)

    return {
        "count": len(df),
        "data": df.to_dict(orient="records"),
    }