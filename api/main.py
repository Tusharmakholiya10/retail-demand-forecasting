from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Retail Demand Forecasting API",
    description=(
        "Production-style REST API for retail demand forecasting, "
        "inventory recommendations, and model evaluation metrics."
    ),
    version="1.2.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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
# DATA LOADING
# ============================================================

def load_csv(path: Path) -> pd.DataFrame:
    """
    Load a CSV file and return a DataFrame.

    Raises:
        HTTPException: If the file does not exist or cannot be read.
    """

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


# ============================================================
# JSON CONVERSION
# ============================================================

def dataframe_to_records(
    df: pd.DataFrame,
) -> list[dict]:
    """
    Convert a DataFrame into JSON-safe records.
    """

    df = df.copy()

    # Convert dates to YYYY-MM-DD strings.
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):
            df[column] = df[column].dt.strftime(
                "%Y-%m-%d"
            )

    # Replace infinity values.
    df = df.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    # Convert NaN/NA to None.
    return (
        df.astype(object)
        .where(df.notna(), None)
        .to_dict(orient="records")
    )


# ============================================================
# FILTERING
# ============================================================

def filter_dataframe(
    df: pd.DataFrame,
    store: Optional[int] = None,
    family: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply store, family, and date filters.
    """

    filtered = df.copy()

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    if store is not None:

        if "store_nbr" not in filtered.columns:
            raise HTTPException(
                status_code=500,
                detail=(
                    "The dataset does not contain "
                    "'store_nbr'."
                ),
            )

        filtered = filtered[
            filtered["store_nbr"] == store
        ]

    # --------------------------------------------------------
    # Product Family
    # --------------------------------------------------------

    if family is not None:

        if "family" not in filtered.columns:
            raise HTTPException(
                status_code=500,
                detail=(
                    "The dataset does not contain "
                    "'family'."
                ),
            )

        filtered = filtered[
            filtered["family"]
            .astype(str)
            .str.upper()
            == family.upper()
        ]

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    if date_from is not None or date_to is not None:

        if "date" not in filtered.columns:
            raise HTTPException(
                status_code=500,
                detail=(
                    "The dataset does not contain "
                    "'date'."
                ),
            )

        filtered = filtered.copy()

        try:
            filtered["date"] = pd.to_datetime(
                filtered["date"]
            )

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Unable to process date column: {exc}"
                ),
            )

        if date_from is not None:

            try:
                start_date = pd.to_datetime(date_from)

            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Invalid date_from. "
                        "Use YYYY-MM-DD."
                    ),
                )

            filtered = filtered[
                filtered["date"] >= start_date
            ]

        if date_to is not None:

            try:
                end_date = pd.to_datetime(date_to)

            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Invalid date_to. "
                        "Use YYYY-MM-DD."
                    ),
                )

            filtered = filtered[
                filtered["date"] <= end_date
            ]

        filtered["date"] = filtered["date"].dt.strftime(
            "%Y-%m-%d"
        )

    return filtered


# ============================================================
# PAGINATION
# ============================================================

def paginate_dataframe(
    df: pd.DataFrame,
    limit: int,
    offset: int,
) -> tuple[pd.DataFrame, int]:
    """
    Apply offset/limit pagination.
    """

    total = len(df)

    paginated = df.iloc[
        offset: offset + limit
    ]

    return paginated, total


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": (
            "Retail Demand Forecasting API is running"
        ),
        "version": "1.2.0",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "forecast": "/forecast",
            "inventory": "/inventory",
            "metrics": "/metrics",
            "docs": "/docs",
        },
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "retail-demand-forecasting-api",
        "version": "1.2.0",
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
        min_length=1,
    ),

    date_from: Optional[str] = Query(
        default=None,
        description="Start date in YYYY-MM-DD format.",
    ),

    date_to: Optional[str] = Query(
        default=None,
        description="End date in YYYY-MM-DD format.",
    ),

    limit: int = Query(
        default=100,
        description="Maximum number of records to return.",
        ge=1,
        le=5000,
    ),

    offset: int = Query(
        default=0,
        description="Number of records to skip.",
        ge=0,
    ),
):
    """
    Return future demand forecasts.

    Supported filters:

    - store
    - family
    - date_from
    - date_to
    - limit
    - offset
    """

    df = load_csv(FORECAST_PATH)

    filtered = filter_dataframe(
        df,
        store=store,
        family=family,
        date_from=date_from,
        date_to=date_to,
    )

    # If filters were supplied but no records match,
    # return a useful API response instead of an empty
    # unexplained dataset.
    if filtered.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "No forecast records found "
                "for the supplied filters."
            ),
        )

    paginated, total = paginate_dataframe(
        filtered,
        limit=limit,
        offset=offset,
    )

    return {
        "count": len(paginated),
        "total": total,
        "limit": limit,
        "offset": offset,
        "filters": {
            "store": store,
            "family": family,
            "date_from": date_from,
            "date_to": date_to,
        },
        "data": dataframe_to_records(
            paginated
        ),
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
        min_length=1,
    ),

    limit: int = Query(
        default=100,
        description="Maximum number of records to return.",
        ge=1,
        le=5000,
    ),

    offset: int = Query(
        default=0,
        description="Number of records to skip.",
        ge=0,
    ),
):
    """
    Return inventory and reorder recommendations.
    """

    df = load_csv(INVENTORY_PATH)

    filtered = filter_dataframe(
        df,
        store=store,
        family=family,
    )

    if filtered.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "No inventory records found "
                "for the supplied filters."
            ),
        )

    paginated, total = paginate_dataframe(
        filtered,
        limit=limit,
        offset=offset,
    )

    return {
        "count": len(paginated),
        "total": total,
        "limit": limit,
        "offset": offset,
        "filters": {
            "store": store,
            "family": family,
        },
        "data": dataframe_to_records(
            paginated
        ),
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

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No model metrics available.",
        )

    return {
        "count": len(df),
        "model": "Tuned LightGBM",
        "data": dataframe_to_records(df),
    }