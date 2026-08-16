"""
16_predict.py

FINAL DEMAND FORECASTING INFERENCE PIPELINE

Loads the trained tuned LightGBM model and generates
predictions from validation/modeling data without retraining.
"""

import os
import pandas as pd
import lightgbm as lgb


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/lightgbm/tuned_lightgbm_model.txt"

INPUT_PATH = "data/processed/validation_data.csv"

OUTPUT_DIR = "data/forecasts/inference"

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "predictions.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "store_nbr",
    "family",
    "onpromotion",
    "city",
    "state",
    "type",
    "cluster",
    "oil_price",
    "year",
    "month",
    "quarter",
    "week_of_year",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "has_national_holiday",
    "is_regional_holiday",
    "is_local_holiday",
    "log_onpromotion",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28"
]


CATEGORICAL_COLUMNS = [
    "family",
    "city",
    "state",
    "type"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("FINAL DEMAND FORECASTING INFERENCE")
    print("=" * 70)

    print("\nLoading validation data...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Input shape: {df.shape}")

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    print("\nPreparing features...")

    missing_features = [
        col for col in FEATURES
        if col not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    X = df[FEATURES].copy()

    # Convert categorical columns to pandas category
    for col in CATEGORICAL_COLUMNS:

        X[col] = X[col].astype("category")

    # Make sure numerical columns are numeric
    numerical_columns = [
        col for col in FEATURES
        if col not in CATEGORICAL_COLUMNS
    ]

    for col in numerical_columns:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    # Check missing values
    missing = X.isna().sum()

    missing = missing[missing > 0]

    if len(missing) > 0:

        print("\nMissing feature values:")

        print(missing)

    else:

        print("No missing feature values.")

    return X


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading tuned LightGBM model...")

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = lgb.Booster(
        model_file=MODEL_PATH
    )

    print("✅ Model loaded successfully.")

    return model


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(model, X):

    print("\nGenerating predictions...")

    predictions = model.predict(X)

    # Demand cannot be negative
    predictions = predictions.clip(min=0)

    print(
        f"Predictions generated: {len(predictions):,}"
    )

    return predictions


# ============================================================
# CREATE OUTPUT
# ============================================================

def create_output(df, predictions):

    print("\nCreating prediction output...")

    output_columns = [
        "id",
        "date",
        "store_nbr",
        "family"
    ]

    output = df[output_columns].copy()

    output["actual_sales"] = df["sales"]

    output["predicted_sales"] = predictions

    output["error"] = (
        output["actual_sales"]
        - output["predicted_sales"]
    )

    output["absolute_error"] = (
        output["error"].abs()
    )

    output["percentage_error"] = (
        output["absolute_error"]
        /
        output["actual_sales"].replace(0, pd.NA)
        * 100
    )

    return output


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_output(output):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n✅ Predictions saved:")

    print(OUTPUT_PATH)

    print(
        f"Rows saved: {len(output):,}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(output):

    print("\n" + "=" * 70)
    print("INFERENCE SUMMARY")
    print("=" * 70)

    print(
        f"Rows predicted       : {len(output):,}"
    )

    print(
        f"Actual total sales   : "
        f"{output['actual_sales'].sum():,.2f}"
    )

    print(
        f"Predicted total sales: "
        f"{output['predicted_sales'].sum():,.2f}"
    )

    print(
        f"Mean absolute error  : "
        f"{output['absolute_error'].mean():,.4f}"
    )

    print(
        f"Mean prediction      : "
        f"{output['predicted_sales'].mean():,.4f}"
    )

    print("\nFirst 10 predictions:")

    print(
        output.head(10).to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # 1. Load data
    df = load_data()

    # 2. Prepare features
    X = prepare_features(df)

    # 3. Load trained model
    model = load_model()

    # 4. Generate predictions
    predictions = generate_predictions(
        model,
        X
    )

    # 5. Create output
    output = create_output(
        df,
        predictions
    )

    # 6. Save
    save_output(output)

    # 7. Summary
    print_summary(output)

    print("\n" + "=" * 70)
    print("✅ INFERENCE PIPELINE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()