from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DATA_PATH = Path("data/processed")
MODEL_PATH = Path("models")
FORECAST_PATH = Path("data/forecasts")

TRAIN_FILE = PROCESSED_DATA_PATH / "train_data.csv"
VALIDATION_FILE = PROCESSED_DATA_PATH / "validation_data.csv"

MODEL_FILE = MODEL_PATH / "xgboost_model.json"
PREDICTIONS_FILE = FORECAST_PATH / "xgboost_predictions.csv"
RESULTS_FILE = FORECAST_PATH / "xgboost_results.csv"


# ============================================================
# METRICS
# ============================================================

def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))


def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def rmsle(actual, predicted):
    actual = np.maximum(actual, 0)
    predicted = np.maximum(predicted, 0)

    return np.sqrt(
        np.mean(
            (np.log1p(actual) - np.log1p(predicted)) ** 2
        )
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("XGBOOST DEMAND FORECASTING")
    print("=" * 70)

    print("\nLoading training data...")

    train = pd.read_csv(
        TRAIN_FILE,
        parse_dates=["date"]
    )

    print(f"Training shape: {train.shape}")

    print("\nLoading validation data...")

    validation = pd.read_csv(
        VALIDATION_FILE,
        parse_dates=["date"]
    )

    print(f"Validation shape: {validation.shape}")

    return train, validation


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(train, validation):

    print("\n" + "=" * 70)
    print("PREPARING FEATURES")
    print("=" * 70)

    excluded = [
        "id",
        "date",
        "sales",
        "transactions"
    ]

    features = [
        col for col in train.columns
        if col not in excluded
    ]

    print(f"\nNumber of features: {len(features)}")

    X_train = train[features].copy()
    y_train = train["sales"].copy()

    X_valid = validation[features].copy()
    y_valid = validation["sales"].copy()

    # --------------------------------------------------------
    # Convert categorical columns to integer category codes.
    # XGBoost expects numeric inputs.
    # --------------------------------------------------------

    categorical_columns = [
        "family",
        "city",
        "state",
        "type"
    ]

    for col in categorical_columns:

        if col in X_train.columns:

            combined = pd.concat(
                [
                    X_train[col],
                    X_valid[col]
                ],
                axis=0
            ).astype("category")

            X_train[col] = (
                combined.iloc[:len(X_train)]
                .cat.codes
                .astype("int32")
            )

            X_valid[col] = (
                combined.iloc[len(X_train):]
                .cat.codes
                .astype("int32")
                .values
            )

    print("\nCategorical columns encoded.")

    return (
        X_train,
        y_train,
        X_valid,
        y_valid,
        features
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
    X_valid,
    y_valid
):

    print("\n" + "=" * 70)
    print("TRAINING XGBOOST")
    print("=" * 70)

    model = xgb.XGBRegressor(

        objective="reg:squarederror",

        n_estimators=1000,

        learning_rate=0.05,

        max_depth=8,

        min_child_weight=5,

        subsample=0.8,

        colsample_bytree=0.8,

        tree_method="hist",

        n_jobs=-1,

        random_state=42
    )

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (
                X_valid,
                y_valid
            )
        ],

        verbose=50
    )

    print("\n✅ XGBoost training complete.")

    return model


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    model,
    X_valid,
    y_valid
):

    print("\n" + "=" * 70)
    print("EVALUATING XGBOOST")
    print("=" * 70)

    predictions = model.predict(X_valid)

    predictions = np.maximum(
        predictions,
        0
    )

    actual = y_valid.values

    result_mae = mae(
        actual,
        predictions
    )

    result_rmse = rmse(
        actual,
        predictions
    )

    result_rmsle = rmsle(
        actual,
        predictions
    )

    print(f"\nMAE   : {result_mae:.4f}")
    print(f"RMSE  : {result_rmse:.4f}")
    print(f"RMSLE : {result_rmsle:.4f}")

    results = pd.DataFrame(
        [
            {
                "model": "XGBoost",
                "rows_evaluated": len(actual),
                "MAE": result_mae,
                "RMSE": result_rmse,
                "RMSLE": result_rmsle
            }
        ]
    )

    return predictions, results


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    model,
    validation,
    predictions,
    results
):

    MODEL_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    FORECAST_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save_model(
        str(MODEL_FILE)
    )

    output = validation[
        [
            "date",
            "store_nbr",
            "family",
            "sales"
        ]
    ].copy()

    output["prediction"] = predictions

    output.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    results.to_csv(
        RESULTS_FILE,
        index=False
    )

    print(f"\n✅ Model saved: {MODEL_FILE}")
    print(f"✅ Predictions saved: {PREDICTIONS_FILE}")
    print(f"✅ Results saved: {RESULTS_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():

    train, validation = load_data()

    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        features
    ) = prepare_features(
        train,
        validation
    )

    model = train_model(
        X_train,
        y_train,
        X_valid,
        y_valid
    )

    predictions, results = evaluate(
        model,
        X_valid,
        y_valid
    )

    save_outputs(
        model,
        validation,
        predictions,
        results
    )

    print("\n" + "=" * 70)
    print("✅ XGBOOST FORECASTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()