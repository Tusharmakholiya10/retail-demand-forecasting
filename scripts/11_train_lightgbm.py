from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DATA_PATH = Path("data/processed")
MODEL_PATH = Path("models")
FORECAST_PATH = Path("data/forecasts")

TRAIN_FILE = (
    PROCESSED_DATA_PATH /
    "train_data.csv"
)

VALIDATION_FILE = (
    PROCESSED_DATA_PATH /
    "validation_data.csv"
)

MODEL_FILE = (
    MODEL_PATH /
    "lightgbm_model.txt"
)

PREDICTIONS_FILE = (
    FORECAST_PATH /
    "lightgbm_predictions.csv"
)

RESULTS_FILE = (
    FORECAST_PATH /
    "lightgbm_results.csv"
)


# ============================================================
# METRICS
# ============================================================

def calculate_mae(actual, predicted):

    return np.mean(
        np.abs(actual - predicted)
    )


def calculate_rmse(actual, predicted):

    return np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )


def calculate_rmsle(actual, predicted):

    actual = np.maximum(
        actual,
        0
    )

    predicted = np.maximum(
        predicted,
        0
    )

    return np.sqrt(
        np.mean(
            (
                np.log1p(actual)
                -
                np.log1p(predicted)
            ) ** 2
        )
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LIGHTGBM DEMAND FORECASTING")
    print("=" * 70)

    print("\nLoading training data...")

    train = pd.read_csv(
        TRAIN_FILE,
        parse_dates=["date"]
    )

    print(
        f"Training shape: {train.shape}"
    )

    print("\nLoading validation data...")

    validation = pd.read_csv(
        VALIDATION_FILE,
        parse_dates=["date"]
    )

    print(
        f"Validation shape: "
        f"{validation.shape}"
    )

    return train, validation


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(train, validation):

    print("\n" + "=" * 70)
    print("PREPARING FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target = "sales"

    # --------------------------------------------------------
    # Features
    #
    # We deliberately exclude:
    #
    # id
    # date
    # sales
    # transactions
    #
    # transactions contains missing values and will be
    # handled later as a separate experiment.
    # --------------------------------------------------------

    excluded_columns = [
        "id",
        "date",
        "sales",
        "transactions"
    ]

    features = [
        column
        for column in train.columns
        if column not in excluded_columns
    ]

    print(
        f"\nNumber of features: "
        f"{len(features)}"
    )

    print("\nFeatures:")

    for feature in features:

        print(
            f"  - {feature}"
        )

    X_train = train[features].copy()
    y_train = train[target].copy()

    X_validation = validation[
        features
    ].copy()

    y_validation = validation[
        target
    ].copy()

    # --------------------------------------------------------
    # Convert categorical columns
    # --------------------------------------------------------

    categorical_columns = [
    "family",
    "city",
    "state",
    "store_type",
    "type"
]

    for column in categorical_columns:

        if column in X_train.columns:

            X_train[column] = (
                X_train[column]
                .astype("category")
            )

            X_validation[column] = (
                X_validation[column]
                .astype("category")
            )

            # Ensure both datasets use the same categories
            categories = (
                X_train[column]
                .cat.categories
            )

            X_validation[column] = (
                X_validation[column]
                .cat.set_categories(
                    categories
                )
            )

    print(
        "\nCategorical features:"
    )

    print(
        categorical_columns
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        features
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
    X_validation,
    y_validation
):

    print("\n" + "=" * 70)
    print("TRAINING LIGHTGBM")
    print("=" * 70)

    model = lgb.LGBMRegressor(

        objective="regression",

        n_estimators=1000,

        learning_rate=0.05,

        num_leaves=64,

        max_depth=-1,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42,

        n_jobs=-1
    )

    print(
        "\nTraining model..."
    )

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (
                X_validation,
                y_validation
            )
        ],

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50
            ),
            lgb.log_evaluation(
                period=50
            )
        ]
    )

    print(
        "\n✅ Model training complete."
    )

    return model


# ============================================================
# EVALUATE
# ============================================================

def evaluate_model(
    model,
    X_validation,
    y_validation
):

    print("\n" + "=" * 70)
    print("EVALUATING LIGHTGBM")
    print("=" * 70)

    predictions = model.predict(
        X_validation
    )

    # --------------------------------------------------------
    # Sales cannot be negative
    # --------------------------------------------------------

    predictions = np.maximum(
        predictions,
        0
    )

    actual = (
        y_validation
        .values
    )

    mae = calculate_mae(
        actual,
        predictions
    )

    rmse = calculate_rmse(
        actual,
        predictions
    )

    rmsle = calculate_rmsle(
        actual,
        predictions
    )

    print(
        f"\nMAE   : {mae:,.4f}"
    )

    print(
        f"RMSE  : {rmse:,.4f}"
    )

    print(
        f"RMSLE : {rmsle:,.4f}"
    )

    results = pd.DataFrame(
        [
            {
                "model": "LightGBM",
                "rows_evaluated": len(
                    actual
                ),
                "MAE": mae,
                "RMSE": rmse,
                "RMSLE": rmsle
            }
        ]
    )

    return (
        predictions,
        results
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    MODEL_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    model.booster_.save_model(
        str(MODEL_FILE)
    )

    print(
        f"\n✅ Model saved:"
        f"\n{MODEL_FILE}"
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    validation,
    predictions
):

    FORECAST_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    output = validation[
        [
            "date",
            "store_nbr",
            "family",
            "sales"
        ]
    ].copy()

    output[
        "prediction"
    ] = predictions

    output.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    print(
        f"\n✅ Predictions saved:"
        f"\n{PREDICTIONS_FILE}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    FORECAST_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        RESULTS_FILE,
        index=False
    )

    print(
        f"\n✅ Results saved:"
        f"\n{RESULTS_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    train, validation = (
        load_data()
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        features
    ) = prepare_features(
        train,
        validation
    )

    model = train_model(
        X_train,
        y_train,
        X_validation,
        y_validation
    )

    (
        predictions,
        results
    ) = evaluate_model(
        model,
        X_validation,
        y_validation
    )

    save_model(
        model
    )

    save_predictions(
        validation,
        predictions
    )

    save_results(
        results
    )

    print("\n" + "=" * 70)
    print("✅ LIGHTGBM FORECASTING COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()