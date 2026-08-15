from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models/lightgbm")
FORECAST_DIR = Path("data/forecasts")

TRAIN_FILE = PROCESSED_DIR / "train_data.csv"
VALIDATION_FILE = PROCESSED_DIR / "validation_data.csv"

BEST_MODEL_FILE = MODEL_DIR / "tuned_lightgbm_model.txt"
RESULTS_FILE = FORECAST_DIR / "lightgbm_tuning_results.csv"


# ============================================================
# METRICS
# ============================================================

def calculate_mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))


def calculate_rmse(actual, predicted):
    return np.sqrt(
        np.mean((actual - predicted) ** 2)
    )


def calculate_rmsle(actual, predicted):

    actual = np.maximum(actual, 0)
    predicted = np.maximum(predicted, 0)

    return np.sqrt(
        np.mean(
            (
                np.log1p(actual)
                - np.log1p(predicted)
            ) ** 2
        )
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LIGHTGBM HYPERPARAMETER TUNING")
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
        f"Validation shape: {validation.shape}"
    )

    return train, validation


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(train, validation):

    print("\n" + "=" * 70)
    print("PREPARING FEATURES")
    print("=" * 70)

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
        f"\nNumber of features: {len(features)}"
    )

    print("\nFeatures:")

    for feature in features:
        print(f"  - {feature}")

    X_train = train[features].copy()
    y_train = train["sales"].copy()

    X_validation = validation[features].copy()
    y_validation = validation["sales"].copy()

    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------

    categorical_columns = [
        col for col in [
            "family",
            "city",
            "state",
            "store_type",
            "type"
        ]
        if col in X_train.columns
    ]

    print("\nCategorical features:")
    print(categorical_columns)

    print(
        "\nCategorical features:"
    )

    print(categorical_columns)

    for column in categorical_columns:

        if column in X_train.columns:

            # Use the same categories for train and validation
            combined = pd.concat(
                [
                    X_train[column],
                    X_validation[column]
                ],
                ignore_index=True
            ).astype("category")

            categories = combined.cat.categories

            X_train[column] = (
                pd.Categorical(
                    X_train[column],
                    categories=categories
                )
            )

            X_validation[column] = (
                pd.Categorical(
                    X_validation[column],
                    categories=categories
                )
            )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        features,
        categorical_columns
    )


# ============================================================
# PARAMETER CONFIGURATIONS
# ============================================================

def get_parameter_configurations():

    configurations = [

        {
            "name": "config_1_balanced",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0
        },

        {
            "name": "config_2_more_leaves",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0
        },

        {
            "name": "config_3_regularized",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1
        },

        {
            "name": "config_4_deeper",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 12,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1
        },

        {
            "name": "config_5_lower_learning_rate",
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1
        }
    ]

    return configurations


# ============================================================
# TRAIN ONE CONFIGURATION
# ============================================================

def train_configuration(
    configuration,
    X_train,
    y_train,
    X_validation,
    y_validation,
    categorical_columns
):

    print("\n" + "-" * 70)

    print(
        f"TRAINING: {configuration['name']}"
    )

    print("-" * 70)

    model = lgb.LGBMRegressor(

        objective="regression",

        n_estimators=1500,

        learning_rate=configuration[
            "learning_rate"
        ],

        num_leaves=configuration[
            "num_leaves"
        ],

        max_depth=configuration[
            "max_depth"
        ],

        min_child_samples=configuration[
            "min_child_samples"
        ],

        subsample=configuration[
            "subsample"
        ],

        colsample_bytree=configuration[
            "colsample_bytree"
        ],

        reg_alpha=configuration[
            "reg_alpha"
        ],

        reg_lambda=configuration[
            "reg_lambda"
        ],

        random_state=42,

        n_jobs=-1,

        verbosity=-1
    )

    print("\nTraining model...")

    model.fit(

        X_train,

        y_train,

        categorical_feature=categorical_columns,

        eval_X=X_validation,

        eval_y=y_validation,

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=75,
                verbose=False
            )
        ]
    )

    print(
        f"Best iteration: "
        f"{model.best_iteration_}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_validation,
        num_iteration=model.best_iteration_
    )

    predictions = np.maximum(
        predictions,
        0
    )

    actual = y_validation.values

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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
        f"\nMAE   : {mae:.4f}"
    )

    print(
        f"RMSE  : {rmse:.4f}"
    )

    print(
        f"RMSLE : {rmsle:.4f}"
    )

    result = {

        "configuration":
            configuration["name"],

        "best_iteration":
            model.best_iteration_,

        "learning_rate":
            configuration["learning_rate"],

        "num_leaves":
            configuration["num_leaves"],

        "max_depth":
            configuration["max_depth"],

        "min_child_samples":
            configuration["min_child_samples"],

        "subsample":
            configuration["subsample"],

        "colsample_bytree":
            configuration["colsample_bytree"],

        "reg_alpha":
            configuration["reg_alpha"],

        "reg_lambda":
            configuration["reg_lambda"],

        "MAE":
            mae,

        "RMSE":
            rmse,

        "RMSLE":
            rmsle
    }

    return model, result


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    FORECAST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "MAE"
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    print(
        f"\n✅ Tuning results saved:"
        f"\n{RESULTS_FILE}"
    )

    return results_df


# ============================================================
# MAIN
# ============================================================

def main():

    train, validation = load_data()

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        features,
        categorical_columns
    ) = prepare_features(
        train,
        validation
    )

    configurations = (
        get_parameter_configurations()
    )

    print(
        "\n" + "=" * 70
    )

    print(
        f"Testing {len(configurations)} "
        f"LightGBM configurations"
    )

    print(
        "=" * 70
    )

    results = []

    best_model = None
    best_result = None

    # --------------------------------------------------------
    # Train configurations
    # --------------------------------------------------------

    for configuration in configurations:

        model, result = train_configuration(

            configuration,

            X_train,
            y_train,

            X_validation,
            y_validation,

            categorical_columns
        )

        results.append(
            result
        )

        # ----------------------------------------------------
        # Select best model based on MAE
        # ----------------------------------------------------

        if (
            best_result is None
            or result["MAE"]
            < best_result["MAE"]
        ):

            best_result = result
            best_model = model

            print(
                "\n🏆 New best configuration:"
            )

            print(
                f"   {result['configuration']}"
            )

            print(
                f"   MAE: {result['MAE']:.4f}"
            )

    # --------------------------------------------------------
    # Save tuning results
    # --------------------------------------------------------

    results_df = save_results(
        results
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_model.booster_.save_model(
        str(BEST_MODEL_FILE)
    )

    print(
        f"\n✅ Best model saved:"
        f"\n{BEST_MODEL_FILE}"
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "LIGHTGBM TUNING RESULTS"
    )

    print(
        "=" * 70
    )

    display_columns = [
        "configuration",
        "best_iteration",
        "MAE",
        "RMSE",
        "RMSLE"
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "🏆 BEST CONFIGURATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nConfiguration:"
        f" {best_result['configuration']}"
    )

    print(
        f"MAE   : {best_result['MAE']:.4f}"
    )

    print(
        f"RMSE  : {best_result['RMSE']:.4f}"
    )

    print(
        f"RMSLE : {best_result['RMSLE']:.4f}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "✅ LIGHTGBM TUNING COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()