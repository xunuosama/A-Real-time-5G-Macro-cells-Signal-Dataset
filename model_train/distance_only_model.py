from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path(
    r"/table/Grid-Based_Feature-Engineered_data/fishnet_30x30_processed_feature_engineered.csv"
)
MODEL_PATH = Path("../distance_only_linear_regression.joblib")

FEATURE_COLUMN = "log_True_3D_Dist"
TARGET_COLUMN = "SS_RSRP"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    df = pd.read_csv(DATA_PATH, usecols=[FEATURE_COLUMN, TARGET_COLUMN])
    df = df.dropna(subset=[FEATURE_COLUMN, TARGET_COLUMN])

    X = df[[FEATURE_COLUMN]]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    joblib.dump(model, MODEL_PATH)

    print(f"Data path: {DATA_PATH}")
    print(f"Rows used: {len(df)}")
    print(f"Feature: {FEATURE_COLUMN}")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Coefficient: {model.coef_[0]:.6f}")
    print(f"Intercept: {model.intercept_:.6f}")
    print(f"R2: {r2:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Model saved to: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
