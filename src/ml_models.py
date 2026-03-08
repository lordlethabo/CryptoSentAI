import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from src.feature_engineering import get_feature_columns


def prepare_training_data(df):
    """
    Split dataframe into features and target.
    """

    features = get_feature_columns()

    X = df[features]

    y = df["target"]

    return X, y


def train_models(df):
    """
    Train multiple ML models.
    """

    X, y = prepare_training_data(df)

    # Linear Regression
    lr_model = LinearRegression()

    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    lr_model.fit(X, y)

    rf_model.fit(X, y)

    return lr_model, rf_model


def evaluate_models(lr_model, rf_model, df):
    """
    Evaluate model performance.
    """

    X, y = prepare_training_data(df)

    lr_pred = lr_model.predict(X)

    rf_pred = rf_model.predict(X)

    lr_mse = mean_squared_error(y, lr_pred)

    rf_mse = mean_squared_error(y, rf_pred)

    return {
        "linear_regression_mse": lr_mse,
        "random_forest_mse": rf_mse
    }


def predict_next_price(lr_model, rf_model, df):
    """
    Predict next price using ensemble of models.
    """

    features = get_feature_columns()

    latest_row = df.iloc[-1:]

    X_latest = latest_row[features]

    lr_prediction = lr_model.predict(X_latest)[0]

    rf_prediction = rf_model.predict(X_latest)[0]

    prediction = (lr_prediction + rf_prediction) / 2

    return {
        "lr_prediction": lr_prediction,
        "rf_prediction": rf_prediction,
        "ensemble_prediction": prediction
    }
