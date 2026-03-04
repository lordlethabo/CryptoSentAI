
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


FEATURE_COLUMNS = [
    "close",
    "sentiment",
    "price_change",
    "rolling_mean",
    "volatility",
    "momentum",
    "sma_5",
    "sma_10",
    "ema_10",
    "rsi"
]


def prepare_training_data(df):
    """
    Extract features and target from the feature-engineered dataframe.
    """

    X = df[FEATURE_COLUMNS]
    y = df["target"]

    return X, y


def train_models(df):
    """
    Train Linear Regression and Random Forest models.
    """

    X, y = prepare_training_data(df)

    lr_model = LinearRegression()
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
    Evaluate model performance using Mean Squared Error.
    """

    X, y = prepare_training_data(df)

    pred_lr = lr_model.predict(X)
    pred_rf = rf_model.predict(X)

    mse_lr = mean_squared_error(y, pred_lr)
    mse_rf = mean_squared_error(y, pred_rf)

    return {
        "linear_regression_mse": mse_lr,
        "random_forest_mse": mse_rf
    }


def ensemble_prediction(lr_model, rf_model, feature_row):
    """
    Combine predictions from multiple models.
    """

    pred_lr = lr_model.predict(feature_row)[0]
    pred_rf = rf_model.predict(feature_row)[0]

    prediction = (pred_lr + pred_rf) / 2

    return prediction


def make_predictions(lr_model, rf_model, df):
    """
    Predict the next price using the most recent feature row.
    """

    latest_features = df[FEATURE_COLUMNS].iloc[-1:]

    prediction = ensemble_prediction(
        lr_model,
        rf_model,
        latest_features
    )

    return prediction
