import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from src.feature_engineering import get_feature_columns


# ---------------------------------------------------
# Prepare training data
# ---------------------------------------------------

def prepare_training_data(df):

    features = get_feature_columns()

    X = df[features]

    y = df["target"]

    return X, y


# ---------------------------------------------------
# Train base models
# ---------------------------------------------------

def train_models(df):

    X, y = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    # Linear model
    lr_model = LinearRegression()

    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )

    # Gradient Boosting
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6
    )

    lr_model.fit(X_train, y_train)

    rf_model.fit(X_train, y_train)

    gb_model.fit(X_train, y_train)

    return lr_model, rf_model, gb_model


# ---------------------------------------------------
# Model evaluation
# ---------------------------------------------------

def evaluate_models(lr_model, rf_model, gb_model, df):

    X, y = prepare_training_data(df)

    lr_pred = lr_model.predict(X)
    rf_pred = rf_model.predict(X)
    gb_pred = gb_model.predict(X)

    return {

        "lr_mse": mean_squared_error(y, lr_pred),

        "rf_mse": mean_squared_error(y, rf_pred),

        "gb_mse": mean_squared_error(y, gb_pred)

    }


# ---------------------------------------------------
# Ensemble prediction
# ---------------------------------------------------

def predict_next_price(lr_model, rf_model, gb_model, df):

    features = get_feature_columns()

    latest_row = df.iloc[-1:]

    X_latest = latest_row[features]

    lr_prediction = lr_model.predict(X_latest)[0]

    rf_prediction = rf_model.predict(X_latest)[0]

    gb_prediction = gb_model.predict(X_latest)[0]

    # Weighted ensemble
    prediction = (

        (lr_prediction * 0.15) +

        (rf_prediction * 0.35) +

        (gb_prediction * 0.50)

    )

    return {

        "lr_prediction": lr_prediction,

        "rf_prediction": rf_prediction,

        "gb_prediction": gb_prediction,

        "ensemble_prediction": prediction
    }
