import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler


# --------------------------------------------------
# Create time-series sequences
# --------------------------------------------------

def create_sequences(data, window_size=20):

    X = []
    y = []

    for i in range(len(data) - window_size):

        X.append(data[i:i + window_size])

        y.append(data[i + window_size])

    return np.array(X), np.array(y)


# --------------------------------------------------
# Prepare LSTM data
# --------------------------------------------------

def prepare_lstm_data(df, window_size=20):

    prices = df["close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()

    scaled_prices = scaler.fit_transform(prices)

    X, y = create_sequences(scaled_prices, window_size)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y, scaler


# --------------------------------------------------
# Build LSTM network
# --------------------------------------------------

def build_lstm_model(window_size):

    model = Sequential()

    model.add(
        LSTM(
            64,
            return_sequences=True,
            input_shape=(window_size, 1)
        )
    )

    model.add(Dropout(0.2))

    model.add(LSTM(64))

    model.add(Dropout(0.2))

    model.add(Dense(32, activation="relu"))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


# --------------------------------------------------
# Train model
# --------------------------------------------------

def train_lstm(df, window_size=20, epochs=20):

    X, y, scaler = prepare_lstm_data(df, window_size)

    model = build_lstm_model(window_size)

    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=32,
        verbose=0
    )

    return model, scaler


# --------------------------------------------------
# Generate prediction
# --------------------------------------------------

def predict_lstm(model, scaler, df, window_size=20):

    prices = df["close"].values.reshape(-1, 1)

    scaled_prices = scaler.transform(prices)

    last_window = scaled_prices[-window_size:]

    last_window = last_window.reshape((1, window_size, 1))

    prediction_scaled = model.predict(last_window)

    prediction = scaler.inverse_transform(prediction_scaled)

    return float(prediction[0][0])
