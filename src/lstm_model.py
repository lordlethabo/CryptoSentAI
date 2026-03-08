import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


def create_sequences(data, window_size=10):
    """
    Convert time series data into LSTM sequences.
    """

    X = []
    y = []

    for i in range(len(data) - window_size):

        X.append(data[i:i + window_size])
        y.append(data[i + window_size])

    return np.array(X), np.array(y)


def prepare_lstm_data(df, window_size=10):
    """
    Prepare closing price data for LSTM training.
    """

    prices = df["close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()

    scaled_prices = scaler.fit_transform(prices)

    X, y = create_sequences(scaled_prices, window_size)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y, scaler


def build_lstm_model(window_size=10):
    """
    Build LSTM neural network.
    """

    model = Sequential()

    model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


def train_lstm(df, window_size=10, epochs=10):
    """
    Train LSTM model.
    """

    X, y, scaler = prepare_lstm_data(df, window_size)

    model = build_lstm_model(window_size)

    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=16,
        verbose=0
    )

    return model, scaler


def predict_lstm(model, scaler, df, window_size=10):
    """
    Generate next price prediction using LSTM.
    """

    prices = df["close"].values.reshape(-1, 1)

    scaled_prices = scaler.transform(prices)

    last_window = scaled_prices[-window_size:]

    last_window = last_window.reshape((1, window_size, 1))

    prediction_scaled = model.predict(last_window)

    prediction = scaler.inverse_transform(prediction_scaled)

    return float(prediction[0][0])
