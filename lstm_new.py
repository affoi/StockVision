import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)

# LOAD DATA

def load_data(ticker="AAPL", period="10y"):
    df = yf.download(ticker, period=period, interval="1d")
    df.dropna(inplace=True)
    df["Return"] = df["Close"].pct_change().astype(float)
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df.dropna(inplace=True)
    return df

# FEATURE PREPARATION

def prepare_lstm_data(df, n_steps=100):
    data = df[["Close"]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i - n_steps:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# TRAIN OPTIMIZED LSTM

def train_optimized_lstm(df, n_steps=100, test_size=0.2):
    X, y, scaler = prepare_lstm_data(df, n_steps)
    split_idx = int(len(X) * (1 - test_size))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(n_steps, 1)),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)

    print("\n Training Optimized LSTM Model...")
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
        callbacks=[early_stop, reduce_lr]
    )

    # Predictions
    y_pred_scaled = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred_scaled)

    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    accuracy = max(0, min(100, r2 * 100))

    print("\nModel Performance:")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R²:   {r2:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Plot loss curves
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Training Loss", linewidth=2)
    plt.plot(history.history["val_loss"], label="Validation Loss", linestyle="--")
    plt.title("Training and Validation Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # Plot predictions
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_inv, label="Actual", color="black", linewidth=2)
    plt.plot(y_pred_inv, label="Predicted", color="dodgerblue", linestyle="--")
    plt.title("Actual vs Predicted Stock Prices (Optimized LSTM)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return model, scaler, rmse, mae, r2, accuracy

# PREDICT TODAY AND TOMORROW

def predict_today_and_tomorrow(df, model, scaler, n_steps=100):
    close_data = df["Close"].values.reshape(-1, 1)
    scaled_data = scaler.transform(close_data)

    # Prepare last n steps for today’s prediction
    X_last = scaled_data[-n_steps:].reshape(1, n_steps, 1)
    today_scaled = model.predict(X_last)
    today_pred = scaler.inverse_transform(today_scaled)[0][0]

    # Predict tomorrow — append today's pred and re-predict
    next_input = np.append(scaled_data[-(n_steps - 1):], today_scaled)
    X_next = next_input.reshape(1, n_steps, 1)
    tomorrow_scaled = model.predict(X_next)
    tomorrow_pred = scaler.inverse_transform(tomorrow_scaled)[0][0]

    print(f"\n Today's Date: {pd.Timestamp.today().strftime('%d-%m-%Y')}")
    print(f" Predicted Closing Price (Today): {today_pred:.2f}")
    print(f" Predicted Closing Price (Tomorrow): {tomorrow_pred:.2f}")

    # return today_pred, tomorrow_pred

# MAIN EXECUTION

if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g., AAPL, TSLA, INFY.NS): ").strip().upper()
    df = load_data(ticker)
    model, scaler, rmse, mae, r2, accuracy = train_optimized_lstm(df)
    predict_today_and_tomorrow(df, model, scaler)
