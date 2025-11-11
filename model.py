import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Load Data

def load_data(ticker, period="10y"):
    df = yf.download(ticker, period=period, interval="1d")
    df.dropna(inplace=True)
    df["Return"] = df["Close"].pct_change()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["MA100"] = df["Close"].rolling(window=100).mean()
    df.dropna(inplace=True)
    return df

# Feature Preparation
def prepare_features(df):
    features = ["Open", "High", "Low", "Volume", "MA50", "MA100", "Return"]
    X = df[features]
    y = df["Close"]
    return X, y

# Train Models + Ensemble
def train_models(df_feat, test_size_days=365, random_state=42):
    X, y = prepare_features(df_feat)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test = X_scaled[:-test_size_days], X_scaled[-test_size_days:]
    y_train, y_test = y[:-test_size_days], y[-test_size_days:]

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    # Random Forest
    rf = RandomForestRegressor(random_state=random_state, n_estimators=200)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    # XGBoost
    xgb_model = xgb.XGBRegressor(random_state=random_state, n_estimators=200, learning_rate=0.05)
    xgb_model.fit(X_train, y_train)
    pred_xgb = xgb_model.predict(X_test)

    # LSTM
    n_steps = 10
    lstm_scaler = MinMaxScaler()
    close_scaled = lstm_scaler.fit_transform(y.values.reshape(-1, 1))

    X_lstm, y_lstm = [], []
    for i in range(n_steps, len(close_scaled)):
        X_lstm.append(close_scaled[i - n_steps:i])
        y_lstm.append(close_scaled[i])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    X_lstm_train, X_lstm_test = X_lstm[:-test_size_days], X_lstm[-test_size_days:]
    y_lstm_train, y_lstm_test = y_lstm[:-test_size_days], y_lstm[-test_size_days:]

    lstm_model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(n_steps, 1)),
        Dense(1)
    ])
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(X_lstm_train, y_lstm_train, epochs=10, batch_size=16, verbose=0)

    pred_lstm = lstm_model.predict(X_lstm_test)
    pred_lstm = lstm_scaler.inverse_transform(pred_lstm)
    y_lstm_test_inv = lstm_scaler.inverse_transform(y_lstm_test)

    pred_lr = np.ravel(pred_lr)
    pred_rf = np.ravel(pred_rf)
    pred_xgb = np.ravel(pred_xgb)
    pred_lstm = np.ravel(pred_lstm)
    y_test = np.ravel(y_lstm_test_inv)

    min_len = min(len(pred_lr), len(pred_lstm), len(y_test))
    preds = pd.DataFrame({
        "lr": pred_lr[-min_len:],
        "rf": pred_rf[-min_len:],
        "xgb": pred_xgb[-min_len:],
        "lstm": pred_lstm[-min_len:],
        "actual": y_test[-min_len:]
    }, index=df_feat.index[-min_len:])

    # Ensemble Meta Model
    meta_X = preds[["lr", "rf", "xgb", "lstm"]]
    meta_y = preds["actual"]
    meta_model = LinearRegression()
    meta_model.fit(meta_X, meta_y)
    preds["meta"] = meta_model.predict(meta_X)

    # Metrics
    rmse = np.sqrt(mean_squared_error(meta_y, preds["meta"]))
    mae = mean_absolute_error(meta_y, preds["meta"])
    r2 = r2_score(meta_y, preds["meta"])
    accuracy = r2 * 100

    print(f"\nModel Performance Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R²:   {r2:.3f}")
    print(f"Accuracy: {accuracy:.2f}%")

    return lr, rf, xgb_model, lstm_model, scaler, lstm_scaler, n_steps, preds, meta_model, rmse, mae, r2, accuracy

# Predict Today's and Tomorrow's Price
def predict_today_and_tomorrow(ticker, period="10y", test_size_days=365):
    df = load_data(ticker, period)
    lr, rf, xgb_model, lstm_model, scaler, lstm_scaler, n_steps, preds, meta_model, rmse, mae, r2, accuracy = train_models(df, test_size_days)

    # Prepare latest feature row
    last_row = df.iloc[-1][["Open", "High", "Low", "Volume", "MA50", "MA100", "Return"]].values.reshape(1, -1)
    last_scaled = scaler.transform(last_row)

    # Predictions from individual models
    pred_lr = float(lr.predict(last_scaled)[0])
    pred_rf = float(rf.predict(last_scaled)[0])
    pred_xgb = float(xgb_model.predict(last_scaled)[0])

    # LSTM today prediction
    close_scaled = lstm_scaler.transform(df["Close"].values.reshape(-1, 1))
    X_last = close_scaled[-n_steps:].reshape(1, n_steps, 1)
    pred_lstm_today = float(lstm_scaler.inverse_transform(lstm_model.predict(X_last))[0][0])

    # Ensemble today's prediction
    meta_input_today = np.array([[pred_lr, pred_rf, pred_xgb, pred_lstm_today]], dtype=float)
    today_pred = meta_model.predict(meta_input_today)[0]

    # Predict Tomorrow
    tomorrow_input_close = np.append(close_scaled[-n_steps+1:], lstm_scaler.transform([[today_pred]]))
    X_tomorrow = tomorrow_input_close.reshape(1, n_steps, 1)
    pred_lstm_tomorrow = float(lstm_scaler.inverse_transform(lstm_model.predict(X_tomorrow))[0][0])

    meta_input_tomorrow = np.array([[pred_lr, pred_rf, pred_xgb, pred_lstm_tomorrow]], dtype=float)
    tomorrow_pred = meta_model.predict(meta_input_tomorrow)[0]

    # Print Results
    print(f"\nToday's Date: {datetime.today().strftime('%d-%m-%Y')}")
    print(f"Predicted Closing Price for {ticker} Today: {today_pred:.2f}")
    print(f"Predicted Closing Price for {ticker} Tomorrow: {tomorrow_pred:.2f}")
    print(f"Ensemble Model Accuracy: {accuracy:.2f}%")

    return today_pred, tomorrow_pred, rmse, mae, r2, accuracy, preds







