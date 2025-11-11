
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

# LOAD DATA

def load_data(ticker, period="5y"):
    df = yf.download(ticker, period=period, interval="1d")
    df.dropna(inplace=True)
    df["Return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df.dropna(inplace=True)
    return df


# PREPARE FEATURES

def prepare_features(df):
    features = ["Open", "High", "Low", "Volume", "MA5", "MA10", "MA20", "Return"]
    X = df[features]
    y = df["Close"]
    return X, y

# TRAIN RANDOM FOREST MODEL WITH GRID SEARCH

def train_rf_model(df, test_size_days=365):
    X, y = prepare_features(df)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test = X_scaled[:-test_size_days], X_scaled[-test_size_days:]
    y_train, y_test = y[:-test_size_days], y[-test_size_days:]

    # Updated grid 
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': [1.0, 'log2']  
    }

    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

    # Flatten y to avoid DataConversionWarning
    y_train = np.ravel(y_train)
    grid.fit(X_train, y_train)

    print("Best RF Params:", grid.best_params_)
    best_rf = grid.best_estimator_


    # EVALUATE MODEL
    y_pred = best_rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = r2 * 100

    print(f"\nModel Performance:")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R²:   {r2:.3f}")
    print(f"Accuracy: {accuracy:.2f}%")

    plt.figure(figsize=(10,5))
    plt.plot(y_test.values, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted (Random Forest)', linestyle='dashed')
    plt.title(f"{ticker} — Actual vs Predicted Closing Price (Random Forest)")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


    

    return best_rf, rmse, mae, r2, accuracy

if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g., AAPL, TSLA, INFY.NS): ").strip().upper()
    df = load_data(ticker)
    model, rmse, mae, r2, accuracy = train_rf_model(df)
