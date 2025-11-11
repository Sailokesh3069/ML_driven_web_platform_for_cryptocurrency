# file: data_preprocessing.py

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore")

# ---------- Step 1: Load preprocessed data ---------- #
def load_cleaned_data(coin_id):
    filename = f'cleaned_{coin_id}_data.csv'  # Should exist from LSTM pipeline
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found. Run LSTM pipeline first.")
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    return df

# ---------- Step 2: Feature Engineering (Rolling stats) ---------- #
def create_rolling_features(df, window_sizes=[3, 7, 14]):
    for window in window_sizes:
        df[f'roll_mean_{window}'] = df['price'].rolling(window=window).mean()
        df[f'roll_std_{window}'] = df['price'].rolling(window=window).std()
        df[f'roll_min_{window}'] = df['price'].rolling(window=window).min()
        df[f'roll_max_{window}'] = df['price'].rolling(window=window).max()
    return df.dropna()

# ---------- Step 3: Lag Features ---------- #
def create_lag_features(df, target_col='price', n_lags=5):
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    return df.dropna()

# ---------- Step 4: Train-Test Split ---------- #
def train_test_split(df, test_ratio=0.2):
    split = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split]
    test = df.iloc[split:]
    return train, test

# ---------- Step 5: Train Random Forest ---------- #
def train_model(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    print("Best Parameters:", grid.best_params_)
    return grid.best_estimator_

# ---------- Step 6: Evaluation ---------- #
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    return preds

# ---------- Step 7: Plotting ---------- #
def plot_predictions(y_test, preds, coin_id):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual')
    plt.plot(y_test.index, preds, label='Predicted')
    plt.title(f'{coin_id.capitalize()} - Random Forest Forecast')
    plt.xlabel('Time')
    plt.ylabel('Scaled Price')
    plt.legend()
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plot_path = f'plots/{coin_id}_rf_forecast.png'
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    print(f"Plot saved: {plot_path}")

# ---------- Step 8: Main Execution ---------- #
if __name__ == "__main__":
    coins = ['bitcoin', 'ethereum', 'litecoin', 'ripple', 'cardano', 'polkadot']

    for coin_id in coins:
        print(f"\n[PROCESSING] {coin_id.upper()}")

        try:
            df = load_cleaned_data(coin_id)
        except FileNotFoundError as e:
            print(f"[SKIPPED] {e}")
            continue

        df = create_rolling_features(df)
        df = create_lag_features(df)

        if df.empty or len(df) < 50:
            print(f"[SKIPPED] Not enough data after feature creation for {coin_id}")
            continue

        train, test = train_test_split(df)

        X_train = train.drop(columns=['price'])
        y_train = train['price']
        X_test = test.drop(columns=['price'])
        y_test = test['price']

        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        try:
            model = train_model(X_train, y_train)
            preds = evaluate(model, X_test, y_test)
            plot_predictions(y_test, preds, coin_id)

            # Save model
            os.makedirs('saved_models', exist_ok=True)
            model_path = f'saved_models/{coin_id}_rf_model.pkl'
            joblib.dump(model, model_path)
            print(f"Model saved: {model_path}")

        except Exception as e:
            print(f"[ERROR] Training failed for {coin_id}: {e}")
