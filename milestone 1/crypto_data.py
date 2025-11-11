# file: cryptodata_v2.py

import numpy as np
import pandas as pd
from pycoingecko import CoinGeckoAPI
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

# Initialize API
cg = CoinGeckoAPI()

# ------------------ MODEL BUILDERS ------------------ #
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_lstm_stack_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru_model(input_shape):
    from tensorflow.keras.layers import GRU
    model = Sequential([
        GRU(64, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru_stack_model(input_shape):
    from tensorflow.keras.layers import GRU
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        GRU(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_bidirectional_lstm_model(input_shape):
    from tensorflow.keras.layers import Bidirectional
    model = Sequential([
        Bidirectional(LSTM(64), input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_bidirectional_gru_model(input_shape):
    from tensorflow.keras.layers import GRU, Bidirectional
    model = Sequential([
        Bidirectional(GRU(64), input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

models_dict = {
    'lstm': build_lstm_model,
    'lstm_stack': build_lstm_stack_model,
    'gru': build_gru_model,
    'gru_stack': build_gru_stack_model,
    'bidirectional_lstm': build_bidirectional_lstm_model,
    'bidirectional_gru': build_bidirectional_gru_model,
}

# ------------------ DATA FUNCTIONS ------------------ #
def fetch_market_data(coin_id='bitcoin', vs_currency='usd', days=365):
    try:
        data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=vs_currency, days=days)
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
        volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])

        df = prices.merge(market_caps, on='timestamp').merge(volumes, on='timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"[ERROR] Failed to fetch data for {coin_id}: {e}")
        return pd.DataFrame()

def handle_missing_values(df):
    return df.ffill().bfill()

def remove_outliers(df):
    if df.empty:
        return df
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    condition = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
    return df[condition]

def normalize_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    return df_scaled, scaler

def create_sequences(data, window_size=24, target_col='price'):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data.iloc[i:i + window_size].values)
        y.append(data.iloc[i + window_size][target_col])
    return np.array(X), np.array(y)

def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return X[:train_end], y[:train_end], X[train_end:val_end], y[train_end:val_end], X[val_end:], y[val_end:]

def plot_predictions(y_test, y_pred, coin_id, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Price (scaled)')
    plt.plot(y_pred.flatten(), label='Predicted Price (scaled)')
    plt.title(f'{coin_id.capitalize()} Price Prediction - {model_name}')
    plt.xlabel('Time Steps')
    plt.ylabel('Scaled Price')
    plt.legend()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{coin_id}_{model_name}_prediction.png')
    plt.show()
    plt.close()

def train_and_save_model(X_train, y_train, X_val, y_val, X_test, y_test, coin_id, model_name, epochs=30, batch_size=64):
    try:
        model = models_dict[model_name](input_shape=(X_train.shape[1], X_train.shape[2]))
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )

        os.makedirs('saved_models', exist_ok=True)
        model.save(f'saved_models/{coin_id}_{model_name}.h5')

        y_pred = model.predict(X_test)
        plot_predictions(y_test, y_pred, coin_id, model_name)
        return model, history
    except Exception as e:
        print(f"[ERROR] Failed to train {model_name} for {coin_id}: {e}")
        return None, None

# ------------------ MAIN EXECUTION ------------------ #
if __name__ == "__main__":
    coins = ['bitcoin', 'ethereum', 'litecoin', 'ripple', 'cardano', 'polkadot']
    window_size = 24
    epochs = 30
    batch_size = 64

    for coin_id in coins:
        print(f"\n--- Processing {coin_id.upper()} ---")
        df = fetch_market_data(coin_id, 'usd', 365)  # 365 days here
        if df.empty:
            print(f"[SKIPPED] No data for {coin_id}")
            continue

        df = handle_missing_values(df)
        df_clean = remove_outliers(df)

        if df_clean.shape[0] < window_size + 10:
            print(f"[SKIPPED] Not enough data after outlier removal for {coin_id}")
            continue

        print(f"Original shape: {df.shape}")
        print(f"After removing outliers: {df_clean.shape}")

        df_scaled, scaler = normalize_data(df_clean)
        df_scaled.to_csv(f'cleaned_{coin_id}_data.csv')
        print(f"Saved cleaned_{coin_id}_data.csv")

        X, y = create_sequences(df_scaled, window_size=window_size, target_col='price')
        if len(X) == 0:
            print(f"[SKIPPED] Not enough data to create sequences for {coin_id}")
            continue

        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
        print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

        for model_name in models_dict.keys():
            print(f"Training model: {model_name} for {coin_id}")
            train_and_save_model(X_train, y_train, X_val, y_val, X_test, y_test,
                                 coin_id, model_name, epochs, batch_size)
