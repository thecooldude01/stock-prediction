print("🚀 Starting ultra-advanced model training...")

# ✅ Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, Add, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_tuner import RandomSearch

print("✅ Libraries imported successfully.")

# 📌 Load stock data (Change ticker as needed)
ticker = "AAPL"  # Apple stock
print(f"📥 Fetching data for {ticker}...")
data = yf.download(ticker, start="2010-01-01", end="2024-01-01")

# Check if data is loaded
if data.empty:
    print("❌ Failed to load stock data! Check the ticker symbol or internet connection.")
    exit()

print("✅ Data loaded successfully.")

# 📊 Feature Engineering: Moving Averages, RSI, MACD
def compute_features(df):
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["MA_200"] = df["Close"].rolling(window=200).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["MACD"] = compute_macd(df["Close"])
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, slow=26, fast=12):
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    return ema_fast - ema_slow

data = compute_features(data)

# 🔄 Preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[["Close", "MA_50", "MA_200", "RSI", "MACD"]])

# 🔹 Create sequences
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)

print(f"✅ Data preprocessed. Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# 🏗️ Building a Transformer + LSTM Hybrid Model
def build_model(hp):
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

    # Transformer-like Attention Layer
    attn_output = Attention()([inputs, inputs])
    attn_output = LayerNormalization()(Add()([inputs, attn_output]))

    # LSTM Layer
    lstm_output = LSTM(hp.Int('lstm_units', min_value=64, max_value=256, step=32), return_sequences=False)(attn_output)
    dropout = Dropout(hp.Float('dropout_rate', 0.2, 0.5, step=0.1))(lstm_output)
    output = Dense(1)(dropout)

    model = Model(inputs, output)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# 🔍 Hyperparameter Tuning
tuner = RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=10,
    executions_per_trial=1,
    directory="tuner_results",
    project_name="stock_prediction"
)

print("🔍 Tuning hyperparameters...")
tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[EarlyStopping(monitor="val_loss", patience=5)])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# 🚀 Train the best model
print("🎯 Training the best model...")
history = best_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                         callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                                    ModelCheckpoint("best_transformer_lstm_model.h5", save_best_only=True)])

# 📈 Make Predictions
print("📊 Making predictions on test data...")
predicted_stock_prices = best_model.predict(X_test)
predicted_stock_prices = scaler.inverse_transform(np.hstack((predicted_stock_prices, np.zeros((predicted_stock_prices.shape[0], 4)))))[:, 0]
actual_stock_prices = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 4)))))[:, 0]

# 📉 Plot Actual vs Predicted Prices
print("📊 Plotting stock predictions...")
plt.figure(figsize=(14,7))
sns.set_style("darkgrid")
plt.plot(actual_stock_prices, label="Actual Prices", color="blue")
plt.plot(predicted_stock_prices, label="Predicted Prices", color="red")
plt.title(f"📈 {ticker} Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.show()

print("✅ Training complete. Model saved as `best_transformer_lstm_model.h5` 🎉")
