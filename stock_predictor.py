import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import subprocess
import sys
import os
import time
import tensorflow as tf

# --- Configuration for Reproducibility ---
# Setting seeds helps stabilize training results across runs
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Time Series Data Preparation Function ---

def create_sequences(data, sequence_length):
    """
    Creates sequences of features (X) and corresponding target values (y).
    X is a sequence_length array of prices; y is the next day's price.
    """
    X, y = [], []
    # Loop up to the point where we can create one last sequence + target
    for i in range(len(data) - sequence_length):
        # Current sequence of sequence_length days (X)
        X.append(data[i:(i + sequence_length), 0])
        # Target value (y) is the price on the day immediately following the sequence
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

# ---------------------------------------------


# --- User Input and Data Fetching ---

print("Welcome to the LSTM Stock Price Predictor.")
stock_name = input("Enter stock ticker (e.g., AAPL, MSFT, GOOG): ").upper()
period = input("Enter the period (e.g., 1mo, 1y, 5y, max): ").lower()

# Call sample.py as subprocess, passing inputs
print("\nFetching live data using sample.py ...")

# Clear old data files if they exist before running the helper script
if os.path.exists("live_data.json"):
    os.remove("live_data.json")
if os.path.exists("live_data.csv"):
    os.remove("live_data.csv")

# Start the subprocess (running the content of the Canvas file)
process = subprocess.Popen(
    [sys.executable, "sample.py"], 
    stdin=subprocess.PIPE, 
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE, 
    text=True
)
# Send inputs to sample.py via stdin
try:
    # Need a small delay to ensure the subprocess is ready to read
    time.sleep(0.1)
    out, err = process.communicate(f"{stock_name}\n{period}\n")
    print(out)
    if err:
        print("Error from sample.py:\n", err)
except Exception as e:
    print(f"Communication error with subprocess: {e}")
    sys.exit(1)
finally:
    process.wait()


# Check if live_data.json exists and load data
if not os.path.exists("live_data.json"):
    print("live_data.json was not created or data fetching failed. Exiting.")
    sys.exit(1)

# Load the data
df = pd.read_json("live_data.json")
df.columns = [c.lower() for c in df.columns]

if 'name' not in df.columns:
    df['name'] = stock_name

if df.empty:
    print("DataFrame is empty after loading. Exiting.")
    sys.exit(1)
    
# Convert date column to datetime objects
df['date'] = pd.to_datetime(df['date'])

# --- Data Preparation for LSTM ---
# We only use the 'close' price for prediction
data = df[['close']].values
# Use MinMaxScaler to scale data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
sequence_length = 60 # Number of previous days to use for prediction (look-back window)

# Train/Test split: 80% for training, 20% for testing
training_data_len = int(np.ceil(len(scaled_data) * 0.8))
train_data = scaled_data[0:training_data_len, :]
# Include 'sequence_length' days from the end of the train set as look-back for the test set
test_data = scaled_data[training_data_len - sequence_length:, :] 

# Create sequences
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Reshape X for LSTM input: [samples, time_steps, features]
# X_train.shape[1] is sequence_length (60), 1 is the number of features (Close price)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Inverse transform actual prices (y_train/y_test) for accurate metric calculation
def inverse_transform_targets(y_scaled, scaler, original_shape):
    """Utility function to inverse transform the 1D target array."""
    y_scaled_reshaped = y_scaled.reshape(-1, 1)
    # Create a dummy array matching the input shape the scaler was fit on (1 column)
    dummy = np.zeros((y_scaled_reshaped.shape[0], original_shape[1]))
    dummy[:, 0] = y_scaled_reshaped.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

# Actual non-scaled target values
y_train_actual = inverse_transform_targets(y_train, scaler, scaled_data.shape)
y_test_actual = inverse_transform_targets(y_test, scaler, scaled_data.shape)

# Define the corresponding dataframes for plotting (only the target indices)
df_train = df.iloc[sequence_length : training_data_len].copy()
df_test = df.iloc[training_data_len : training_data_len + len(y_test)].copy()

if df_test.empty:
    print("Not enough data to create a test set. Exiting.")
    sys.exit(1)

# --- Build and Train LSTM Model ---

print("\nBuilding the LSTM model...")
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.3))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=25, return_sequences=False)) 
model.add(Dropout(0.3))
model.add(Dense(units=1)) 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# Increased patience for better chance of finding a good minimum
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

print("\nTraining the model (this may take a moment)...")
history = model.fit(
    X_train, y_train, 
    epochs=20, # Increased max epochs to give more training time
    batch_size=32, 
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1,
    shuffle=False  # Crucial for time series data
)

# --- Make Predictions and Inverse Transform ---

print("\nMaking predictions...")
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform predictions
test_predictions_scaled = inverse_transform_targets(test_predictions, scaler, scaled_data.shape)
train_predictions_scaled = inverse_transform_targets(train_predictions, scaler, scaled_data.shape)

# Prepare plot variables
test_predictions_plot = test_predictions_scaled
min_len = min(len(df_test), len(test_predictions_plot))
df_test = df_test.iloc[:min_len].copy() 
test_predictions_plot = test_predictions_plot[:min_len]

# Calculate moving averages for the test period
df_test['ma10'] = df_test['close'].rolling(window=10).mean()
df_test['ma20'] = df_test['close'].rolling(window=20).mean()

# --- Calculate Evaluation Metrics and Recommendation ---
# Metrics calculated only on the portion of actual data that has a prediction
test_actual_for_metrics = y_test_actual[:min_len]
train_mae = mean_absolute_error(y_train_actual, train_predictions_scaled)
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions_scaled))
test_mae = mean_absolute_error(test_actual_for_metrics, test_predictions_plot)
test_rmse = np.sqrt(mean_squared_error(test_actual_for_metrics, test_predictions_plot))

print(f"\nModel Performance:")
print(f"Training MAE: ${train_mae:.2f}")
print(f"Training RMSE: ${train_rmse:.2f}")
print(f"Test MAE: ${test_mae:.2f}")
print(f"Test RMSE: ${test_rmse:.2f}")

# Get the last prediction and actual price for recommendation
last_real_price = df_test['close'].iloc[-1]
last_predicted_price = test_predictions_plot[-1]
price_diff = last_predicted_price - last_real_price
percent_change = (price_diff / last_real_price) * 100

print("\n" + "="*50)
print(f"Stock: {stock_name}")
print(f"Last Real Closing Price: ${last_real_price:.2f}")
print(f"Predicted Next Closing Price: ${last_predicted_price:.2f}")
print(f"Predicted Change: {price_diff:+.2f} ({percent_change:+.2f}%)")

# Recommendation logic
if percent_change > 2:
    recommendation = "STRONG BUY (price expected to rise significantly)"
elif percent_change > 0.5:
    recommendation = "BUY (price expected to rise)"
elif percent_change < -2:
    recommendation = "STRONG SELL (price expected to fall significantly)"
elif percent_change < -0.5:
    recommendation = "SELL (price expected to fall)"
else:
    recommendation = "HOLD (no significant change expected)"

print(f"Recommendation: {recommendation}")
print("="*50)

# --- Plot Training History (Matplotlib) ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# --- Interactive Candlestick Chart with Predictions (Plotly) ---
print("\n--- Displaying Interactive Candlestick Chart with Predicted Line (Test Period) ---\n")

fig = go.Figure()

# Add Candlestick trace for actual prices
fig.add_trace(go.Candlestick(
    x=df_test['date'],
    open=df_test['open'],
    high=df_test['high'],
    low=df_test['low'],
    close=df_test['close'],
    name='Actual Price',
    increasing_line_color='green',
    decreasing_line_color='red'
))

# Add moving averages (MA10 and MA20)
fig.add_trace(go.Scatter(
    x=df_test['date'], 
    y=df_test['ma10'], 
    mode='lines', 
    line=dict(color='blue', width=1),
    name='10-day MA'
))

fig.add_trace(go.Scatter(
    x=df_test['date'], 
    y=df_test['ma20'], 
    mode='lines', 
    line=dict(color='orange', width=1),
    name='20-day MA'
))

# Add predicted line
fig.add_trace(go.Scatter(
    x=df_test['date'], 
    y=test_predictions_plot,
    mode='lines', 
    line=dict(color='purple', width=3, dash='dot'),
    name='LSTM Prediction'
))

fig.update_layout(
    title=f"{stock_name} Stock Price Analysis with LSTM Predictions<br>Test MAE: ${test_mae:.2f} | Test RMSE: ${test_rmse:.2f}",
    xaxis_title='Date',
    yaxis_title='Price ($)',
    xaxis_rangeslider_visible=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    height=600
)

fig.show()

# --- Line Plot Comparison (Matplotlib) ---
print("\n--- Displaying Line Graph of Real vs Predicted Prices (Matplotlib) ---\n")

plt.figure(figsize=(15, 8))

# 1. Plot actual vs predicted for test period
plt.subplot(2, 1, 1)
plt.plot(df_test['date'], df_test['close'], color='blue', label="Actual Price", linewidth=2)
plt.plot(df_test['date'], test_predictions_plot, color='red', label="Predicted Price", linewidth=2, linestyle='--')
plt.title(f"{stock_name} Stock Price Prediction - Test Period")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Plot prediction error
plt.subplot(2, 1, 2)
error = df_test['close'] - test_predictions_plot
plt.plot(df_test['date'], error, color='green', label="Prediction Error", linewidth=1)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.title("Prediction Error Over Time")
plt.xlabel("Date")
plt.ylabel("Error ($)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nModel successfully trained and evaluated for {stock_name}!")
print(f"The model achieved a test RMSE of ${test_rmse:.2f} and MAE of ${test_mae:.2f}")
