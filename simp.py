# --- 1. SETUP AND IMPORTS ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.layers import (GRU, LSTM, Dense, Dropout, Input, Bidirectional, Conv1D, GlobalAveragePooling1D, Multiply, BatchNormalization, LayerNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import warnings
import sys
import json
import os
import glob
from datetime import datetime
warnings.filterwarnings('ignore')

# --- Configuration for Reproducibility ---
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- 2. PROFESSIONAL TRADER FEATURE ENGINEERING ---

def calculate_professional_indicators(df):
    """Calculate all professional trading indicators"""
    
    # === TECHNICAL INDICATORS ===
    
    # Moving Averages (Professional traders use multiple timeframes)
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # Moving Average Crossovers (Golden Cross, Death Cross)
    df['golden_cross'] = ((df['sma_50'] > df['sma_200']) & (df['sma_50'].shift(1) <= df['sma_200'].shift(1))).astype(int)
    df['death_cross'] = ((df['sma_50'] < df['sma_200']) & (df['sma_50'].shift(1) >= df['sma_200'].shift(1))).astype(int)
    
    # Price position relative to moving averages
    df['price_vs_sma20'] = (df['close'] / df['sma_20'] - 1) * 100
    df['price_vs_sma50'] = (df['close'] / df['sma_50'] - 1) * 100
    df['price_vs_sma200'] = (df['close'] / df['sma_200'] - 1) * 100
    
    # === MOMENTUM INDICATORS ===
    
    # RSI with multiple timeframes
    for period in [14, 21]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD with signal line
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14, min_periods=1).min()
    high_14 = df['high'].rolling(window=14, min_periods=1).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
    df['stoch_d'] = df['stoch_k'].rolling(window=3, min_periods=1).mean()
    
    # Williams %R
    df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14 + 1e-10)
    
    # === VOLATILITY INDICATORS ===
    
    # Bollinger Bands with multiple standard deviations
    for std_dev in [1, 2, 3]:
        bb_period = 20
        bb_sma = df['close'].rolling(window=bb_period, min_periods=1).mean()
        bb_std = df['close'].rolling(window=bb_period, min_periods=1).std()
        df[f'bb_upper_{std_dev}'] = bb_sma + (bb_std * std_dev)
        df[f'bb_lower_{std_dev}'] = bb_sma - (bb_std * std_dev)
        df[f'bb_width_{std_dev}'] = df[f'bb_upper_{std_dev}'] - df[f'bb_lower_{std_dev}']
        df[f'bb_position_{std_dev}'] = (df['close'] - df[f'bb_lower_{std_dev}']) / (df[f'bb_width_{std_dev}'] + 1e-10)
    
    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14, min_periods=1).mean()
    df['atr_percent'] = (df['atr'] / df['close']) * 100
    
    # === VOLUME INDICATORS ===
    
    # Volume Moving Averages
    df['volume_sma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)
    
    # On-Balance Volume (OBV)
    df['obv'] = (df['volume'] * np.sign(df['close'].diff())).cumsum()
    
    # Volume Price Trend (VPT)
    df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
    
    # === CHART PATTERN RECOGNITION ===
    
    # Support and Resistance levels
    df['resistance'] = df['high'].rolling(window=20, min_periods=1).max()
    df['support'] = df['low'].rolling(window=20, min_periods=1).min()
    df['resistance_distance'] = (df['resistance'] - df['close']) / df['close'] * 100
    df['support_distance'] = (df['close'] - df['support']) / df['close'] * 100
    
    # Trend strength
    df['trend_strength'] = df['close'].rolling(window=20, min_periods=1).apply(
        lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0
    )
    
    # === MARKET PSYCHOLOGY INDICATORS ===
    
    # Fear and Greed indicators
    df['price_momentum'] = df['close'].pct_change(5)  # 5-day momentum
    df['volatility_spike'] = (df['close'].rolling(window=5, min_periods=1).std() / 
                             df['close'].rolling(window=20, min_periods=1).std())
    
    # Gap analysis
    df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
    df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
    df['gap_size'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    
    # === QUANTITATIVE FEATURES ===
    
    # Price action patterns
    df['higher_high'] = ((df['high'] > df['high'].shift(1)) & 
                        (df['high'].shift(1) > df['high'].shift(2))).astype(int)
    df['lower_low'] = ((df['low'] < df['low'].shift(1)) & 
                      (df['low'].shift(1) < df['low'].shift(2))).astype(int)
    
    # Fibonacci levels (simplified)
    recent_high = df['high'].rolling(window=20, min_periods=1).max()
    recent_low = df['low'].rolling(window=20, min_periods=1).min()
    fib_range = recent_high - recent_low
    df['fib_23.6'] = recent_low + (fib_range * 0.236)
    df['fib_38.2'] = recent_low + (fib_range * 0.382)
    df['fib_61.8'] = recent_low + (fib_range * 0.618)
    
    # Distance to Fibonacci levels
    df['dist_to_fib_23.6'] = (df['close'] - df['fib_23.6']) / df['close'] * 100
    df['dist_to_fib_38.2'] = (df['close'] - df['fib_38.2']) / df['close'] * 100
    df['dist_to_fib_61.8'] = (df['close'] - df['fib_61.8']) / df['close'] * 100
    
    return df

def calculate_rsi(df, window=14):
    """Calculate RSI with proper handling of edge cases"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    # Handle division by zero
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)  # Neutral RSI for NaN values
    return df

def calculate_macd(df, fast=12, slow=26, signal=9):
    """MACD calculation with signal line"""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df

def add_moving_averages(df):
    """Add various moving averages"""
    for period in [5, 10, 20, 50]:
        df[f'ma{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
        df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    return df

def add_lagged_features(df, lags=[1, 2, 3, 5, 10]):
    """Add lagged features for multiple columns"""
    for lag in lags:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag) if 'returns' in df.columns else 0
    return df

def calculate_bollinger_bands(df, window=20, num_std_dev=2):
    """Bollinger Bands with position indicator"""
    sma = df['close'].rolling(window=window, min_periods=1).mean()
    std_dev = df['close'].rolling(window=window, min_periods=1).std()
    df['bb_upper'] = sma + (std_dev * num_std_dev)
    df['bb_lower'] = sma - (std_dev * num_std_dev)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_width'] + 1e-10)
    return df

def calculate_atr(df, window=14):
    """Average True Range for volatility"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=window, min_periods=1).mean()
    return df

def calculate_roc(df, periods=[5, 10, 20]):
    """Rate of Change for multiple periods"""
    for period in periods:
        df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / 
                               (df['close'].shift(period) + 1e-10)) * 100
    return df

def add_price_features(df):
    """Add price-based features"""
    # High-Low spread
    df['hl_spread'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
    # Close position in day's range
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    # Price changes
    df['price_change'] = df['close'].pct_change()
    df['price_change_abs'] = np.abs(df['price_change'])
    return df

def add_volume_features(df):
    """Add volume-based features"""
    df['volume_sma'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
    df['volume_change'] = df['volume'].pct_change()
    return df

def add_cyclical_features(df):
    """Cyclical encoding for time features"""
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month_of_year'] = df.index.month
    
    # Cyclical encoding
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
    df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month_of_year'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_of_year'] / 12)
    return df

def add_volatility_features(df, windows=[5, 10, 20]):
    """Volatility and statistical features"""
    df['returns'] = df['close'].pct_change()
    
    for window in windows:
        df[f'volatility_{window}'] = df['returns'].rolling(window=window, min_periods=1).std()
        df[f'skew_{window}'] = df['returns'].rolling(window=window, min_periods=3).skew()
        df[f'kurt_{window}'] = df['returns'].rolling(window=window, min_periods=4).kurt()
    
    return df

# --- 3. DATA PREPARATION AND UTILITIES ---

def cleanup_old_data():
    """Clean up old prediction files before starting new analysis"""
    file_patterns = ['*_prediction_results.json', '*_prediction_summary.txt', '*_metrics.csv']
    
    for pattern in file_patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                print(f"Cleaned up old file: {file}")
            except Exception as e:
                print(f"Could not remove {file}: {e}")

def save_prediction_data(stock_name, current_price, next_day_price, price_change, percent_change, 
                        recommendation, confidence, test_metrics, volatility, risk_level, 
                        train_metrics, val_metrics, data_info):
    """Save comprehensive prediction data to files"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save detailed JSON data
    json_data = {
        "timestamp": timestamp,
        "stock_name": stock_name,
        "prediction_date": datetime.now().isoformat(),
        "current_price": float(current_price),
        "predicted_price": float(next_day_price),
        "price_change": float(price_change),
        "percent_change": float(percent_change),
        "recommendation": recommendation,
        "confidence": float(confidence),
        "risk_level": risk_level,
        "annual_volatility": float(volatility),
        "data_info": data_info,
        "metrics": {
            "training": {k: float(v) for k, v in train_metrics.items()},
            "validation": {k: float(v) for k, v in val_metrics.items()},
            "test": {k: float(v) for k, v in test_metrics.items()}
        }
    }
    
    json_filename = f"{stock_name}_{timestamp}_prediction_results.json"
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # 2. Save summary text file
    summary_filename = f"{stock_name}_{timestamp}_prediction_summary.txt"
    with open(summary_filename, 'w') as f:
        f.write(f"STOCK PREDICTION ANALYSIS REPORT\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Stock: {stock_name}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Period: {data_info['period']}\n")
        f.write(f"Data Points: {data_info['total_days']}\n")
        f.write(f"Date Range: {data_info['start_date']} to {data_info['end_date']}\n\n")
        
        f.write(f"PRICE PREDICTION\n")
        f.write(f"-" * 20 + "\n")
        f.write(f"Current Price: ${current_price:.2f}\n")
        f.write(f"Predicted Price: ${next_day_price:.2f}\n")
        f.write(f"Expected Change: ${price_change:+.2f} ({percent_change:+.2f}%)\n")
        f.write(f"Recommendation: {recommendation}\n")
        f.write(f"Confidence: {confidence:.1f}%\n\n")
        
        f.write(f"RISK ASSESSMENT\n")
        f.write(f"-" * 20 + "\n")
        f.write(f"Annual Volatility: {volatility:.1f}%\n")
        f.write(f"Risk Level: {risk_level}\n\n")
        
        f.write(f"MODEL PERFORMANCE\n")
        f.write(f"-" * 20 + "\n")
        f.write(f"Test MAPE: {test_metrics['MAPE']:.2f}%\n")
        f.write(f"Test MAE: ${test_metrics['MAE']:.2f}\n")
        f.write(f"Test RMSE: ${test_metrics['RMSE']:.2f}\n")
        f.write(f"Directional Accuracy: {test_metrics['Directional Accuracy']:.1f}%\n")
    
    # 3. Save metrics CSV
    csv_filename = f"{stock_name}_{timestamp}_metrics.csv"
    with open(csv_filename, 'w') as f:
        f.write("Metric,Training,Validation,Test\n")
        for metric in train_metrics.keys():
            f.write(f"{metric},{train_metrics[metric]:.4f},{val_metrics[metric]:.4f},{test_metrics[metric]:.4f}\n")
    
    print(f"\nData saved to files:")
    print(f"   Detailed data: {json_filename}")
    print(f"   Summary report: {summary_filename}")
    print(f"   Metrics CSV: {csv_filename}")
    
    return json_filename, summary_filename, csv_filename

def create_sequences_with_future(data, sequence_length, predict_days=1):
    """
    Create sequences with proper future targets and improved data handling
    """
    X, y = [], []
    for i in range(len(data) - sequence_length - predict_days + 1):
        # Input sequence
        X.append(data[i:(i + sequence_length), :])
        # Target: future close price (predict_days ahead)
        y.append(data[i + sequence_length + predict_days - 1, 3])  # Column 3 is 'close'
    
    X, y = np.array(X), np.array(y)
    
    # Add data augmentation: slight noise to prevent overfitting
    if len(X) > 0:
        noise_factor = 0.001
        X_noise = X + np.random.normal(0, noise_factor, X.shape)
        X = np.concatenate([X, X_noise], axis=0)
        y = np.concatenate([y, y], axis=0)
    
    return X, y

def create_train_test_split(df, train_ratio=0.8, val_ratio=0.1):
    """Create proper train/val/test split"""
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    return train_df, val_df, test_df

def prepare_data_for_model(train_df, val_df, test_df, feature_cols, sequence_length=60):
    """Prepare and scale data with advanced preprocessing"""
    
    # Separate price features from other features for different scaling
    price_features = ['open', 'high', 'low', 'close', 'volume']
    technical_features = [col for col in feature_cols if col not in price_features]
    
    # Use different scalers for different feature types
    price_scaler = RobustScaler()
    technical_scaler = StandardScaler()
    
    # Prepare data
    train_data = train_df[feature_cols].values
    val_data = val_df[feature_cols].values
    test_data = test_df[feature_cols].values
    
    # Scale price features separately
    price_indices = [feature_cols.index(col) for col in price_features if col in feature_cols]
    technical_indices = [feature_cols.index(col) for col in technical_features if col in feature_cols]
    
    scaled_train = train_data.copy()
    scaled_val = val_data.copy()
    scaled_test = test_data.copy()
    
    if price_indices:
        price_scaler.fit(train_data[:, price_indices])
        scaled_train[:, price_indices] = price_scaler.transform(train_data[:, price_indices])
        scaled_val[:, price_indices] = price_scaler.transform(val_data[:, price_indices])
        scaled_test[:, price_indices] = price_scaler.transform(test_data[:, price_indices])
    
    if technical_indices:
        technical_scaler.fit(train_data[:, technical_indices])
        scaled_train[:, technical_indices] = technical_scaler.transform(train_data[:, technical_indices])
        scaled_val[:, technical_indices] = technical_scaler.transform(val_data[:, technical_indices])
        scaled_test[:, technical_indices] = technical_scaler.transform(test_data[:, technical_indices])
    
    # Create sequences
    X_train, y_train = create_sequences_with_future(scaled_train, sequence_length)
    X_val, y_val = create_sequences_with_future(scaled_val, sequence_length)
    X_test, y_test = create_sequences_with_future(scaled_test, sequence_length)
    
    # Create combined scaler for inverse transform
    class CombinedScaler:
        def __init__(self, price_scaler, technical_scaler, price_indices, technical_indices, feature_cols):
            self.price_scaler = price_scaler
            self.technical_scaler = technical_scaler
            self.price_indices = price_indices
            self.technical_indices = technical_indices
            self.feature_cols = feature_cols
            
        def transform(self, data):
            """Transform data using the appropriate scalers"""
            result = data.copy()
            if self.price_indices:
                result[:, self.price_indices] = self.price_scaler.transform(data[:, self.price_indices])
            if self.technical_indices:
                result[:, self.technical_indices] = self.technical_scaler.transform(data[:, self.technical_indices])
            return result
            
        def inverse_transform(self, data):
            """Inverse transform data using the appropriate scalers"""
            result = data.copy()
            if self.price_indices:
                result[:, self.price_indices] = self.price_scaler.inverse_transform(data[:, self.price_indices])
            if self.technical_indices:
                result[:, self.technical_indices] = self.technical_scaler.inverse_transform(data[:, self.technical_indices])
            return result
    
    scaler = CombinedScaler(price_scaler, technical_scaler, price_indices, technical_indices, feature_cols)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

def inverse_transform_predictions(scaled_predictions, scaler, feature_count, close_idx=3):
    """Properly inverse transform predictions"""
    dummy_array = np.zeros((len(scaled_predictions), feature_count))
    dummy_array[:, close_idx] = scaled_predictions.flatten()
    return scaler.inverse_transform(dummy_array)[:, close_idx]

# --- 4. MODEL ARCHITECTURE ---

def attention_block(inputs):
    """Attention mechanism with normalization"""
    # Calculate attention scores using GlobalAveragePooling1D for simplicity
    attention_probs = GlobalAveragePooling1D()(inputs)
    attention_probs = Dense(inputs.shape[-1], activation='softmax')(attention_probs)
    attention_probs = tf.keras.layers.Reshape((1, -1))(attention_probs)
    attention_mul = Multiply()([inputs, attention_probs])
    return attention_mul

def build_professional_trader_model(input_shape, learning_rate=0.0003):
    """Professional-grade model optimized for single-point future prediction"""
    inp = Input(shape=input_shape)
    
    # === SIMPLIFIED BUT POWERFUL ARCHITECTURE ===
    # Focus on price prediction accuracy rather than complexity
    
    # Primary LSTM branch for temporal patterns
    x = Bidirectional(LSTM(256, return_sequences=True))(inp)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Dense layers for final decision
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
    
    # === SINGLE PRICE PREDICTION OUTPUT ===
    price_prediction = Dense(1, activation='linear', name='price_prediction')(x)
    
    # === CONFIDENCE OUTPUT ===
    confidence = Dense(1, activation='sigmoid', name='confidence')(x)
    
    # Create model with multiple outputs
    model = Model(inputs=inp, outputs=[price_prediction, confidence])
    
    # === PROFESSIONAL OPTIMIZER ===
    optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        clipnorm=0.3  # Reduced gradient clipping
    )
    
    # === PROFESSIONAL LOSS FUNCTIONS ===
    def professional_loss(y_true, y_pred):
        """Custom loss function optimized for price prediction"""
        # Use MSE for better price accuracy
        mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        # Add small Huber component for robustness
        huber_loss = tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)
        return 0.8 * mse_loss + 0.2 * huber_loss
    
    def confidence_loss(y_true, y_pred):
        """Confidence loss for uncertainty estimation"""
        return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    
    # Compile with multiple outputs
    model.compile(
        optimizer=optimizer,
        loss={
            'price_prediction': professional_loss,
            'confidence': confidence_loss
        },
        loss_weights={'price_prediction': 1.0, 'confidence': 0.05},  # Reduced confidence weight
        metrics={
            'price_prediction': ['mae', 'mse'],
            'confidence': ['accuracy']
        }
    )
    
    return model

# --- 5. EVALUATION FUNCTIONS ---

def calculate_metrics(y_true, y_pred):
    """Calculate various evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE with protection against division by zero
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # Directional accuracy
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    directional_accuracy = np.mean(direction_true == direction_pred) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Directional Accuracy': directional_accuracy
    }

# --- 6. MAIN EXECUTION ---

if __name__ == "__main__":
    print("="*60)
    print("ENHANCED STOCK PRICE PREDICTOR v2.0")
    print("="*60)
    
    # Clean up old prediction files
    print("Cleaning up old prediction files...")
    cleanup_old_data()
    stock_name = input("Enter stock ticker (e.g., AAPL, MSFT, GOOG): ").upper()
    period = input("Enter the period (e.g., 2y, 5y, max): ").lower()
    
    print(f"\nFetching data for {stock_name}...")
    
    try:
        # Download data
        df = yf.download(stock_name, period=period, progress=True)
        if df.empty:
            raise ValueError("No data returned from yfinance.")
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)
    
    # Data preprocessing
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.columns = [c.lower() for c in df.columns]
    
    print(f"Data loaded: {len(df)} days of data")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    # Professional trader feature engineering
    print("\nEngineering professional trader features...")
    
    # Calculate all professional indicators
    df = calculate_professional_indicators(df)
    
    # Add additional features
    df = add_volatility_features(df)
    df = add_price_features(df)
    df = add_volume_features(df)
    df = add_cyclical_features(df)
    df = add_lagged_features(df)
    
    # Handle NaN values
    df = df.ffill().bfill()
    df = df.replace([np.inf, -np.inf], 0)
    
    # Remove any remaining NaN
    initial_len = len(df)
    df = df.dropna()
    print(f"Removed {initial_len - len(df)} rows with NaN values")
    
    if len(df) < 200:
        print("Not enough data after preprocessing. Try a longer period.")
        sys.exit(1)
    
    # Select features with correlation analysis
    base_features = ['open', 'high', 'low', 'close', 'volume']
    
    # Calculate correlation with future price for feature selection
    future_price = df['close'].shift(-1)
    correlations = {}
    
    for col in df.columns:
        if col not in base_features and col != 'close':
            try:
                corr = df[col].corr(future_price)
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
            except:
                continue
    
    # Select top correlated features
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    top_features = [feat[0] for feat in sorted_features[:25]]  # Top 25 features
    
    feature_cols = base_features + top_features
    
    # Keep only available features
    feature_cols = [col for col in feature_cols if col in df.columns]
    print(f"Using {len(feature_cols)} features")
    
    # Split data - adjust ratios for small datasets
    if len(df) < 100:
        train_df, val_df, test_df = create_train_test_split(df, train_ratio=0.8, val_ratio=0.1)
    else:
        train_df, val_df, test_df = create_train_test_split(df, train_ratio=0.7, val_ratio=0.15)
    
    print(f"\nData split:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Testing: {len(test_df)} samples")
    
    # Prepare sequences
    sequence_length = min(60, len(train_df) // 4)  # Adaptive sequence length
    print(f"  Sequence length: {sequence_length}")
    
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_data_for_model(
        train_df, val_df, test_df, feature_cols, sequence_length
    )
    
    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Build professional trader model
    print("\nBuilding professional trader model...")
    model = build_professional_trader_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        learning_rate=0.0003
    )
    
    print(f"Total parameters: {model.count_params():,}")
    
    # Advanced callbacks for better training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=8,
            min_lr=1e-8,
            verbose=1,
            cooldown=5
        ),
        ModelCheckpoint(
            f'{stock_name}_best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=0,
            save_weights_only=False
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 0.001 * (0.95 ** epoch),
            verbose=0
        )
    ]
    
    # Prepare targets for multi-output model
    # Create confidence targets (1 for correct direction, 0 for wrong)
    train_direction = np.diff(y_train) > 0
    val_direction = np.diff(y_val) > 0 if len(y_val) > 1 else np.array([])
    test_direction = np.diff(y_test) > 0 if len(y_test) > 1 else np.array([])
    
    # Pad with first value to match length
    train_conf = np.concatenate([[train_direction[0]], train_direction]).astype(float)
    val_conf = np.concatenate([[val_direction[0]], val_direction]).astype(float) if len(val_direction) > 0 else np.array([])
    test_conf = np.concatenate([[test_direction[0]], test_direction]).astype(float) if len(test_direction) > 0 else np.array([])
    
    # Train professional trader model
    print("\nTraining professional trader model...")
    
    # Handle validation data based on availability
    if len(X_val) > 0 and len(val_conf) > 0:
        validation_data = (X_val, [y_val, val_conf])
    else:
        validation_data = None
        print("Warning: No validation data available, training without validation")
    
    history = model.fit(
        X_train, [y_train, train_conf],
        validation_data=validation_data,
        epochs=200,
        batch_size=8,  # Even smaller batch for better convergence
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    # Load best model
    model = tf.keras.models.load_model(f'{stock_name}_best_model.h5', compile=False)
    
    # Recompile with proper loss functions
    def professional_loss(y_true, y_pred):
        return tf.keras.losses.Huber()(y_true, y_pred)
    
    def confidence_loss(y_true, y_pred):
        return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0003),
        loss={
            'price_prediction': professional_loss,
            'confidence': confidence_loss
        },
        loss_weights={'price_prediction': 1.0, 'confidence': 0.05},
        metrics={
            'price_prediction': ['mae', 'mse'],
            'confidence': ['accuracy']
        }
    )
    
    # Single-point predictions (no ensemble averaging)
    print("\nMaking single-point predictions...")
    
    # Get predictions for all sets
    train_pred_outputs = model.predict(X_train, verbose=0)
    
    # Extract price predictions (first output)
    train_pred_scaled = train_pred_outputs[0] if isinstance(train_pred_outputs, list) else train_pred_outputs
    
    # Inverse transform
    close_idx = feature_cols.index('close')
    train_pred = inverse_transform_predictions(train_pred_scaled, scaler, len(feature_cols), close_idx)
    y_train_actual = inverse_transform_predictions(y_train, scaler, len(feature_cols), close_idx)
    
    # Calculate metrics
    print("\nPerformance Metrics:")
    print("-" * 50)
    
    train_metrics = calculate_metrics(y_train_actual, train_pred)
    
    # Handle validation and test sets if available
    if len(X_val) > 0:
        val_pred_outputs = model.predict(X_val, verbose=0)
        val_pred_scaled = val_pred_outputs[0] if isinstance(val_pred_outputs, list) else val_pred_outputs
        val_pred = inverse_transform_predictions(val_pred_scaled, scaler, len(feature_cols), close_idx)
        y_val_actual = inverse_transform_predictions(y_val, scaler, len(feature_cols), close_idx)
        val_metrics = calculate_metrics(y_val_actual, val_pred)
    else:
        val_metrics = {'MAE': 0, 'RMSE': 0, 'MAPE': 0, 'Directional Accuracy': 0}
    
    if len(X_test) > 0:
        test_pred_outputs = model.predict(X_test, verbose=0)
        test_pred_scaled = test_pred_outputs[0] if isinstance(test_pred_outputs, list) else test_pred_outputs
        test_pred = inverse_transform_predictions(test_pred_scaled, scaler, len(feature_cols), close_idx)
        y_test_actual = inverse_transform_predictions(y_test, scaler, len(feature_cols), close_idx)
        test_metrics = calculate_metrics(y_test_actual, test_pred)
    else:
        # Use training metrics as test metrics if no test data
        test_metrics = train_metrics.copy()
    
    print("Training Set:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.2f}")
    
    print("\nValidation Set:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.2f}")
    
    print("\nTest Set:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.2f}")
    
    # Make single-point future prediction
    print("\nFuture Price Prediction:")
    print("-" * 50)
    
    # Get the last sequence from the entire dataset
    last_sequence = df[feature_cols].values[-sequence_length:]
    last_sequence_scaled = scaler.transform(last_sequence)
    last_sequence_scaled = last_sequence_scaled.reshape(1, sequence_length, len(feature_cols))
    
    # Single-point prediction (no ensemble averaging)
    next_day_outputs = model.predict(last_sequence_scaled, verbose=0)
    next_day_scaled = next_day_outputs[0] if isinstance(next_day_outputs, list) else next_day_outputs
    next_day_price = inverse_transform_predictions(next_day_scaled, scaler, len(feature_cols), close_idx)[0]
    
    # Professional trader adjustment based on market conditions
    current_price = df['close'].iloc[-1]
    
    # Advanced market condition analysis
    recent_volatility = df['returns'].tail(20).std()
    avg_volatility = df['returns'].std()
    
    # Trend analysis
    recent_trend = df['close'].pct_change().tail(5).mean()
    short_trend = df['close'].pct_change().tail(10).mean()
    long_trend = df['close'].pct_change().tail(20).mean()
    
    # Support/Resistance analysis
    resistance_level = df['high'].tail(20).max()
    support_level = df['low'].tail(20).min()
    current_position = (current_price - support_level) / (resistance_level - support_level)
    
    # Professional adjustment logic
    adjustment = 0
    
    # Volatility adjustment
    if recent_volatility > avg_volatility * 1.5:
        adjustment -= 0.015  # High volatility - conservative
    elif recent_volatility < avg_volatility * 0.5:
        adjustment += 0.01   # Low volatility - trend continuation
    
    # Trend adjustment
    if recent_trend > 0.01:  # Strong upward trend
        adjustment += 0.005
    elif recent_trend < -0.01:  # Strong downward trend
        adjustment -= 0.005
    
    # Position adjustment (mean reversion)
    if current_position > 0.8:  # Near resistance
        adjustment -= 0.01
    elif current_position < 0.2:  # Near support
        adjustment += 0.01
    
    # Apply professional adjustment
    next_day_price = next_day_price * (1 + adjustment)
    
    # Ensure prediction is reasonable (not more than 10% change)
    max_change = 0.10
    price_change_raw = next_day_price - current_price
    if abs(price_change_raw / current_price) > max_change:
        next_day_price = current_price * (1 + np.sign(price_change_raw) * max_change)
    price_change = next_day_price - current_price
    percent_change = (price_change / current_price) * 100
    
    print(f"Stock: {stock_name}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Predicted Next Day Price: ${next_day_price:.2f}")
    print(f"Expected Change: ${price_change:+.2f} ({percent_change:+.2f}%)")
    
    # Enhanced recommendation logic with confidence
    confidence = 100 - min(test_metrics['MAPE'], 50)  # Cap at 50% MAPE for confidence
    
    if percent_change > 2 and confidence > 60:
        recommendation = "STRONG BUY"
        emoji = "[STRONG BUY]"
    elif percent_change > 0.5 and confidence > 50:
        recommendation = "BUY"
        emoji = "[BUY]"
    elif percent_change < -2 and confidence > 60:
        recommendation = "STRONG SELL"
        emoji = "[STRONG SELL]"
    elif percent_change < -0.5 and confidence > 50:
        recommendation = "SELL"
        emoji = "[SELL]"
    else:
        recommendation = "HOLD"
        emoji = "[HOLD]"
    
    print(f"\n{emoji} Recommendation: {recommendation}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"Model Accuracy: {test_metrics['Directional Accuracy']:.1f}%")
    
    # Risk assessment
    volatility = df['returns'].std() * np.sqrt(252) * 100  # Annualized volatility
    print(f"\nRisk Assessment:")
    print(f"  Annual Volatility: {volatility:.1f}%")
    
    if volatility > 40:
        risk_level = "HIGH - This stock is very volatile"
        print(f"  Risk Level: {risk_level}")
    elif volatility > 25:
        risk_level = "MEDIUM - Moderate volatility"
        print(f"  Risk Level: {risk_level}")
    else:
        risk_level = "LOW - Relatively stable"
        print(f"  Risk Level: {risk_level}")
    
    # Prepare data info for saving
    data_info = {
        "period": period,
        "total_days": len(df),
        "start_date": str(df.index[0].date()),
        "end_date": str(df.index[-1].date()),
        "features_used": len(feature_cols),
        "sequence_length": sequence_length
    }
    
    # Save prediction data to files
    json_file, summary_file, csv_file = save_prediction_data(
        stock_name, current_price, next_day_price, price_change, percent_change,
        recommendation, confidence, test_metrics, volatility, risk_level,
        train_metrics, val_metrics, data_info
    )
    
    # Visualization
    print("\n Generating visualizations...")
    
    # Plot 1: Training History
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training history for multi-output model
    axes[0].plot(history.history['price_prediction_loss'], label='Training Price Loss', alpha=0.8)
    axes[0].plot(history.history['val_price_prediction_loss'], label='Validation Price Loss', alpha=0.8)
    axes[0].set_title('Price Prediction Loss During Training')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['price_prediction_mae'], label='Training MAE', alpha=0.8)
    axes[1].plot(history.history['val_price_prediction_mae'], label='Validation MAE', alpha=0.8)
    axes[1].set_title('Price Prediction MAE During Training')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Interactive Plotly Chart
    # Prepare data for plotting
    test_dates = test_df.index[sequence_length:]
    
    # Ensure all arrays have the same length
    min_len = min(len(test_dates), len(y_test_actual), len(test_pred))
    test_dates = test_dates[:min_len]
    y_test_plot = y_test_actual[:min_len]
    test_pred_plot = test_pred[:min_len]
    
    fig = go.Figure()
    
    # Add actual prices
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=y_test_plot,
        mode='lines',
        name='Actual Price',
        line=dict(color='blue', width=2)
    ))
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=test_pred_plot,
        mode='lines',
        name='Predicted Price',
        line=dict(color='red', width=2, dash='dot')
    ))
    
    # Add confidence band (simplified)
    error_margin = np.std(y_test_plot - test_pred_plot)
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=test_pred_plot + error_margin,
        mode='lines',
        line=dict(color='rgba(255,0,0,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=test_pred_plot - error_margin,
        mode='lines',
        line=dict(color='rgba(255,0,0,0)'),
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.1)',
        name='Confidence Band',
        hoverinfo='skip'
    ))
    
    # Add next day prediction
    fig.add_trace(go.Scatter(
        x=[df.index[-1] + pd.Timedelta(days=1)],
        y=[next_day_price],
        mode='markers',
        marker=dict(color='green', size=12, symbol='star'),
        name='Next Day Prediction'
    ))
    
    fig.update_layout(
        title=f"{stock_name} - Stock Price Prediction (Test MAPE: {test_metrics['MAPE']:.2f}%)",
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="Black",
            borderwidth=1
        ),
        template='plotly_white',
        height=600
    )
    
    fig.show()
    
    print("\n Analysis complete!")
    print("="*60)