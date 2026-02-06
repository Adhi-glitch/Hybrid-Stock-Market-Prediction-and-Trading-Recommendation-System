"""
Configuration Template for Stock Analysis System

Copy this file to 'config.py' and fill in your API keys.
The system will automatically use these keys if config.py exists.

IMPORTANT: Never commit config.py to version control!
Add it to .gitignore to keep your keys secure.
"""

# =============================================================================
# NEWS API KEYS (Optional but recommended for better news coverage)
# =============================================================================

# NewsAPI - Get free key at: https://newsapi.org
# Free tier: 100 requests per day
NEWSAPI_KEY = "aeadd16edf8947f2b765f81947cf67b5"  # Enter your key here: "your_newsapi_key_here"

# Alpha Vantage - Get free key at: https://www.alphavantage.co
# Free tier: 25 requests per day
ALPHA_VANTAGE_KEY = "Q1HA9DH8YMGZF680"  # Enter your key here: "your_alphavantage_key_here"

# Finnhub - Get free key at: https://finnhub.io
# Free tier: 60 API calls per minute
FINNHUB_KEY = "d3v3lk1r01qt2ctntoc0d3v3lk1r01qt2ctntocg"  # Enter your key here: "your_finnhub_key_here"

# MarketAux - Get free key at: https://www.marketaux.com
# Free tier: 100 requests per day
MARKETAUX_KEY = "RqDtbZ3cqxMcWdqlFtiK3i97wF2KLsTl1BWHN142"  # Enter your key here: "your_marketaux_key_here"

# =============================================================================
# MODEL CONFIGURATION (Advanced users)
# =============================================================================

# Model training parameters
MODEL_CONFIG = {
    'sequence_length': 60,        # Number of days to look back
    'learning_rate': 0.0003,      # Initial learning rate
    'batch_size': 8,              # Batch size for training
    'epochs': 200,                # Maximum epochs (early stopping will kick in)
    'patience': 20,               # Early stopping patience
}

# Feature selection
FEATURE_CONFIG = {
    'use_all_features': True,     # Use all available features
    'top_n_features': 25,         # Number of top features to select
    'min_correlation': 0.01,      # Minimum correlation with target
}

# Risk thresholds
RISK_CONFIG = {
    'low_volatility_threshold': 25,    # Below this = LOW risk
    'high_volatility_threshold': 40,   # Above this = HIGH risk
    'max_prediction_change': 0.10,     # Cap predictions at 10% change
}

# Recommendation thresholds
RECOMMENDATION_CONFIG = {
    'strong_buy_threshold': 2.0,       # % change for STRONG BUY
    'buy_threshold': 0.5,              # % change for BUY
    'strong_sell_threshold': -2.0,     # % change for STRONG SELL
    'sell_threshold': -0.5,            # % change for SELL
    'min_confidence': 50.0,            # Minimum confidence for non-HOLD
}

# =============================================================================
# SENTIMENT ANALYSIS CONFIGURATION
# =============================================================================

SENTIMENT_CONFIG = {
    'max_articles': 30,                # Maximum articles to analyze
    'news_lookback_days': 7,          # Days of news to fetch
    'min_sentiment_confidence': 0.5,   # Minimum confidence for sentiment
}

# =============================================================================
# DATA FETCHING CONFIGURATION
# =============================================================================

DATA_CONFIG = {
    'default_period': '2y',           # Default data period
    'retry_attempts': 3,              # Number of retry attempts
    'retry_delay': 10,                # Seconds between retries
}

# =============================================================================
# INSTRUCTIONS
# =============================================================================

"""
HOW TO USE THIS FILE:

1. Copy this file to 'config.py':
   - Windows: copy config_template.py config.py
   - Linux/Mac: cp config_template.py config.py

2. Edit config.py and fill in your API keys:
   NEWSAPI_KEY = "your_actual_key_here"
   ALPHA_VANTAGE_KEY = "your_actual_key_here"

3. (Optional) Adjust other parameters if you want custom behavior

4. Add config.py to .gitignore:
   echo "config.py" >> .gitignore

5. Run the system normally - it will automatically use your config.py

SECURITY NOTE:
- Never share config.py with others
- Never commit config.py to GitHub
- Keep your API keys secret

ALTERNATIVE METHODS:
You can also set API keys via environment variables:
- Windows PowerShell: $env:NEWSAPI_KEY = "your_key"
- Windows CMD: set NEWSAPI_KEY=your_key
- Linux/Mac: export NEWSAPI_KEY="your_key"
"""

# =============================================================================
# ADVANCED: Custom Feature Engineering
# =============================================================================

"""
If you want to add custom technical indicators, modify simp.py's
calculate_professional_indicators() function.

Example custom indicators you might add:
- Ichimoku Cloud
- Parabolic SAR
- Commodity Channel Index (CCI)
- Money Flow Index (MFI)
- Elder's Force Index
- Chaikin Oscillator
"""

# =============================================================================
# PERFORMANCE TUNING TIPS
# =============================================================================

"""
For faster training (at cost of accuracy):
- Reduce sequence_length to 30
- Reduce epochs to 100
- Increase batch_size to 16
- Use shorter data period (1y instead of 2y)

For better accuracy (slower training):
- Increase sequence_length to 90
- Keep epochs at 200
- Keep batch_size at 8
- Use longer data period (5y or more)
- Increase top_n_features to 40

For GPU acceleration:
- Install: pip install tensorflow-gpu
- Ensure CUDA and cuDNN are installed
- Training will be 5-10x faster on GPU
"""

