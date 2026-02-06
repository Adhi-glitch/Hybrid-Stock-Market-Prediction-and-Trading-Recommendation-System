# 📈 AI-Powered Stock Prediction & Sentiment Analysis System

A comprehensive stock analysis system that combines deep learning price predictions with FinBERT sentiment analysis to provide justified trading recommendations.

## 🌟 Features

- **🔮 Price Prediction**: Advanced LSTM/GRU neural network with 100+ technical indicators
- **📰 News Sentiment Analysis**: FinBERT-based analysis of financial news
- **💡 AI Justification**: Automated reasoning that explains predictions using market sentiment
- **📊 Multiple Data Sources**: Yahoo Finance, NewsAPI, Alpha Vantage
- **📈 Comprehensive Metrics**: MAPE, MAE, RMSE, Directional Accuracy
- **🎯 Risk Assessment**: Volatility analysis and risk level classification

## 📋 Components

### 1. `simp.py` - Stock Price Predictor
- Deep learning model with Bidirectional LSTM layers
- Professional trading indicators (RSI, MACD, Bollinger Bands, etc.)
- Fibonacci levels, support/resistance analysis
- Multi-output model with confidence estimation
- Automated model checkpointing and optimization

### 2. `sample.py` - Data Fetcher
- Fetches historical stock data from Yahoo Finance
- Handles retries and error recovery
- Saves data to JSON and CSV formats
- Validates data quality

### 3. `reason.py` - FinBERT Sentiment Analyzer
- Uses FinBERT (Financial BERT) for sentiment analysis
- Aggregates news from multiple sources
- Generates justification reports
- Aligns sentiment with predictions
- Provides risk-adjusted recommendations

### 4. `run_full_analysis.py` - Complete Pipeline
- Runs all components in sequence
- Automated workflow management
- Generates comprehensive reports

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages
- numpy >= 1.21.0
- pandas >= 1.3.0
- tensorflow >= 2.8.0
- torch >= 1.9.0
- transformers >= 4.20.0
- yfinance >= 0.1.70
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- plotly >= 5.0.0
- requests >= 2.27.0

## 📖 Usage

### Option 1: Complete Analysis (Recommended)

Run the complete pipeline for prediction + sentiment analysis:

```bash
python run_full_analysis.py
```

**Example:**
```
Enter stock ticker: AAPL
Enter period: 2y
```

This will:
1. ✅ Fetch historical data
2. ✅ Train prediction model
3. ✅ Generate price predictions
4. ✅ Fetch recent news
5. ✅ Analyze sentiment
6. ✅ Create justification report

### Option 2: Individual Components

#### Price Prediction Only
```bash
python simp.py
```

#### Sentiment Analysis Only (after running simp.py)
```bash
python reason.py AAPL
```

#### Data Fetching Only
```bash
echo -e "AAPL\n2y" | python sample.py
```

## 🔑 API Keys (Optional but Recommended)

For better news coverage, set these environment variables:

### Windows (PowerShell):
```powershell
$env:NEWSAPI_KEY = "your_key_here"
$env:ALPHA_VANTAGE_KEY = "your_key_here"
```

### Windows (Command Prompt):
```cmd
set NEWSAPI_KEY=your_key_here
set ALPHA_VANTAGE_KEY=your_key_here
```

### Linux/Mac:
```bash
export NEWSAPI_KEY="your_key_here"
export ALPHA_VANTAGE_KEY="your_key_here"
```

### Get Free API Keys:
- **NewsAPI**: https://newsapi.org (Free tier: 100 requests/day)
- **Alpha Vantage**: https://www.alphavantage.co (Free tier: 25 requests/day)

**Note:** The system works without API keys using Yahoo Finance news, but additional sources improve analysis quality.

## 📊 Output Files

After running the analysis, you'll get several files:

### Prediction Results
- `{STOCK}_{TIMESTAMP}_prediction_results.json` - Detailed prediction data
- `{STOCK}_{TIMESTAMP}_prediction_summary.txt` - Human-readable summary
- `{STOCK}_{TIMESTAMP}_metrics.csv` - Model performance metrics

### Sentiment Analysis
- `{STOCK}_{TIMESTAMP}_justification.txt` - Complete justification report
- `{STOCK}_{TIMESTAMP}_sentiment_analysis.json` - Detailed sentiment data

### Model Files
- `{STOCK}_best_model.h5` - Trained model (for future use)
- `live_data.json` / `live_data.csv` - Latest fetched data

## 📈 Example Output

```
=====================================================================
PREDICTION JUSTIFICATION REPORT FOR AAPL
=====================================================================

📊 PRICE PREDICTION SUMMARY
----------------------------------------------------------------------
Current Price: $178.45
Predicted Price: $182.30
Expected Change: +$3.85 (+2.16%)
Recommendation: BUY
Model Confidence: 87.3%

📰 NEWS SENTIMENT ANALYSIS
----------------------------------------------------------------------
Overall Sentiment: POSITIVE
Sentiment Confidence: 73.5%
Articles Analyzed: 28
  • Positive: 18 articles
  • Negative: 4 articles
  • Neutral: 6 articles

🎯 SENTIMENT-PREDICTION ALIGNMENT
----------------------------------------------------------------------
Status: STRONG ALIGNMENT ✅
The positive market sentiment SUPPORTS the BUY recommendation.
Both technical indicators and news sentiment suggest a positive outlook.

📌 KEY NEWS HIGHLIGHTS
----------------------------------------------------------------------
🟢 Most Positive News:
1. Apple reports record quarterly earnings
   Confidence: 94.2%
2. New iPhone sales exceed expectations
   Confidence: 89.7%
...
```

## 🎯 Understanding Recommendations

### Recommendation Levels
- **STRONG BUY** 🚀: >2% predicted increase, high confidence (>60%)
- **BUY** 📈: >0.5% predicted increase, moderate confidence (>50%)
- **HOLD** ➡️: Minimal change predicted or low confidence
- **SELL** 📉: >0.5% predicted decrease, moderate confidence (>50%)
- **STRONG SELL** 🔻: >2% predicted decrease, high confidence (>60%)

### Alignment Status
- **✅ STRONG ALIGNMENT**: Sentiment matches prediction direction
- **⚠️ PARTIAL ALIGNMENT**: Sentiment is neutral while prediction is directional
- **⚠️ DIVERGENCE**: Sentiment contradicts prediction - exercise caution

## ⚠️ Risk Levels

- **LOW**: Annual volatility < 25% (relatively stable)
- **MEDIUM**: Annual volatility 25-40% (moderate volatility)
- **HIGH**: Annual volatility > 40% (very volatile)

## 🧪 Model Architecture

### Neural Network Features:
- **Layers**: Bidirectional LSTM (256, 128, 64 units)
- **Regularization**: Layer Normalization, Dropout (0.2-0.3)
- **Optimization**: Adam optimizer with learning rate scheduling
- **Loss Function**: Custom hybrid (80% MSE + 20% Huber)
- **Callbacks**: Early stopping, ReduceLROnPlateau, ModelCheckpoint

### Technical Indicators (100+):
- Moving Averages (SMA, EMA: 5, 10, 20, 50, 100, 200 periods)
- Momentum: RSI, MACD, Stochastic Oscillator, Williams %R
- Volatility: Bollinger Bands, ATR
- Volume: OBV, VPT, Volume Ratios
- Chart Patterns: Support/Resistance, Fibonacci levels
- Market Psychology: Gap analysis, trend strength

## 🔬 FinBERT Sentiment Analysis

### Model Details:
- **Base Model**: ProsusAI/finbert
- **Training**: Fine-tuned on financial texts
- **Output**: Positive, Negative, Neutral probabilities
- **Confidence**: Based on probability distribution

### News Sources:
1. **Yahoo Finance** (Free, no API key required)
2. **NewsAPI** (Optional, requires API key)
3. **Alpha Vantage** (Optional, requires API key)

## 📊 Performance Metrics

The system evaluates models using:
- **MAE** (Mean Absolute Error): Average prediction error in dollars
- **RMSE** (Root Mean Squared Error): Penalizes larger errors
- **MAPE** (Mean Absolute Percentage Error): Error as percentage
- **Directional Accuracy**: % of correct up/down predictions

## 🛡️ Disclaimer

**IMPORTANT**: This tool is for educational and research purposes only.

- ❌ NOT financial advice
- ❌ NOT a guarantee of future performance
- ✅ Use as one of many factors in investment decisions
- ✅ Always do your own research
- ✅ Consult with licensed financial advisors
- ✅ Never invest more than you can afford to lose

Past performance does not guarantee future results. Stock markets are inherently risky.

## 🐛 Troubleshooting

### Issue: "No prediction results found"
**Solution**: Run `simp.py` first to generate predictions before running `reason.py`

### Issue: "No news articles found"
**Solution**: 
1. Check internet connection
2. Set API keys for NewsAPI and Alpha Vantage
3. Yahoo Finance news may have limited coverage for some stocks

### Issue: "Module not found"
**Solution**: Install requirements: `pip install -r requirements.txt`

### Issue: "CUDA/GPU errors"
**Solution**: TensorFlow will automatically use CPU. For GPU support, install `tensorflow-gpu`

### Issue: "Model training is slow"
**Solution**: 
1. Reduce sequence length (default: 60)
2. Use smaller period (e.g., 1y instead of 5y)
3. Reduce batch size in simp.py

## 🔄 Update History

### Version 2.0 (Current)
- ✅ Added FinBERT sentiment analysis
- ✅ Multi-source news aggregation
- ✅ Automated justification generation
- ✅ Complete analysis pipeline
- ✅ Enhanced risk assessment

### Version 1.0
- ✅ Initial LSTM prediction model
- ✅ Technical indicator engineering
- ✅ Basic visualization

## 📝 Contributing

This is a research/educational project. Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Share results (anonymously)

## 📧 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the example outputs
3. Verify all dependencies are installed
4. Check API key configuration (if using external news sources)

## 📜 License

This project is provided as-is for educational purposes.

---

**Happy Trading! 📈🚀**

*Remember: This is a tool to assist analysis, not replace human judgment and professional advice.*

