# 🚀 Quick Start Guide

Get up and running with the Stock Analysis System in 5 minutes!

## ⚡ 3-Step Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Analysis
```bash
python run_full_analysis.py
```

### Step 3: Enter Stock Details
```
Enter stock ticker: AAPL
Enter period: 2y
```

That's it! The system will:
- ✅ Fetch historical data
- ✅ Train AI model
- ✅ Predict next day price
- ✅ Analyze news sentiment
- ✅ Generate justification report

## 📊 Quick Example

```bash
# Analyze Apple stock with 2 years of data
python run_full_analysis.py
# Enter: AAPL
# Enter: 2y

# Wait 5-10 minutes for training...

# Results will be saved in:
# - AAPL_TIMESTAMP_prediction_results.json
# - AAPL_TIMESTAMP_justification.txt
# - AAPL_TIMESTAMP_sentiment_analysis.json
```

## 🎯 What You Get

### 1. Price Prediction
```
Current Price: $178.45
Predicted Price: $182.30
Expected Change: +$3.85 (+2.16%)
Recommendation: BUY 📈
Confidence: 87.3%
```

### 2. Sentiment Analysis
```
Overall Sentiment: POSITIVE
Sentiment Confidence: 73.5%
Articles Analyzed: 28
  • Positive: 18 articles
  • Negative: 4 articles
  • Neutral: 6 articles
```

### 3. Justification
```
Status: STRONG ALIGNMENT ✅
The positive market sentiment SUPPORTS the BUY recommendation.
Both technical indicators and news sentiment suggest a positive outlook.
```

## 🔑 Optional: Better News Coverage

For more news sources (recommended):

### Get Free API Keys:
1. **NewsAPI**: https://newsapi.org (2 minutes signup)
2. **Alpha Vantage**: https://www.alphavantage.co (2 minutes signup)

### Set API Keys:

**Windows PowerShell:**
```powershell
$env:NEWSAPI_KEY = "your_key_here"
$env:ALPHA_VANTAGE_KEY = "your_key_here"
python run_full_analysis.py
```

**Windows Command Prompt:**
```cmd
set NEWSAPI_KEY=your_key_here
set ALPHA_VANTAGE_KEY=your_key_here
python run_full_analysis.py
```

**Linux/Mac:**
```bash
export NEWSAPI_KEY="your_key_here"
export ALPHA_VANTAGE_KEY="your_key_here"
python run_full_analysis.py
```

**Or use config file:**
```bash
copy config_template.py config.py
# Edit config.py and add your keys
python run_full_analysis.py
```

## 📱 Popular Stock Tickers

Try analyzing these popular stocks:

### Tech Giants
- **AAPL** - Apple
- **MSFT** - Microsoft
- **GOOG** - Google (Alphabet)
- **AMZN** - Amazon
- **META** - Meta (Facebook)
- **TSLA** - Tesla
- **NVDA** - NVIDIA

### Blue Chips
- **JPM** - JPMorgan Chase
- **JNJ** - Johnson & Johnson
- **V** - Visa
- **WMT** - Walmart
- **PG** - Procter & Gamble

### Popular ETFs
- **SPY** - S&P 500 ETF
- **QQQ** - NASDAQ-100 ETF
- **DIA** - Dow Jones ETF
- **IWM** - Russell 2000 ETF

## ⏱️ Time Periods

Choose based on your needs:

| Period | Use Case | Training Time |
|--------|----------|---------------|
| **1mo** | Very short-term, limited data | 2-3 min |
| **3mo** | Short-term momentum | 3-5 min |
| **6mo** | Medium-term trends | 4-6 min |
| **1y** | Balanced analysis | 5-7 min |
| **2y** | ⭐ Recommended - Good balance | 7-10 min |
| **5y** | Long-term patterns | 10-15 min |
| **max** | All available data | 15-30 min |

## 🎨 Output Files Explained

After analysis, you'll see:

```
📁 Generated Files:

📊 Prediction Results:
   • AAPL_20250126_143022_prediction_results.json  ← Detailed data
   
📋 Prediction Summaries:
   • AAPL_20250126_143022_prediction_summary.txt  ← Human-readable
   
💡 Sentiment Justifications:
   • AAPL_20250126_143022_justification.txt  ← Why this prediction?
   
📰 Sentiment Analysis Data:
   • AAPL_20250126_143022_sentiment_analysis.json  ← News analysis
   
📈 Model Metrics:
   • AAPL_20250126_143022_metrics.csv  ← Performance stats
```

## 🎯 Reading the Results

### Recommendations

| Icon | Meaning | Action |
|------|---------|--------|
| 🚀 **STRONG BUY** | >2% increase predicted, high confidence | Strong positive signal |
| 📈 **BUY** | >0.5% increase predicted | Positive signal |
| ➡️ **HOLD** | Minimal change or low confidence | Wait for better signals |
| 📉 **SELL** | >0.5% decrease predicted | Negative signal |
| 🔻 **STRONG SELL** | >2% decrease predicted, high confidence | Strong negative signal |

### Alignment Status

| Status | Meaning | Risk |
|--------|---------|------|
| ✅ **STRONG ALIGNMENT** | Sentiment matches prediction | Lower risk |
| ⚠️ **PARTIAL ALIGNMENT** | Sentiment is neutral | Medium risk |
| ⚠️ **DIVERGENCE** | Sentiment contradicts prediction | Higher risk |

### Risk Levels

| Level | Volatility | Meaning |
|-------|------------|---------|
| 🟢 **LOW** | <25% | Relatively stable stock |
| 🟡 **MEDIUM** | 25-40% | Moderate volatility |
| 🔴 **HIGH** | >40% | Very volatile, risky |

## 🐛 Common Issues

### Issue: "Module not found"
```bash
pip install -r requirements.txt
```

### Issue: "No prediction results found"
Run prediction first:
```bash
python simp.py
```
Then run sentiment analysis:
```bash
python reason.py AAPL
```

### Issue: "No news articles found"
This is normal! The system works with:
- Yahoo Finance news (free, no API key needed)
- Optional: NewsAPI and Alpha Vantage (better coverage)

### Issue: Training is slow
- Use shorter period (1y instead of 2y)
- Normal: 5-10 minutes for 2y data
- GPU speeds up 5-10x (optional)

## 💡 Pro Tips

1. **Start with 2y period** - Best balance of data and speed
2. **Set API keys** - Get better news coverage
3. **Compare multiple stocks** - Use batch analysis
4. **Check justification file** - Most important insights
5. **Monitor alignment** - Divergence = higher risk
6. **Consider risk level** - High volatility = reduce position size

## 🔄 Typical Workflow

```
1. Run complete analysis
   → python run_full_analysis.py

2. Review justification file
   → Open STOCK_TIMESTAMP_justification.txt

3. Check alignment status
   → ✅ STRONG ALIGNMENT = More confident
   → ⚠️ DIVERGENCE = Be cautious

4. Make decision
   → Consider recommendation + sentiment + risk

5. Repeat for other stocks
   → Compare multiple opportunities
```

## 📚 Learn More

- **Full Documentation**: See `README.md`
- **Examples**: Run `python example_usage.py`
- **Troubleshooting**: Check `README.md` troubleshooting section
- **Configuration**: Copy `config_template.py` to `config.py`

## ⚠️ Important Reminder

**This is NOT financial advice!**

- ✅ Use as ONE tool among many
- ✅ Do your own research
- ✅ Consult financial advisors
- ✅ Never invest more than you can afford to lose
- ❌ Don't rely solely on AI predictions

Past performance ≠ Future results

## 🎉 You're Ready!

Now run your first analysis:

```bash
python run_full_analysis.py
```

Enter your favorite stock ticker and start analyzing!

Good luck! 📈🚀

---

**Need help?** Check `README.md` or run `python example_usage.py`

