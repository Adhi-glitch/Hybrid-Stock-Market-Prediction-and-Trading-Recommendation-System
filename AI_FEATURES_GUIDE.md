# 🤖 AI-Powered Features Guide

## Enhanced Sentiment Analysis with NLP Seq2Seq Models

This guide explains the new AI-powered features that generate natural language explanations for stock predictions.

---

## 🌟 New Features

### 1. **Multi-Source News Aggregation**
Fetch news from 5 different API sources for comprehensive coverage:

| Source | Type | Free Tier | Signup |
|--------|------|-----------|--------|
| **Yahoo Finance** | Free | Unlimited | No signup needed ✅ |
| **NewsAPI** | Optional | 100 req/day | https://newsapi.org |
| **Alpha Vantage** | Optional | 25 req/day | https://www.alphavantage.co |
| **Finnhub** | Optional | 60 req/min | https://finnhub.io |
| **MarketAux** | Optional | 100 req/day | https://www.marketaux.com |

### 2. **AI Text Generation (Seq2Seq)**
Uses **Google FLAN-T5** model to generate:
- 📝 Natural language sentiment summaries
- 💡 Detailed recommendation explanations
- ⚠️ Risk analysis narratives
- 🎯 Actionable insights

### 3. **FinBERT Sentiment Analysis**
- Specialized BERT model trained on financial texts
- Analyzes positive/negative/neutral sentiment
- Provides confidence scores for each article

---

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- `torch` - PyTorch for AI models
- `transformers` - Hugging Face library for FinBERT & T5
- `sentencepiece` - Required for T5 tokenization

### Step 2: Run Analysis
```bash
python run_full_analysis.py
```

The system will:
1. ✅ Fetch news from available sources
2. ✅ Analyze sentiment with FinBERT
3. ✅ Generate AI-powered explanations with T5
4. ✅ Create comprehensive justification report

---

## 🔑 API Keys Setup

### Option 1: Config File (Recommended)

```bash
# Copy template
copy config_template.py config.py

# Edit config.py and add your keys:
NEWSAPI_KEY = "your_newsapi_key_here"
ALPHA_VANTAGE_KEY = "your_alphavantage_key_here"
FINNHUB_KEY = "your_finnhub_key_here"
MARKETAUX_KEY = "your_marketaux_key_here"
```

### Option 2: Environment Variables

**Windows PowerShell:**
```powershell
$env:NEWSAPI_KEY = "your_key"
$env:ALPHA_VANTAGE_KEY = "your_key"
$env:FINNHUB_KEY = "your_key"
$env:MARKETAUX_KEY = "your_key"
```

**Windows CMD:**
```cmd
set NEWSAPI_KEY=your_key
set ALPHA_VANTAGE_KEY=your_key
set FINNHUB_KEY=your_key
set MARKETAUX_KEY=your_key
```

**Linux/Mac:**
```bash
export NEWSAPI_KEY="your_key"
export ALPHA_VANTAGE_KEY="your_key"
export FINNHUB_KEY="your_key"
export MARKETAUX_KEY="your_key"
```

### Get Free API Keys:

#### NewsAPI (Recommended)
1. Visit: https://newsapi.org
2. Sign up (2 minutes)
3. Copy API key
4. Free tier: 100 requests/day

#### Alpha Vantage (Recommended)
1. Visit: https://www.alphavantage.co/support/#api-key
2. Enter email
3. Instant API key
4. Free tier: 25 requests/day

#### Finnhub (Optional)
1. Visit: https://finnhub.io/register
2. Sign up (2 minutes)
3. Copy API key from dashboard
4. Free tier: 60 calls/minute

#### MarketAux (Optional)
1. Visit: https://www.marketaux.com/account/signup
2. Sign up (2 minutes)
3. Copy API key
4. Free tier: 100 requests/day

**Note:** System works without API keys using Yahoo Finance!

---

## 🤖 How AI Text Generation Works

### Architecture

```
News Articles → FinBERT → Sentiment Scores
                              ↓
                    [Context Building]
                              ↓
                         FLAN-T5 Model
                              ↓
                  Natural Language Explanation
```

### Example Flow

**Input to T5:**
```
Summarize the market sentiment based on these news headlines:
Apple reports record quarterly earnings | New iPhone sales exceed expectations | ...

Sentiment scores: 18 positive, 4 negative, 6 neutral articles.
Overall sentiment: positive.

Generate a professional 2-3 sentence summary explaining the market sentiment:
```

**T5 Output:**
```
Market sentiment is predominantly positive with strong earnings reports 
and robust iPhone sales driving optimism. The 18 positive articles 
significantly outweigh the 4 negative ones, indicating broad market 
confidence in the company's performance. Recent developments suggest 
favorable conditions for continued growth.
```

### Three Types of AI-Generated Content

#### 1. Sentiment Summary
- Analyzes overall market mood
- Synthesizes key themes from news
- Professional 2-3 sentence summary

#### 2. Recommendation Explanation  
- Explains WHY the recommendation makes sense
- Addresses technical-sentiment alignment
- Mentions risks and opportunities
- Provides actionable insights
- Professional 3-4 sentence explanation

#### 3. Risk Analysis
- Analyzes volatility and risk factors
- Checks for signal divergence
- Recommends risk management approach
- Professional 2-3 sentence analysis

---

## 📊 Sample Output

### Before (Without AI Generation):
```
The BUY recommendation is based on technical indicators.
Market sentiment is positive.
Consider the risks.
```

### After (With AI Generation):
```
🤖 AI-GENERATED SENTIMENT SUMMARY
----------------------------------------------------------------------
Market sentiment is predominantly positive (73.5% confidence) with 
strong earnings reports and robust iPhone sales driving optimism. 
The 18 positive articles significantly outweigh the 4 negative ones, 
indicating broad market confidence in the company's performance.

💡 AI-GENERATED DETAILED EXPLANATION
----------------------------------------------------------------------
The BUY recommendation is strongly supported by both technical analysis 
predicting a 2.16% upward movement and positive market sentiment from 
recent news. Key factors include record quarterly earnings and exceeding 
iPhone sales expectations, which align with the bullish technical signals. 
However, investors should monitor supply chain concerns mentioned in 
recent reports. The strong alignment between technical and sentiment 
indicators suggests this is a high-confidence trade opportunity, though 
moderate position sizing is recommended given the 32.4% annual volatility.

⚠️ AI-GENERATED RISK ASSESSMENT
----------------------------------------------------------------------
With 32.4% annual volatility, this stock carries moderate risk typical 
of large-cap tech stocks. The alignment between technical signals and 
market sentiment reduces uncertainty, though normal portfolio risk 
management practices should still apply. Consider using stop-losses 
at 5-7% below entry to manage downside risk effectively.
```

---

## ⚙️ Technical Details

### Models Used

#### FinBERT (Sentiment Analysis)
- **Model:** ProsusAI/finbert
- **Purpose:** Analyze financial text sentiment
- **Input:** News headlines and descriptions
- **Output:** Positive/Negative/Neutral probabilities
- **Size:** ~440MB
- **Speed:** ~1-2 seconds per article

#### FLAN-T5 Base (Text Generation)
- **Model:** google/flan-t5-base
- **Purpose:** Generate natural language explanations
- **Input:** Structured prompts with context
- **Output:** Professional financial narratives
- **Size:** ~990MB
- **Speed:** ~3-5 seconds per generation

### Hardware Requirements

**Minimum:**
- CPU: Any modern processor
- RAM: 8GB
- Disk: 2GB free space
- GPU: Not required (runs on CPU)

**Recommended:**
- CPU: Multi-core processor
- RAM: 16GB
- Disk: 5GB free space
- GPU: NVIDIA GPU with 4GB+ VRAM (10x faster)

### GPU Acceleration

If you have an NVIDIA GPU:

```bash
# Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

The system automatically uses GPU if available!

---

## 🎨 Customization

### Change Text Generation Model

Edit `reason.py`:

```python
# Use larger model for better quality (slower)
self.text_generator = TextGenerator(model_name="google/flan-t5-large")

# Or use smaller model for speed (lower quality)
self.text_generator = TextGenerator(model_name="google/flan-t5-small")
```

**Model Comparison:**

| Model | Size | Speed | Quality | Memory |
|-------|------|-------|---------|--------|
| flan-t5-small | 300MB | Fast | Good | 4GB RAM |
| flan-t5-base | 990MB | Medium | Better | 8GB RAM |
| flan-t5-large | 3GB | Slow | Best | 16GB RAM |

### Adjust Generation Parameters

In `reason.py`, modify the `generate()` call:

```python
outputs = self.model.generate(
    inputs.input_ids,
    max_length=200,        # Increase for longer text
    num_beams=5,           # More beams = better quality, slower
    temperature=0.7,       # Higher = more creative, lower = more conservative
    top_p=0.9,            # Nucleus sampling threshold
    do_sample=True,       # Enable sampling for diversity
    repetition_penalty=1.2 # Penalize repetitions
)
```

---

## 🔍 Troubleshooting

### Issue: "Model download is slow"
**Solution:** First download takes 5-10 minutes for models (~1.5GB total). Subsequent runs are fast as models are cached.

### Issue: "Out of memory error"
**Solution:** 
```python
# Use smaller model
TextGenerator(model_name="google/flan-t5-small")

# Or reduce batch processing
# Process fewer news articles at once
```

### Issue: "AI generation fails"
**Solution:** System automatically falls back to rule-based generation. Check error messages for details.

### Issue: "No news articles found"
**Solution:**
1. Check internet connection
2. Set up API keys for more sources
3. Yahoo Finance may have limited coverage for some stocks

### Issue: "Generated text is repetitive"
**Solution:** Increase `repetition_penalty` parameter (e.g., 1.5)

### Issue: "Generated text is too short"
**Solution:** Increase `max_length` parameter (e.g., 250)

---

## 📈 Performance Tips

### For Faster Analysis:
1. **Use smaller model:** `flan-t5-small`
2. **Enable GPU:** Install CUDA-enabled PyTorch
3. **Reduce news articles:** Limit to top 20 articles
4. **Cache models:** Models auto-cache after first download

### For Better Quality:
1. **Use larger model:** `flan-t5-large`
2. **More API sources:** Set up all API keys
3. **Increase beam search:** `num_beams=7`
4. **Longer outputs:** `max_length=300`

---

## 🆚 Comparison: Before vs After

### Without AI Features:
- ✅ Price prediction
- ✅ Basic sentiment (positive/negative/neutral)
- ✅ Simple rule-based explanations
- ❌ No detailed reasoning
- ❌ Generic recommendations

### With AI Features:
- ✅ Price prediction
- ✅ Advanced sentiment analysis (FinBERT)
- ✅ AI-generated explanations
- ✅ Detailed reasoning and context
- ✅ Personalized, specific recommendations
- ✅ Professional financial narratives
- ✅ Multi-source news aggregation

---

## 💡 Best Practices

### 1. API Key Management
- ✅ Use `config.py` for permanent setup
- ✅ Never commit `config.py` to git
- ✅ Start with NewsAPI and Alpha Vantage
- ✅ Add more sources as needed

### 2. Model Selection
- ✅ Start with `flan-t5-base` (good balance)
- ✅ Upgrade to `flan-t5-large` for better quality
- ✅ Use `flan-t5-small` for quick tests

### 3. News Coverage
- ✅ Set up at least 2 API sources
- ✅ More sources = better sentiment analysis
- ✅ Check API rate limits

### 4. Interpretation
- ✅ AI explanations provide context, not absolute truth
- ✅ Combine AI insights with your own research
- ✅ Check alignment between technical and sentiment
- ✅ Consider risk assessments seriously

---

## 🔬 Advanced Usage

### Custom Prompts

Modify prompts in `reason.py` for different styles:

**Conservative Style:**
```python
prompt = f"""As a conservative financial analyst, explain this 
recommendation with emphasis on risk management..."""
```

**Aggressive Style:**
```python
prompt = f"""As a growth-focused analyst, explain this 
recommendation with emphasis on opportunities..."""
```

### Multi-Stock Analysis

```python
from reason import SentimentJustifier

justifier = SentimentJustifier()

stocks = ['AAPL', 'GOOG', 'MSFT']
for stock in stocks:
    justifier.analyze_stock(stock)
```

---

## 📚 Further Reading

- **FinBERT Paper:** https://arxiv.org/abs/1908.10063
- **T5 Paper:** https://arxiv.org/abs/1910.10683
- **FLAN Paper:** https://arxiv.org/abs/2210.11416
- **Transformers Docs:** https://huggingface.co/docs/transformers

---

## ⚠️ Disclaimer

AI-generated content is based on:
- Historical news articles
- Sentiment patterns
- Statistical models

**Remember:**
- ❌ Not financial advice
- ❌ No guarantee of accuracy
- ✅ Use as ONE tool among many
- ✅ Do your own research
- ✅ Consult licensed professionals

---

## 🎯 Summary

You now have:
- 🤖 AI-powered text generation
- 📰 Multi-source news aggregation
- 💡 Intelligent explanations
- ⚠️ Smart risk assessment
- 🎯 Professional-grade reports

**Start analyzing with AI:**
```bash
python run_full_analysis.py
```

Good luck! 📈🚀

