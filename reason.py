"""
FinBERT Sentiment Analysis + NLP Text Generation for Stock Prediction Justification
Uses financial news sentiment + Seq2Seq models to provide AI-generated reasoning
"""

import os
import json
import sys
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import required libraries
try:
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        T5ForConditionalGeneration,
        T5Tokenizer,
        pipeline
    )
    import numpy as np
    import pandas as pd
    import requests
    from typing import Dict, List, Tuple
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Installing required packages...")
    import subprocess
    packages = ['torch', 'transformers', 'requests', 'pandas', 'numpy', 'sentencepiece']
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
    print("Packages installed. Please run the script again.")
    sys.exit(0)


class FinBERTAnalyzer:
    """FinBERT-based sentiment analyzer for financial news"""
    
    def __init__(self):
        """Initialize FinBERT model and tokenizer"""
        print("Initializing FinBERT model...")
        try:
            # Load FinBERT model trained on financial texts
            self.model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            
            # Sentiment labels
            self.labels = ['positive', 'negative', 'neutral']
            print("FinBERT model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Error loading FinBERT: {e}")
            print("Falling back to alternative sentiment analysis...")
            self.model = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of financial text using FinBERT
        
        Args:
            text: Financial news text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if self.model is None:
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
        
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                   max_length=512, padding=True)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert to scores
            scores = predictions[0].tolist()
            sentiment_dict = {
                'positive': scores[0],
                'negative': scores[1],
                'neutral': scores[2]
            }
            
            return sentiment_dict
            
        except Exception as e:
            print(f"[WARNING] Error in sentiment analysis: {e}")
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
    
    def get_dominant_sentiment(self, sentiment_scores: Dict[str, float]) -> Tuple[str, float]:
        """Get the dominant sentiment and its confidence"""
        dominant = max(sentiment_scores.items(), key=lambda x: x[1])
        return dominant[0], dominant[1]


class TextGenerator:
    """NLP Seq2Seq model for generating natural language explanations"""
    
    def __init__(self, model_name="google/flan-t5-base"):
        """Initialize text generation model"""
        print("[AI] Initializing NLP Text Generation model...")
        try:
            self.model_name = model_name
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            print(f"[OK] Text generation model loaded on {self.device}")
        except Exception as e:
            print(f"[WARNING] Could not load {model_name}: {e}")
            print("Falling back to rule-based generation...")
            self.model = None
    
    def generate_sentiment_summary(self, news_articles: List[Dict], sentiment_data: Dict) -> str:
        """Generate natural language summary of sentiment analysis"""
        if self.model is None:
            return self._rule_based_summary(news_articles, sentiment_data)
        
        try:
            # Prepare context from top news headlines
            headlines = []
            for article in news_articles[:10]:
                title = article.get('title', article.get('headline', ''))
                if title:
                    headlines.append(title)
            
            context = " | ".join(headlines[:5])
            
            # Create prompt for T5
            prompt = f"""Summarize the market sentiment based on these news headlines: {context}
            
Sentiment scores: {sentiment_data['positive_count']} positive, {sentiment_data['negative_count']} negative, {sentiment_data['neutral_count']} neutral articles.
Overall sentiment: {sentiment_data['overall_sentiment']}.

Generate a professional 2-3 sentence summary explaining the market sentiment:"""
            
            # Generate
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=150,
                    num_beams=4,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    early_stopping=True
                )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary.strip()
            
        except Exception as e:
            print(f"[WARNING] Error in text generation: {e}")
            return self._rule_based_summary(news_articles, sentiment_data)
    
    def generate_recommendation_explanation(self, prediction_data: Dict, sentiment_data: Dict, 
                                           key_news: List[str]) -> str:
        """Generate detailed explanation for the recommendation"""
        if self.model is None:
            return self._rule_based_explanation(prediction_data, sentiment_data, key_news)
        
        try:
            stock = prediction_data['stock_name']
            pred_change = prediction_data['percent_change']
            recommendation = prediction_data['recommendation']
            sentiment = sentiment_data['overall_sentiment']
            
            # Prepare news context
            news_context = " | ".join(key_news[:3]) if key_news else "Limited news available"
            
            prompt = f"""As a financial analyst, explain this stock recommendation:

Stock: {stock}
Predicted Change: {pred_change:.2f}%
Recommendation: {recommendation}
Market Sentiment: {sentiment} ({sentiment_data['confidence']:.1f}% confidence)
Key News: {news_context}

Generate a professional 3-4 sentence explanation that:
1. Explains why this recommendation makes sense
2. Addresses alignment between technical prediction and market sentiment
3. Mentions key risks or opportunities
4. Provides actionable insight

Explanation:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=200,
                    num_beams=5,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    early_stopping=True,
                    repetition_penalty=1.2
                )
            
            explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return explanation.strip()
            
        except Exception as e:
            print(f"[WARNING] Error in explanation generation: {e}")
            return self._rule_based_explanation(prediction_data, sentiment_data, key_news)
    
    def generate_risk_analysis(self, prediction_data: Dict, sentiment_data: Dict) -> str:
        """Generate risk analysis narrative"""
        if self.model is None:
            return self._rule_based_risk_analysis(prediction_data, sentiment_data)
        
        try:
            volatility = prediction_data.get('annual_volatility', 0)
            risk_level = prediction_data.get('risk_level', 'UNKNOWN')
            sentiment = sentiment_data['overall_sentiment']
            
            # Check for divergence
            pred_direction = 'bullish' if prediction_data['percent_change'] > 0 else 'bearish'
            sent_direction = 'bullish' if sentiment == 'positive' else 'bearish' if sentiment == 'negative' else 'neutral'
            
            divergence = "yes" if pred_direction != sent_direction and sent_direction != 'neutral' else "no"
            
            prompt = f"""Analyze the risk factors for this investment:

Volatility: {volatility:.1f}% (annual)
Risk Level: {risk_level}
Technical Signal: {pred_direction}
Market Sentiment: {sent_direction}
Signal Divergence: {divergence}

Generate a professional 2-3 sentence risk analysis that explains:
1. The main risk factors
2. Whether divergence increases risk
3. Recommended risk management approach

Risk Analysis:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=150,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,
                    early_stopping=True
                )
            
            analysis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return analysis.strip()
            
        except Exception as e:
            print(f"[WARNING] Error in risk analysis: {e}")
            return self._rule_based_risk_analysis(prediction_data, sentiment_data)
    
    def _rule_based_summary(self, news_articles: List[Dict], sentiment_data: Dict) -> str:
        """Fallback rule-based summary"""
        sentiment = sentiment_data['overall_sentiment']
        confidence = sentiment_data['confidence']
        pos = sentiment_data['positive_count']
        neg = sentiment_data['negative_count']
        
        if sentiment == 'positive':
            return f"Market sentiment is predominantly positive ({confidence:.1f}% confidence) with {pos} positive articles outweighing {neg} negative ones. Recent news suggests optimism and favorable market conditions."
        elif sentiment == 'negative':
            return f"Market sentiment is predominantly negative ({confidence:.1f}% confidence) with {neg} negative articles dominating the narrative. Recent news indicates concerns and potential headwinds."
        else:
            return f"Market sentiment is neutral ({confidence:.1f}% confidence) with mixed signals from news coverage. Investors appear cautious with no clear directional bias."
    
    def _rule_based_explanation(self, prediction_data: Dict, sentiment_data: Dict, key_news: List[str]) -> str:
        """Fallback rule-based explanation"""
        pred_change = prediction_data['percent_change']
        recommendation = prediction_data['recommendation']
        sentiment = sentiment_data['overall_sentiment']
        
        pred_dir = "upward" if pred_change > 0 else "downward"
        align = "aligns with" if (pred_change > 0 and sentiment == 'positive') or (pred_change < 0 and sentiment == 'negative') else "diverges from"
        
        explanation = f"The {recommendation} recommendation is based on a predicted {abs(pred_change):.2f}% {pred_dir} movement. "
        explanation += f"This technical prediction {align} the current {sentiment} market sentiment. "
        
        if align == "aligns with":
            explanation += "The convergence of technical indicators and market sentiment strengthens this signal, suggesting higher confidence in the prediction."
        else:
            explanation += "The divergence between technical and sentiment signals suggests caution, and position sizing should be adjusted accordingly."
        
        return explanation
    
    def _rule_based_risk_analysis(self, prediction_data: Dict, sentiment_data: Dict) -> str:
        """Fallback rule-based risk analysis"""
        volatility = prediction_data.get('annual_volatility', 0)
        risk_level = prediction_data.get('risk_level', 'UNKNOWN')
        
        analysis = f"With {volatility:.1f}% annual volatility, this stock carries {risk_level.lower()} risk. "
        
        if volatility > 40:
            analysis += "The high volatility suggests significant price swings and requires careful position sizing. "
        elif volatility > 25:
            analysis += "Moderate volatility indicates normal market fluctuations with manageable risk. "
        else:
            analysis += "Low volatility suggests relatively stable price action with lower risk exposure. "
        
        # Check divergence
        pred_change = prediction_data['percent_change']
        sentiment = sentiment_data['overall_sentiment']
        
        if (pred_change > 0 and sentiment == 'negative') or (pred_change < 0 and sentiment == 'positive'):
            analysis += "Signal divergence between technicals and sentiment adds uncertainty, warranting reduced position sizes."
        
        return analysis


class NewsAggregator:
    """Fetch and aggregate financial news from multiple sources"""
    
    def __init__(self):
        """Initialize news aggregator"""
        # Try to get API keys from config.py first, then environment variables
        self.newsapi_key = None
        self.alpha_vantage_key = None
        
        try:
            import config
            self.newsapi_key = getattr(config, 'NEWSAPI_KEY', None)
            self.alpha_vantage_key = getattr(config, 'ALPHA_VANTAGE_KEY', None)
            if self.newsapi_key or self.alpha_vantage_key:
                print("[OK] Loaded API keys from config.py")
        except ImportError:
            pass  # config.py doesn't exist, will use env vars
        
        # Fall back to environment variables if not in config
        if not self.newsapi_key:
            self.newsapi_key = os.getenv('NEWSAPI_KEY', None)
        if not self.alpha_vantage_key:
            self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY', None)
    
    def fetch_newsapi(self, stock_symbol: str, days: int = 7) -> List[Dict]:
        """Fetch news from NewsAPI"""
        if not self.newsapi_key:
            print("[WARNING] NewsAPI key not found. Set NEWSAPI_KEY environment variable.")
            return []
        
        try:
            url = 'https://newsapi.org/v2/everything'
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            params = {
                'q': f"{stock_symbol} OR stock OR earnings OR revenue",
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'apiKey': self.newsapi_key,
                'pageSize': 20
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                print(f"[OK] Fetched {len(articles)} articles from NewsAPI")
                return articles
            else:
                print(f"[WARNING] NewsAPI returned status code: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"[WARNING] Error fetching from NewsAPI: {e}")
            return []
    
    def fetch_alpha_vantage_news(self, stock_symbol: str) -> List[Dict]:
        """Fetch news from Alpha Vantage"""
        if not self.alpha_vantage_key:
            print("[WARNING] Alpha Vantage key not found. Set ALPHA_VANTAGE_KEY environment variable.")
            return []
        
        try:
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': stock_symbol,
                'apikey': self.alpha_vantage_key,
                'limit': 50
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('feed', [])
                print(f"[OK] Fetched {len(articles)} articles from Alpha Vantage")
                return articles
            else:
                print(f"[WARNING] Alpha Vantage returned status code: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"[WARNING] Error fetching from Alpha Vantage: {e}")
            return []
    
    def fetch_yahoo_finance_news(self, stock_symbol: str) -> List[Dict]:
        """Fetch news from Yahoo Finance (via yfinance)"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(stock_symbol)
            news = ticker.news
            
            if news:
                print(f"[OK] Fetched {len(news)} articles from Yahoo Finance")
                return news
            return []
            
        except Exception as e:
            print(f"[WARNING] Error fetching from Yahoo Finance: {e}")
            return []
    
    def fetch_finnhub_news(self, stock_symbol: str) -> List[Dict]:
        """Fetch news from Finnhub API"""
        finnhub_key = os.getenv('FINNHUB_KEY', None)
        
        try:
            import config
            finnhub_key = finnhub_key or getattr(config, 'FINNHUB_KEY', None)
        except ImportError:
            pass
        
        if not finnhub_key:
            return []
        
        try:
            url = 'https://finnhub.io/api/v1/company-news'
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            params = {
                'symbol': stock_symbol,
                'from': from_date,
                'to': to_date,
                'token': finnhub_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                articles = response.json()
                print(f"[OK] Fetched {len(articles)} articles from Finnhub")
                return articles
            else:
                print(f"[WARNING] Finnhub returned status code: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"[WARNING] Error fetching from Finnhub: {e}")
            return []
    
    def fetch_marketaux_news(self, stock_symbol: str) -> List[Dict]:
        """Fetch news from MarketAux API"""
        marketaux_key = os.getenv('MARKETAUX_KEY', None)
        
        try:
            import config
            marketaux_key = marketaux_key or getattr(config, 'MARKETAUX_KEY', None)
        except ImportError:
            pass
        
        if not marketaux_key:
            return []
        
        try:
            url = 'https://api.marketaux.com/v1/news/all'
            
            params = {
                'symbols': stock_symbol,
                'filter_entities': 'true',
                'language': 'en',
                'api_token': marketaux_key,
                'limit': 20
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('data', [])
                print(f"[OK] Fetched {len(articles)} articles from MarketAux")
                return articles
            else:
                print(f"[WARNING] MarketAux returned status code: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"[WARNING] Error fetching from MarketAux: {e}")
            return []
    
    def get_all_news(self, stock_symbol: str) -> List[Dict]:
        """Aggregate news from all available sources"""
        print(f"\n[NEWS] Fetching news for {stock_symbol}...")
        
        all_news = []
        
        # Fetch from Yahoo Finance (free, no API key needed)
        yahoo_news = self.fetch_yahoo_finance_news(stock_symbol)
        all_news.extend(yahoo_news)
        
        # Fetch from NewsAPI if key available
        newsapi_articles = self.fetch_newsapi(stock_symbol)
        all_news.extend(newsapi_articles)
        
        # Fetch from Alpha Vantage if key available
        av_articles = self.fetch_alpha_vantage_news(stock_symbol)
        all_news.extend(av_articles)
        
        # Fetch from Finnhub if key available
        finnhub_articles = self.fetch_finnhub_news(stock_symbol)
        all_news.extend(finnhub_articles)
        
        # Fetch from MarketAux if key available
        marketaux_articles = self.fetch_marketaux_news(stock_symbol)
        all_news.extend(marketaux_articles)
        
        # Deduplicate based on titles
        seen_titles = set()
        unique_news = []
        for article in all_news:
            title = article.get('title', article.get('headline', ''))
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_news.append(article)
        
        print(f"[DATA] Total unique articles collected: {len(unique_news)}")
        return unique_news


class SentimentJustifier:
    """Generate justification for stock predictions based on sentiment analysis"""
    
    def __init__(self):
        """Initialize the justifier"""
        self.finbert = FinBERTAnalyzer()
        self.news_aggregator = NewsAggregator()
        self.text_generator = TextGenerator()
    
    def load_prediction_results(self, stock_symbol: str) -> Dict:
        """Load the most recent prediction results for a stock"""
        pattern = f"{stock_symbol}_*_prediction_results.json"
        files = glob.glob(pattern)
        
        if not files:
            print(f"[WARNING] No prediction results found for {stock_symbol}")
            return None
        
        # Get the most recent file
        latest_file = max(files, key=os.path.getctime)
        
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        print(f"[OK] Loaded prediction results from: {latest_file}")
        return data
    
    def analyze_news_sentiment(self, news_articles: List[Dict]) -> Dict:
        """Analyze sentiment of all news articles"""
        if not news_articles:
            return {
                'overall_sentiment': 'neutral',
                'confidence': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'article_sentiments': []
            }
        
        print("\n[ANALYZING] Analyzing sentiment of news articles...")
        
        article_sentiments = []
        sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for i, article in enumerate(news_articles[:30]):  # Analyze up to 30 articles
            # Extract text from different news formats
            title = article.get('title', article.get('headline', ''))
            description = article.get('description', article.get('summary', ''))
            
            if not title and not description:
                continue
            
            # Combine title and description
            text = f"{title}. {description}"
            
            # Analyze sentiment
            scores = self.finbert.analyze_sentiment(text)
            dominant, confidence = self.finbert.get_dominant_sentiment(scores)
            
            article_sentiments.append({
                'title': title[:100],
                'sentiment': dominant,
                'confidence': confidence,
                'scores': scores,
                'date': article.get('publishedAt', article.get('time_published', 'N/A'))
            })
            
            sentiment_scores[dominant] += 1
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Analyzed {i + 1} articles...")
        
        # Calculate overall sentiment
        total = sum(sentiment_scores.values())
        if total == 0:
            overall_sentiment = 'neutral'
            confidence = 0.0
        else:
            overall_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])[0]
            confidence = sentiment_scores[overall_sentiment] / total * 100
        
        return {
            'overall_sentiment': overall_sentiment,
            'confidence': confidence,
            'positive_count': sentiment_scores['positive'],
            'negative_count': sentiment_scores['negative'],
            'neutral_count': sentiment_scores['neutral'],
            'total_analyzed': total,
            'article_sentiments': article_sentiments
        }
    
    def generate_justification(self, prediction_data: Dict, sentiment_data: Dict) -> str:
        """Generate a comprehensive justification for the prediction"""
        
        stock = prediction_data['stock_name']
        pred_price = prediction_data['predicted_price']
        current_price = prediction_data['current_price']
        percent_change = prediction_data['percent_change']
        recommendation = prediction_data['recommendation']
        model_confidence = prediction_data['confidence']
        
        sentiment = sentiment_data['overall_sentiment']
        sentiment_confidence = sentiment_data['confidence']
        
        # Start building justification
        justification = f"\n{'='*70}\n"
        justification += f"PREDICTION JUSTIFICATION REPORT FOR {stock}\n"
        justification += f"{'='*70}\n\n"
        
        # Section 1: Price Prediction Summary
        justification += "[DATA] PRICE PREDICTION SUMMARY\n"
        justification += "-" * 70 + "\n"
        justification += f"Current Price: ${current_price:.2f}\n"
        justification += f"Predicted Price: ${pred_price:.2f}\n"
        justification += f"Expected Change: {percent_change:+.2f}%\n"
        justification += f"Recommendation: {recommendation}\n"
        justification += f"Model Confidence: {model_confidence:.1f}%\n\n"
        
        # Section 2: Sentiment Analysis
        justification += "[NEWS] NEWS SENTIMENT ANALYSIS\n"
        justification += "-" * 70 + "\n"
        justification += f"Overall Sentiment: {sentiment.upper()}\n"
        justification += f"Sentiment Confidence: {sentiment_confidence:.1f}%\n"
        justification += f"Articles Analyzed: {sentiment_data['total_analyzed']}\n"
        justification += f"  • Positive: {sentiment_data['positive_count']} articles\n"
        justification += f"  • Negative: {sentiment_data['negative_count']} articles\n"
        justification += f"  • Neutral: {sentiment_data['neutral_count']} articles\n\n"
        
        # Section 3: Sentiment-Prediction Alignment
        justification += "[TARGET] SENTIMENT-PREDICTION ALIGNMENT\n"
        justification += "-" * 70 + "\n"
        
        # Check if sentiment aligns with prediction
        prediction_direction = 'positive' if percent_change > 0 else 'negative' if percent_change < 0 else 'neutral'
        
        if prediction_direction == sentiment:
            alignment = "STRONG ALIGNMENT [OK]"
            justification += f"Status: {alignment}\n"
            justification += f"The {sentiment} market sentiment SUPPORTS the {recommendation} recommendation.\n"
            justification += f"Both technical indicators and news sentiment suggest a {prediction_direction} outlook.\n\n"
        elif (prediction_direction == 'positive' and sentiment == 'neutral') or \
             (prediction_direction == 'negative' and sentiment == 'neutral'):
            alignment = "PARTIAL ALIGNMENT [WARNING]"
            justification += f"Status: {alignment}\n"
            justification += f"Market sentiment is {sentiment}, while prediction suggests {prediction_direction} movement.\n"
            justification += f"Consider exercising caution. Monitor news developments closely.\n\n"
        else:
            alignment = "DIVERGENCE [WARNING]"
            justification += f"Status: {alignment}\n"
            justification += f"Market sentiment ({sentiment}) DIVERGES from prediction ({prediction_direction}).\n"
            justification += f"This suggests mixed signals. Consider waiting for clearer indicators.\n\n"
        
        # Section 4: AI-Generated Sentiment Summary
        justification += "[AI] AI-GENERATED SENTIMENT SUMMARY\n"
        justification += "-" * 70 + "\n"
        
        # Get all articles for context
        all_articles = []
        for art in sentiment_data['article_sentiments']:
            all_articles.append({'title': art['title'], 'sentiment': art['sentiment']})
        
        # Generate AI summary
        ai_summary = self.text_generator.generate_sentiment_summary(all_articles, sentiment_data)
        justification += f"{ai_summary}\n\n"
        
        # Section 5: Key News Highlights
        justification += "📌 KEY NEWS HIGHLIGHTS\n"
        justification += "-" * 70 + "\n"
        
        # Get top positive and negative articles
        articles = sentiment_data['article_sentiments']
        positive_articles = [a for a in articles if a['sentiment'] == 'positive']
        negative_articles = [a for a in articles if a['sentiment'] == 'negative']
        
        positive_articles.sort(key=lambda x: x['confidence'], reverse=True)
        negative_articles.sort(key=lambda x: x['confidence'], reverse=True)
        
        key_positive_news = []
        key_negative_news = []
        
        if positive_articles:
            justification += "\n[POSITIVE] Most Positive News:\n"
            for i, article in enumerate(positive_articles[:3], 1):
                justification += f"{i}. {article['title']}\n"
                justification += f"   Confidence: {article['confidence']*100:.1f}%\n"
                key_positive_news.append(article['title'])
        
        if negative_articles:
            justification += "\n[NEGATIVE] Most Negative News:\n"
            for i, article in enumerate(negative_articles[:3], 1):
                justification += f"{i}. {article['title']}\n"
                justification += f"   Confidence: {article['confidence']*100:.1f}%\n"
                key_negative_news.append(article['title'])
        
        justification += "\n"
        
        # Collect key news for AI explanation
        key_news = key_positive_news + key_negative_news
        
        # Section 6: AI-Generated Risk Assessment
        justification += "[WARNING] AI-GENERATED RISK ASSESSMENT\n"
        justification += "-" * 70 + "\n"
        risk_level = prediction_data.get('risk_level', 'UNKNOWN')
        volatility = prediction_data.get('annual_volatility', 0)
        justification += f"Risk Level: {risk_level}\n"
        justification += f"Annual Volatility: {volatility:.1f}%\n\n"
        
        # Generate AI risk analysis
        ai_risk_analysis = self.text_generator.generate_risk_analysis(prediction_data, sentiment_data)
        justification += f"{ai_risk_analysis}\n"
        
        # Additional risk factors based on sentiment divergence
        if alignment == "DIVERGENCE [WARNING]":
            justification += "\n[WARNING] CAUTION: Sentiment-prediction divergence increases risk.\n"
            justification += "Consider reducing position size or waiting for confirmation.\n"
        elif sentiment_data['negative_count'] > sentiment_data['positive_count'] * 1.5:
            justification += "\n[WARNING] CAUTION: Significantly more negative news than positive.\n"
            justification += "Market may be facing headwinds.\n"
        
        justification += "\n"
        
        # Section 7: AI-Generated Detailed Explanation
        justification += "[INSIGHT] AI-GENERATED DETAILED EXPLANATION\n"
        justification += "-" * 70 + "\n"
        
        # Generate comprehensive AI explanation
        ai_explanation = self.text_generator.generate_recommendation_explanation(
            prediction_data, sentiment_data, key_news
        )
        justification += f"{ai_explanation}\n\n"
        
        # Section 8: Final Recommendation
        justification += "[TARGET] FINAL RECOMMENDATION\n"
        justification += "-" * 70 + "\n"
        
        # Adjust recommendation based on sentiment
        if recommendation in ['STRONG BUY', 'BUY'] and sentiment == 'positive':
            justification += f"[OK] {recommendation} is REINFORCED by positive market sentiment.\n"
            justification += "Both technical analysis and news sentiment support upward movement.\n"
            final_rec = recommendation
        elif recommendation in ['STRONG SELL', 'SELL'] and sentiment == 'negative':
            justification += f"[OK] {recommendation} is REINFORCED by negative market sentiment.\n"
            justification += "Both technical analysis and news sentiment suggest downward pressure.\n"
            final_rec = recommendation
        elif recommendation in ['STRONG BUY', 'BUY'] and sentiment == 'negative':
            justification += f"[WARNING] {recommendation} CONTRADICTS negative market sentiment.\n"
            justification += "Consider a more conservative approach. Reduce position size.\n"
            justification += "REVISED RECOMMENDATION: Consider HOLD or small position.\n"
            final_rec = "HOLD (Revised from " + recommendation + ")"
        elif recommendation in ['STRONG SELL', 'SELL'] and sentiment == 'positive':
            justification += f"[WARNING] {recommendation} CONTRADICTS positive market sentiment.\n"
            justification += "Technical indicators suggest caution despite positive news.\n"
            justification += "REVISED RECOMMENDATION: Monitor closely, consider HOLD.\n"
            final_rec = "HOLD (Revised from " + recommendation + ")"
        else:
            justification += f"[INFO] {recommendation} recommendation stands.\n"
            justification += "Sentiment is neutral, rely on technical analysis.\n"
            final_rec = recommendation
        
        justification += f"\n[REPORT] FINAL RECOMMENDATION: {final_rec}\n"
        
        justification += f"\n{'='*70}\n"
        justification += f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        justification += f"Powered by FinBERT + AI Text Generation (Seq2Seq)\n"
        justification += f"{'='*70}\n"
        
        return justification
    
    def save_justification(self, stock_symbol: str, justification: str, 
                          prediction_data: Dict, sentiment_data: Dict):
        """Save justification to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save text report
        text_filename = f"{stock_symbol}_{timestamp}_justification.txt"
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write(justification)
        
        # Save detailed JSON
        json_filename = f"{stock_symbol}_{timestamp}_sentiment_analysis.json"
        detailed_data = {
            'timestamp': timestamp,
            'stock_symbol': stock_symbol,
            'prediction': {
                'current_price': prediction_data['current_price'],
                'predicted_price': prediction_data['predicted_price'],
                'percent_change': prediction_data['percent_change'],
                'recommendation': prediction_data['recommendation'],
                'confidence': prediction_data['confidence']
            },
            'sentiment': sentiment_data
        }
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2)
        
        print(f"\n[SAVED] Justification saved to:")
        print(f"   [FILE] {text_filename}")
        print(f"   [DATA] {json_filename}")
        
        return text_filename, json_filename
    
    def analyze_stock(self, stock_symbol: str):
        """Main method to analyze stock and generate justification"""
        print(f"\n{'='*70}")
        print(f"FinBERT SENTIMENT ANALYSIS FOR {stock_symbol}")
        print(f"{'='*70}")
        
        # Load prediction results
        prediction_data = self.load_prediction_results(stock_symbol)
        if not prediction_data:
            print("[ERROR] No prediction data found. Please run simp.py first.")
            return
        
        # Fetch news
        news_articles = self.news_aggregator.get_all_news(stock_symbol)
        
        if not news_articles:
            print("[WARNING] No news articles found. Generating basic justification...")
            news_articles = []
        
        # Analyze sentiment
        sentiment_data = self.analyze_news_sentiment(news_articles)
        
        # Generate justification
        justification = self.generate_justification(prediction_data, sentiment_data)
        
        # Display justification
        print(justification)
        
        # Save to files
        self.save_justification(stock_symbol, justification, prediction_data, sentiment_data)
        
        print("\n[OK] Analysis complete!")


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("FINBERT SENTIMENT ANALYSIS FOR STOCK PREDICTIONS")
    print("="*70 + "\n")
    
    # Get stock symbol
    if len(sys.argv) > 1:
        stock_symbol = sys.argv[1].upper()
    else:
        stock_symbol = input("Enter stock ticker (e.g., AAPL, GOOG, MSFT): ").strip().upper()
    
    if not stock_symbol:
        print("[ERROR] Stock symbol is required.")
        sys.exit(1)
    
    # Create justifier and analyze
    justifier = SentimentJustifier()
    justifier.analyze_stock(stock_symbol)


if __name__ == "__main__":
    main()

