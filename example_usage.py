"""
Quick Example: How to use the Stock Analysis System

This script demonstrates various ways to use the system.
"""

import os
import sys

def example_1_complete_analysis():
    """Example 1: Run complete analysis (recommended for most users)"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Complete Analysis Pipeline")
    print("="*70)
    print("\nThis will run:")
    print("  1. Stock price prediction")
    print("  2. News sentiment analysis")
    print("  3. Generate justification report")
    print("\nCommand:")
    print("  python run_full_analysis.py")
    print("\nThen enter:")
    print("  Stock ticker: AAPL")
    print("  Period: 2y")
    print("\nOutput: Multiple files with predictions, sentiment, and justification")

def example_2_prediction_only():
    """Example 2: Run prediction only"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Prediction Only")
    print("="*70)
    print("\nFor just price predictions without sentiment analysis:")
    print("\nCommand:")
    print("  python simp.py")
    print("\nThen enter:")
    print("  Stock ticker: GOOG")
    print("  Period: 5y")
    print("\nOutput: Prediction results, metrics, and visualizations")

def example_3_sentiment_only():
    """Example 3: Run sentiment analysis on existing predictions"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Sentiment Analysis Only")
    print("="*70)
    print("\nIf you already ran simp.py and want to add sentiment analysis:")
    print("\nCommand:")
    print("  python reason.py AAPL")
    print("\nNote: Replace AAPL with your stock symbol")
    print("\nOutput: Justification report with news sentiment")

def example_4_with_api_keys():
    """Example 4: Using API keys for better news coverage"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Using API Keys (Windows)")
    print("="*70)
    print("\nFor better news coverage, set API keys first:")
    print("\nPowerShell:")
    print('  $env:NEWSAPI_KEY = "your_newsapi_key_here"')
    print('  $env:ALPHA_VANTAGE_KEY = "your_alphavantage_key_here"')
    print("  python run_full_analysis.py")
    print("\nCommand Prompt:")
    print('  set NEWSAPI_KEY=your_newsapi_key_here')
    print('  set ALPHA_VANTAGE_KEY=your_alphavantage_key_here')
    print("  python run_full_analysis.py")
    print("\nGet free keys:")
    print("  NewsAPI: https://newsapi.org")
    print("  Alpha Vantage: https://www.alphavantage.co")

def example_5_batch_analysis():
    """Example 5: Analyze multiple stocks"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Batch Analysis (Multiple Stocks)")
    print("="*70)
    print("\nCreate a script to analyze multiple stocks:")
    
    print("\n--- batch_analyze.py ---")
    print("""
import subprocess
import sys

stocks = ['AAPL', 'GOOG', 'MSFT', 'TSLA', 'AMZN']
period = '2y'

for stock in stocks:
    print(f"\\nAnalyzing {stock}...")
    
    # Run prediction
    process = subprocess.Popen(
        [sys.executable, 'simp.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    process.communicate(input=f"{stock}\\n{period}\\n")
    
    # Run sentiment analysis
    subprocess.run([sys.executable, 'reason.py', stock])
    
print("\\nAll analyses complete!")
""")
    print("--- end of script ---\n")
    print("Run with: python batch_analyze.py")

def example_6_programmatic_usage():
    """Example 6: Use as a library in your own code"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Programmatic Usage")
    print("="*70)
    print("\nUse the modules in your own Python scripts:")
    
    print("\n--- your_script.py ---")
    print("""
from reason import SentimentJustifier
import json

# Analyze a stock
stock = 'AAPL'
justifier = SentimentJustifier()

# First, make sure you have prediction results
# (run simp.py first, or integrate it in your script)

# Load and analyze
prediction_data = justifier.load_prediction_results(stock)
if prediction_data:
    news = justifier.news_aggregator.get_all_news(stock)
    sentiment = justifier.analyze_news_sentiment(news)
    
    print(f"Overall Sentiment: {sentiment['overall_sentiment']}")
    print(f"Confidence: {sentiment['confidence']:.1f}%")
    print(f"Positive articles: {sentiment['positive_count']}")
    print(f"Negative articles: {sentiment['negative_count']}")
""")
    print("--- end of script ---\n")

def show_file_structure():
    """Show the expected file structure"""
    print("\n" + "="*70)
    print("PROJECT FILE STRUCTURE")
    print("="*70)
    print("""
stock/
│
├── simp.py                    # Main prediction model
├── sample.py                  # Data fetcher
├── reason.py                  # FinBERT sentiment analyzer
├── run_full_analysis.py       # Complete pipeline
│
├── requirements.txt           # Python dependencies
├── README.md                  # Documentation
├── example_usage.py           # This file
│
├── live_data.json            # Latest stock data (generated)
├── live_data.csv             # Latest stock data (generated)
│
├── AAPL_best_model.h5        # Trained models (generated)
├── GOOG_best_model.h5
│
└── Output Files (generated):
    ├── {STOCK}_{TIMESTAMP}_prediction_results.json
    ├── {STOCK}_{TIMESTAMP}_prediction_summary.txt
    ├── {STOCK}_{TIMESTAMP}_metrics.csv
    ├── {STOCK}_{TIMESTAMP}_justification.txt
    └── {STOCK}_{TIMESTAMP}_sentiment_analysis.json
""")

def main():
    """Main menu"""
    print("\n" + "="*70)
    print("  STOCK ANALYSIS SYSTEM - USAGE EXAMPLES")
    print("="*70)
    
    print("\nChoose an example to view:")
    print("  1. Complete Analysis Pipeline (Recommended)")
    print("  2. Prediction Only")
    print("  3. Sentiment Analysis Only")
    print("  4. Using API Keys for Better News")
    print("  5. Batch Analysis (Multiple Stocks)")
    print("  6. Programmatic Usage (Use as Library)")
    print("  7. Show File Structure")
    print("  0. Exit")
    
    choice = input("\nEnter choice (0-7): ").strip()
    
    if choice == '1':
        example_1_complete_analysis()
    elif choice == '2':
        example_2_prediction_only()
    elif choice == '3':
        example_3_sentiment_only()
    elif choice == '4':
        example_4_with_api_keys()
    elif choice == '5':
        example_5_batch_analysis()
    elif choice == '6':
        example_6_programmatic_usage()
    elif choice == '7':
        show_file_structure()
    elif choice == '0':
        print("\nGoodbye!")
        return
    else:
        print("\nInvalid choice!")
        return
    
    print("\n" + "="*70)
    print("\n💡 TIP: Check README.md for complete documentation")
    print("📚 For more help, visit the troubleshooting section in README.md")
    
    # Ask if user wants to see another example
    another = input("\nView another example? (y/n): ").strip().lower()
    if another == 'y':
        main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)

