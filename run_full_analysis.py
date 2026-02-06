"""
Complete Stock Analysis Pipeline
Runs data fetching, prediction, and sentiment analysis in sequence
"""

import subprocess
import sys
import time
import os

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def run_prediction_analysis(stock_symbol, period):
    """Run the prediction analysis using simp.py"""
    print_section("STEP 1: STOCK PRICE PREDICTION ANALYSIS")
    
    print(f"[PREDICTION] Running prediction model for {stock_symbol}...")
    print(f"📅 Period: {period}\n")
    
    try:
        # Run simp.py with input
        process = subprocess.Popen(
            [sys.executable, 'simp.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Send inputs
        inputs = f"{stock_symbol}\n{period}\n"
        
        # Communicate and stream output
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print("\n[OK] Prediction analysis completed successfully!")
            return True
        else:
            print(f"\n[ERROR] Prediction analysis failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error running prediction analysis: {e}")
        return False

def run_sentiment_analysis(stock_symbol):
    """Run the sentiment analysis using reason.py"""
    print_section("STEP 2: FINBERT SENTIMENT ANALYSIS")
    
    print(f"[NEWS] Running sentiment analysis for {stock_symbol}...")
    print("[AI] Loading FinBERT model and analyzing news...\n")
    
    try:
        # Run reason.py
        result = subprocess.run(
            [sys.executable, 'reason.py', stock_symbol],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n[OK] Sentiment analysis completed successfully!")
            return True
        else:
            print(f"\n[WARNING] Sentiment analysis completed with warnings")
            return True  # Still continue even if there are minor issues
            
    except Exception as e:
        print(f"[ERROR] Error running sentiment analysis: {e}")
        return False

def display_summary(stock_symbol):
    """Display a summary of generated files"""
    print_section("ANALYSIS COMPLETE - FILES GENERATED")
    
    import glob
    
    # Find generated files
    prediction_files = glob.glob(f"{stock_symbol}_*_prediction_results.json")
    summary_files = glob.glob(f"{stock_symbol}_*_prediction_summary.txt")
    metrics_files = glob.glob(f"{stock_symbol}_*_metrics.csv")
    justification_files = glob.glob(f"{stock_symbol}_*_justification.txt")
    sentiment_files = glob.glob(f"{stock_symbol}_*_sentiment_analysis.json")
    
    print("[FILES] Generated Files:")
    print("-" * 70)
    
    if prediction_files:
        print(f"\n[DATA] Prediction Results:")
        for f in sorted(prediction_files, key=os.path.getctime, reverse=True)[:3]:
            print(f"   • {f}")
    
    if summary_files:
        print(f"\n[REPORT] Prediction Summaries:")
        for f in sorted(summary_files, key=os.path.getctime, reverse=True)[:3]:
            print(f"   • {f}")
    
    if justification_files:
        print(f"\n[TIP] Sentiment Justifications:")
        for f in sorted(justification_files, key=os.path.getctime, reverse=True)[:3]:
            print(f"   • {f}")
    
    if sentiment_files:
        print(f"\n[NEWS] Sentiment Analysis Data:")
        for f in sorted(sentiment_files, key=os.path.getctime, reverse=True)[:3]:
            print(f"   • {f}")
    
    if metrics_files:
        print(f"\n📈 Model Metrics:")
        for f in sorted(metrics_files, key=os.path.getctime, reverse=True)[:3]:
            print(f"   • {f}")
    
    print("\n" + "-" * 70)
    print("\n[TIP] TIP: Check the justification file for detailed reasoning!")
    print("[NOTE] NOTE: To get better news coverage, set these environment variables:")
    print("   • NEWSAPI_KEY - Get free key at: https://newsapi.org")
    print("   • ALPHA_VANTAGE_KEY - Get free key at: https://www.alphavantage.co")

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("  COMPLETE STOCK ANALYSIS PIPELINE")
    print("  Prediction + Sentiment Analysis + Justification")
    print("="*70 + "\n")
    
    # Get user inputs
    stock_symbol = input("[DATA] Enter stock ticker (e.g., AAPL, GOOG, MSFT): ").strip().upper()
    
    if not stock_symbol:
        print("[ERROR] Stock symbol is required.")
        sys.exit(1)
    
    print("\n📅 Available periods:")
    print("   • 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")
    period = input("📅 Enter period (default: 2y): ").strip().lower() or "2y"
    
    print(f"\n[TARGET] Analyzing {stock_symbol} with {period} of historical data...")
    time.sleep(1)
    
    # Step 1: Run prediction analysis
    prediction_success = run_prediction_analysis(stock_symbol, period)
    
    if not prediction_success:
        print("\n[ERROR] Failed to complete prediction analysis. Exiting.")
        sys.exit(1)
    
    time.sleep(2)
    
    # Step 2: Run sentiment analysis
    sentiment_success = run_sentiment_analysis(stock_symbol)
    
    if not sentiment_success:
        print("\n[WARNING] Sentiment analysis had issues, but prediction results are available.")
    
    time.sleep(1)
    
    # Step 3: Display summary
    display_summary(stock_symbol)
    
    print("\n" + "="*70)
    print("  [OK] COMPLETE ANALYSIS FINISHED!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Analysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)

