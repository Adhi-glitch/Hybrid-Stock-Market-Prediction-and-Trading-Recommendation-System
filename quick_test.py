"""
Quick test script to run stock analysis with predefined inputs
"""

import sys
import subprocess
import os

def run_prediction():
    """Run prediction analysis"""
    print("Running prediction analysis for AAPL...")
    
    # Create input file
    with open('input.txt', 'w') as f:
        f.write("AAPL\n2y\n")
    
    # Run simp.py with input file
    try:
        with open('input.txt', 'r') as f:
            result = subprocess.run(
                [sys.executable, 'simp.py'],
                stdin=f,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
        
        print("Prediction completed!")
        print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        if result.stderr:
            print("STDERR:", result.stderr[-500:])  # Last 500 chars
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Prediction timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"Error running prediction: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists('input.txt'):
            os.remove('input.txt')

def run_sentiment():
    """Run sentiment analysis"""
    print("\nRunning sentiment analysis...")
    
    try:
        result = subprocess.run(
            [sys.executable, 'reason.py', 'AAPL'],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        print("Sentiment analysis completed!")
        print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        if result.stderr:
            print("STDERR:", result.stderr[-500:])
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Sentiment analysis timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"Error running sentiment analysis: {e}")
        return False

def main():
    print("="*70)
    print("QUICK STOCK ANALYSIS TEST")
    print("="*70)
    
    # Step 1: Run prediction
    pred_success = run_prediction()
    
    if not pred_success:
        print("\nPrediction failed. Stopping.")
        return
    
    # Step 2: Run sentiment analysis
    sent_success = run_sentiment()
    
    if sent_success:
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print("\nCheck the generated files:")
        print("- AAPL_*_prediction_results.json")
        print("- AAPL_*_justification.txt")
        print("- AAPL_*_sentiment_analysis.json")
    else:
        print("\nSentiment analysis failed, but prediction completed.")

if __name__ == "__main__":
    main()
