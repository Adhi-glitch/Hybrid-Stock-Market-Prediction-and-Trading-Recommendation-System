"""
Test script to run complete analysis step by step
"""

import sys
import subprocess
import os
import glob

def run_complete_analysis():
    """Run complete analysis with proper error handling"""
    
    print("="*70)
    print("COMPLETE STOCK ANALYSIS TEST")
    print("="*70)
    
    # Step 1: Run prediction
    print("\nStep 1: Running price prediction...")
    print("-" * 50)
    
    with open('input.txt', 'w') as f:
        f.write("AAPL\n2y\n")
    
    try:
        with open('input.txt', 'r') as f:
            result = subprocess.run(
                [sys.executable, 'simp.py'],
                stdin=f,
                capture_output=True,
                text=True,
                timeout=600
            )
        
        print(f"Prediction return code: {result.returncode}")
        
        if result.returncode == 0:
            print("Prediction completed successfully!")
        else:
            print("Prediction had issues but may have completed")
            print("STDERR:", result.stderr[-500:] if result.stderr else "No errors")
        
        # Check for files
        print("\nChecking for generated files...")
        files = glob.glob("AAPL_*")
        print(f"Found {len(files)} AAPL files:")
        for f in files:
            print(f"  - {f}")
        
        if not files:
            print("No prediction files found!")
            return False
            
    except Exception as e:
        print(f"Error in prediction: {e}")
        return False
    finally:
        if os.path.exists('input.txt'):
            os.remove('input.txt')
    
    # Step 2: Run sentiment analysis
    print("\nStep 2: Running sentiment analysis...")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            [sys.executable, 'reason.py', 'AAPL'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(f"Sentiment return code: {result.returncode}")
        
        if result.returncode == 0:
            print("Sentiment analysis completed successfully!")
            print("STDOUT:", result.stdout[-500:] if result.stdout else "No output")
        else:
            print("Sentiment analysis had issues")
            print("STDERR:", result.stderr[-500:] if result.stderr else "No errors")
        
        # Check for additional files
        print("\nChecking for sentiment files...")
        files = glob.glob("AAPL_*")
        print(f"Total AAPL files now: {len(files)}")
        for f in files:
            print(f"  - {f}")
            
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return False
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    
    return True

if __name__ == "__main__":
    run_complete_analysis()
