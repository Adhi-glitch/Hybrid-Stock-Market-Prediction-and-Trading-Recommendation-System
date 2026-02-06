"""
Simple test to run prediction and check file generation
"""

import sys
import subprocess
import os
import glob

def test_prediction():
    """Test prediction with detailed output"""
    print("Testing prediction for AAPL...")
    
    # Create input
    with open('test_input.txt', 'w') as f:
        f.write("AAPL\n1y\n")  # Use 1y for faster testing
    
    try:
        # Run with timeout
        with open('test_input.txt', 'r') as f:
            result = subprocess.run(
                [sys.executable, 'simp.py'],
                stdin=f,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
        
        print("Return code:", result.returncode)
        print("STDOUT length:", len(result.stdout))
        print("STDERR length:", len(result.stderr))
        
        if result.stdout:
            print("\nLast 1000 chars of STDOUT:")
            print(result.stdout[-1000:])
        
        if result.stderr:
            print("\nLast 500 chars of STDERR:")
            print(result.stderr[-500:])
        
        # Check for generated files
        print("\nChecking for generated files...")
        patterns = [
            "AAPL_*_prediction_results.json",
            "AAPL_*_prediction_summary.txt", 
            "AAPL_*_metrics.csv",
            "*prediction*",
            "*AAPL*"
        ]
        
        for pattern in patterns:
            files = glob.glob(pattern)
            print(f"Pattern '{pattern}': {files}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Timeout after 5 minutes")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        if os.path.exists('test_input.txt'):
            os.remove('test_input.txt')

if __name__ == "__main__":
    test_prediction()
