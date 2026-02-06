"""
Installation Test Script
Verifies that all components are properly installed and working
"""

import sys
import importlib

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def check_python_version():
    """Check Python version"""
    print_header("Python Version Check")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("[OK] Python version is compatible (3.8+)")
        return True
    else:
        print("[ERROR] Python 3.8 or higher is required")
        return False

def check_module(module_name, package_name=None):
    """Check if a module can be imported"""
    if package_name is None:
        package_name = module_name
    
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"[OK] {package_name:20s} {version}")
        return True
    except ImportError:
        print(f"[ERROR] {package_name:20s} NOT FOUND")
        return False

def check_dependencies():
    """Check all required dependencies"""
    print_header("Dependency Check")
    
    dependencies = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('plotly', 'plotly'),
        ('sklearn', 'scikit-learn'),
        ('tensorflow', 'tensorflow'),
        ('yfinance', 'yfinance'),
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('requests', 'requests'),
    ]
    
    results = []
    for module, package in dependencies:
        results.append(check_module(module, package))
    
    return all(results)

def check_files():
    """Check if required files exist"""
    print_header("File Structure Check")
    
    import os
    
    required_files = [
        ('simp.py', 'Main prediction model'),
        ('sample.py', 'Data fetcher'),
        ('reason.py', 'FinBERT sentiment analyzer'),
        ('run_full_analysis.py', 'Complete pipeline'),
        ('requirements.txt', 'Dependencies list'),
        ('README.md', 'Documentation'),
        ('QUICKSTART.md', 'Quick start guide'),
    ]
    
    all_exist = True
    for filename, description in required_files:
        exists = os.path.exists(filename)
        status = "[OK]" if exists else "[ERROR]"
        print(f"{status} {filename:25s} - {description}")
        all_exist = all_exist and exists
    
    return all_exist

def check_api_keys():
    """Check API key configuration"""
    print_header("API Keys Check (Optional)")
    
    import os
    
    newsapi = os.getenv('NEWSAPI_KEY')
    alphavantage = os.getenv('ALPHA_VANTAGE_KEY')
    
    config_exists = False
    try:
        import config
        newsapi = newsapi or getattr(config, 'NEWSAPI_KEY', None)
        alphavantage = alphavantage or getattr(config, 'ALPHA_VANTAGE_KEY', None)
        config_exists = True
    except ImportError:
        pass
    
    print("News API Keys (optional for better news coverage):")
    
    if newsapi and len(newsapi) > 5:
        print(f"[OK] NewsAPI key configured")
    else:
        print(f"[WARNING]  NewsAPI key not found (optional)")
    
    if alphavantage and len(alphavantage) > 5:
        print(f"[OK] Alpha Vantage key configured")
    else:
        print(f"[WARNING]  Alpha Vantage key not found (optional)")
    
    if config_exists:
        print(f"[INFO]  Using config.py for API keys")
    else:
        print(f"[INFO]  No config.py found (using environment variables)")
    
    if not newsapi and not alphavantage:
        print("\n[TIP] TIP: For better news coverage, set API keys:")
        print("   1. Copy config_template.py to config.py")
        print("   2. Add your API keys to config.py")
        print("   3. Get free keys:")
        print("      • NewsAPI: https://newsapi.org")
        print("      • Alpha Vantage: https://www.alphavantage.co")
        print("\n   System will work without them using Yahoo Finance news.")

def test_basic_functionality():
    """Test basic functionality"""
    print_header("Basic Functionality Test")
    
    try:
        # Test FinBERT model loading (quick check)
        print("Testing FinBERT model initialization...")
        from reason import FinBERTAnalyzer
        
        print("[OK] FinBERT module can be imported")
        
        # Test news aggregator
        print("Testing news aggregator...")
        from reason import NewsAggregator
        
        print("[OK] News aggregator can be imported")
        
        # Test basic imports
        print("Testing TensorFlow...")
        import tensorflow as tf
        print(f"[OK] TensorFlow {tf.__version__} is working")
        
        print("Testing PyTorch...")
        import torch
        print(f"[OK] PyTorch {torch.__version__} is working")
        
        print("\n[SUCCESS] All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error during functionality test: {e}")
        return False

def print_summary(python_ok, deps_ok, files_ok, func_ok):
    """Print summary of test results"""
    print_header("TEST SUMMARY")
    
    print(f"Python Version: {'[OK] PASS' if python_ok else '[ERROR] FAIL'}")
    print(f"Dependencies:   {'[OK] PASS' if deps_ok else '[ERROR] FAIL'}")
    print(f"File Structure: {'[OK] PASS' if files_ok else '[ERROR] FAIL'}")
    print(f"Functionality:  {'[OK] PASS' if func_ok else '[ERROR] FAIL'}")
    
    print("\n" + "="*70)
    
    if python_ok and deps_ok and files_ok and func_ok:
        print("\n[SUCCESS] SUCCESS! All tests passed!")
        print("\n[OK] Your installation is complete and ready to use!")
        print("\n🚀 Next steps:")
        print("   1. (Optional) Set up API keys for better news coverage")
        print("   2. Run: python run_full_analysis.py")
        print("   3. Enter a stock ticker (e.g., AAPL)")
        print("   4. Wait for analysis to complete")
        print("\n[DOCS] For help, see:")
        print("   • QUICKSTART.md - Quick start guide")
        print("   • README.md - Full documentation")
        print("   • python example_usage.py - Usage examples")
    else:
        print("\n[WARNING] Some tests failed!")
        
        if not python_ok:
            print("\n[ERROR] Python version issue:")
            print("   Install Python 3.8 or higher")
        
        if not deps_ok:
            print("\n[ERROR] Missing dependencies:")
            print("   Run: pip install -r requirements.txt")
        
        if not files_ok:
            print("\n[ERROR] Missing files:")
            print("   Ensure all required files are in the directory")
        
        if not func_ok:
            print("\n[ERROR] Functionality issues:")
            print("   1. Try: pip install --upgrade -r requirements.txt")
            print("   2. Restart Python/terminal")
            print("   3. Check error messages above")

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("  STOCK ANALYSIS SYSTEM - INSTALLATION TEST")
    print("="*70)
    print("\nThis will verify your installation is working correctly...")
    
    # Run tests
    python_ok = check_python_version()
    deps_ok = check_dependencies()
    files_ok = check_files()
    check_api_keys()  # This is optional, doesn't affect overall status
    func_ok = test_basic_functionality()
    
    # Print summary
    print_summary(python_ok, deps_ok, files_ok, func_ok)
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error during testing: {e}")
        print("\nPlease report this error if the problem persists.")
        sys.exit(1)

