import sys
import subprocess
import json
import pandas as pd
import os
import time

# --- Setup yfinance ---
try:
    import yfinance as yf
except ImportError:
    print("yfinance not found. Attempting to install...")
    try:
        # Use subprocess to ensure installation works
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
        import yfinance as yf
        print("yfinance installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing yfinance: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during import: {e}")
        sys.exit(1)
# ----------------------


def fetch_and_save_data():
    """Reads stock ticker and period from stdin, fetches data, and saves it to live_data.json and live_data.csv."""
    
    # Read ticker and period from standard input (passed by the main script)
    try:
        # Read the stock ticker from the first line of stdin
        stock_name = sys.stdin.readline().strip().upper()
        # Read the period from the second line of stdin
        period = sys.stdin.readline().strip().lower()
    except Exception as e:
        print(f"Error reading input: {e}")
        # Use defaults if input reading fails
        stock_name = 'AAPL'
        period = '1y'
        print(f"Using default inputs: {stock_name}, {period}")
        
    if not stock_name or not period:
        print("Stock ticker and period are required. Cannot fetch data.")
        sys.exit(1)

    print(f"Fetching data for Ticker: {stock_name}, Period: {period}")

    # --- Determine appropriate download interval based on period input ---
    def select_interval(period):
        # Helper to parse number and unit from period (e.g. '1d', '6mo', '2y')
        if period.endswith('d'):
            days = int(period[:-1])
            if days == 1:
                return "1m"  # yfinance: minute data for up to 7d
            elif days < 10:
                return "1h"  # hour data for periods up to 60d
            elif days < 31:
                return "1d"  # daily data for periods >10d
            else:
                return "1d"
        elif period.endswith('mo'):
            months = int(period[:-2])
            if months <= 1:
                return "1d"  # daily
            elif months <= 12:
                return "1d"
            else:
                return "1wk"  # weekly
        elif period.endswith('y'):
            years = int(period[:-1])
            if years == 1:
                return "1d"  # daily
            else:
                return "1wk"  # weekly
        else:
            return "1d"  # fallback

    interval = select_interval(period)
    print(f"Selected interval for period '{period}': {interval}")

    import time
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt} to fetch data for {stock_name}...")
            data = yf.download(stock_name, period=period, interval=interval, progress=False, threads=True)
            # Check for timeout or partial download
            if data is None or data.empty:
                print(f"No data fetched for {stock_name} with period {period}. Check ticker or period.")
                if attempt < max_retries:
                    print("Retrying in 10 seconds...")
                    time.sleep(10)
                    continue
                else:
                    print("All attempts failed. Please check your internet connection, ticker, or period.")
                    if os.path.exists("live_data.json"):
                        os.remove("live_data.json")
                    if os.path.exists("live_data.csv"):
                        os.remove("live_data.csv")
                    sys.exit(1)
            # If data is not empty but has missing columns, treat as error
            required_cols = {'date', 'open', 'high', 'low', 'close'}
            data = data.reset_index()
            data.columns = [c.lower() for c in data.columns]
            if not required_cols.issubset(set(data.columns)):
                print(f"Fetched data is incomplete or malformed. Columns found: {data.columns.tolist()}")
                if attempt < max_retries:
                    print("Retrying in 10 seconds...")
                    time.sleep(10)
                    continue
                else:
                    print("All attempts failed. Please check your internet connection, ticker, or period.")
                    if os.path.exists("live_data.json"):
                        os.remove("live_data.json")
                    if os.path.exists("live_data.csv"):
                        os.remove("live_data.csv")
                    sys.exit(1)
            data['name'] = stock_name
            data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
            # --- Generate unique filenames based on ticker, period, and current timestamp ---
            from datetime import datetime
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{stock_name}_{period}_{timestamp_str}"
            csv_filename = f"{base_filename}.csv"
            json_filename = f"{base_filename}.json"
            data.to_csv(csv_filename, index=False)
            print(f"Data also saved to {csv_filename}")
            data.to_json(json_filename, orient="records", date_format="iso")
            print(f"Successfully fetched {len(data)} data points for {stock_name} and saved to {json_filename}.")
            break
        except Exception as e:
            print(f"Error fetching data: {e}")
            if attempt < max_retries:
                print("Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print("All attempts failed. Please check your internet connection, ticker, or period.")
                if os.path.exists("live_data.json"):
                    os.remove("live_data.json")
                if os.path.exists("live_data.csv"):
                    os.remove("live_data.csv")
                sys.exit(1)

if __name__ == "__main__":
    # Wait briefly to ensure inputs from the parent process are available
    time.sleep(0.1)
    fetch_and_save_data()
