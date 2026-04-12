import yfinance as yf
import pandas as pd

def fetch_silver_data():
    """
    Fetches the last 1 month of Global Silver Futures data.
    """
    # Ticker for Global Silver Futures (Standard "SI=F")
    # Note: MCX follows this global price closely.
    ticker = "GC=F"

    print(f"--- 🚀 Fetching data for {ticker} ---")

    # Download last 1 month of data with 1-hour candles
    # 'interval' controls the timeframe (e.g., '1h', '1d', '5m')
    data = yf.download(ticker, period="5d", interval="15m")

    if data.empty:
        print("❌ No data found! Check your internet connection.")
        return

    # print(data.head()) shows the first 5 rows (Oldest data in the set)
    print("\n--- 📊 First 5 Candles (Start of Month) ---")
    print(data.head())

    # print(data.tail()) shows the last 5 rows (Most recent data)
    print("\n--- 📉 Last 5 Candles (Current Market) ---")
    print(data.tail())

    # Get the absolute latest closing price using .iloc (index location)
    # We use .item() to convert the numpy value to a standard float
    try:
        latest_price = data['Close'].iloc[-1].item()
        print(f"\n💎 Latest Silver Price (USD): ${latest_price:.2f}")
    except Exception as e:
        # Fallback if the data format is slightly different
        print(f"\n💎 Latest Silver Price (USD): {data['Close'].iloc[-1]}")

if __name__ == "__main__":
    fetch_silver_data()
