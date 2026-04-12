# 1. Define your watchlist
from google.genai import client
import yfinance as yf
from google import genai

from real_data import generate_llm_prompt, calculate_rsi, evaluate_price
from config import client

watchlist = ["RVNL.NS", "LTIM.NS", "TCS.NS", "RELIANCE.NS"]

print("🚀 Starting Daily Market Scan...")

# 2. Loop through each stock
for ticker_symbol in watchlist:
    print(f"\n=========================================")
    print(f"Fetching data for {ticker_symbol}....")

    try:
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(period="1mo")

        # Ensure we actually got data back
        if history.empty:
            print(f"⚠️ No data found for {ticker_symbol}. Skipping...")
            continue

        latest_close = history['Close'].iloc[-1]
        avg_value = history['Close'].mean()
        current_rsi = calculate_rsi(history)
        signal = evaluate_price(latest_close, avg_value)

        final_prompt = generate_llm_prompt(ticker_symbol, latest_close, avg_value, signal, current_rsi)
        # client = genai.Client(api_key="AIzaSyBeag_l2F-U1EHLSyjorJcHjOtT7IAjbb8")  # Removed, now imported from config
        # Call the LLM
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=final_prompt
        )

        print(f"--- 🎯 AI VERDICT FOR {ticker_symbol} ---")
        print(response.text)

        # TODO: Add your database saving logic right here!
        # save_to_db(ticker_symbol, response.text...)

    except Exception as e:
        print(f"❌ Failed to process {ticker_symbol}: {e}")

print("\n✅ Market Scan Complete!")