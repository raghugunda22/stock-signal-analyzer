import yfinance as yf
from google import genai
from pydantic import BaseModel
from google.genai import types
import json
import sqlite3
from config import client


class TradingDecision(BaseModel) :
    ticker_name: str
    decision :str
    confidence :int
    reason :str

def evaluate_price(pr: float, move_avg: float) -> str:
    if pr > move_avg:
        return "BUY"
    return "SELL/HOLD"


def generate_llm_prompt(ticker: str, current_price: float, avg_price: float, algo_signal: str,rsi:float ) -> str:
    # prompt = f"""
    # SYSTEM: You are an expert quantitative AI trading assistant.
    # MARKET DATA:
    # - Asset: {ticker}
    # - Current Price: {current_price:.2f}
    # - Moving Average: {avg_price:.2f}
    # - Baseline Algorithmic Signal: {algo_signal}
    #
    # TASK: Based on the market data provided above, the stock is currently
    # trading {'below' if current_price < avg_price else 'above'} its moving average.
    # Provide a brief, 2-sentence analysis on whether you agree with the baseline algorithmic signal and why.
    # """
    # return prompt
        prompt = f"""
        SYSTEM: You are an expert quantitative AI trading assistant. 
        MARKET DATA: 
        - Asset: {ticker}
        - Current Price: {current_price:.2f}
        - Moving Average: {avg_price:.2f}
        - 14-Day RSI (Relative Strength Index): {rsi}
        - Baseline Algorithmic Signal: {algo_signal}

        TASK: Based on the market data provided above, provide a brief analysis. 
        The RSI is {rsi} (Note: <30 is traditionally oversold, >70 is overbought). 
        Do you agree with the baseline algorithmic signal? 
        Respond in STRICT JSON format with exactly three keys: "decision" (Agree/Disagree), "confidence" (0-100), and "reason" (your 2-sentence explanation).
        """
        return prompt

# ticker_symbol = "RVNL.NS"
# print(f"Fetching data for {ticker_symbol}....")
#
# ticker = yf.Ticker(ticker_symbol)
# history = ticker.history(period="10d")
#
# latest_close = history['Close'].iloc[-1]
# avg_value = history['Close'].mean()
#
# signal = evaluate_price(latest_close, avg_value)

# final_prompt = generate_llm_prompt(ticker_symbol, latest_close, avg_value, signal)
#
# print("\n--- 🧠 SENDING TO AI FOR ANALYSIS ---")
#
# # Put your BRAND NEW API key here!
# client = genai.Client(api_key="AIzaSyBeag_l2F-U1EHLSyjorJcHjOtT7IAjbb8")
#
# # Change the model from 1.5 to 2.5!
# response = client.models.generate_content(
#     model='gemini-2.5-flash',
#     contents=final_prompt ,
#     config.py= types.GenerateContentConfig(
#         response_mime_type='application/json',response_schema=TradingDecision, )
# )
#
# # print("\n--- 🎯 AI VERDICT ---")
# # print(response.text)
#
#
# data = json.loads(response.text)
#
# conn = sqlite3.connect('ai_trading.db')
# cursor = conn.cursor()
#
# # Create table
# cursor.execute('''
#     CREATE TABLE IF NOT EXISTS predictions (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         ticker TEXT,
#         decision TEXT,
#         confidence INTEGER,
#         reason TEXT,
#         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
#     )
# ''')
#
# # Insert the data
# cursor.execute('''
#     INSERT INTO predictions (ticker, decision, confidence, reason)
#     VALUES (?, ?, ?, ?)
# ''', (data['ticker_name'], data['decision'], data['confidence'], data['reason']))
#
# conn.commit()
# print(f"Successfully saved {data['ticker_name']} prediction to database!")
#
# # Let's read it back just to prove it worked!
# cursor.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT 1')
# saved_row = cursor.fetchone()
# print(f"Read from DB: {saved_row}")
#

def calculate_rsi(history_data, window=14):
    """Calculates the Relative Strength Index (RSI)"""
    delta = history_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    # Handle division by zero
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Return the most recent RSI value, rounded to 2 decimals
    return round(rsi.iloc[-1], 2)

if __name__ == "__main__":
    ticker_symbol = "RVNL.NS" # Or LTIM.NS
    print(f"Fetching data for {ticker_symbol}....")

    ticker = yf.Ticker(ticker_symbol)
    # CHANGED: Fetching 1 month of data so we have enough days to calculate a 14-day RSI
    history = ticker.history(period="1mo")

    latest_close = history['Close'].iloc[-1]
    avg_value = history['Close'].mean()

    # Calculate our new indicator
    current_rsi = calculate_rsi(history)

    sign = evaluate_price(latest_close, avg_value)

    # Pass the new RSI value into the prompt generator
    final_prompt = generate_llm_prompt(ticker_symbol, latest_close, avg_value, sign, current_rsi)

    print("\n--- 🧠 SENDING TO AI FOR ANALYSIS ---")
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=final_prompt
    )

    print("\n--- 🎯 AI VERDICT ---")
    print(response.text)
