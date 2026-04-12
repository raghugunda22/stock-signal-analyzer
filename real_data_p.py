import yfinance as yf
import matplotlib.pyplot as plt

def evaluate_price(pr: float, move_avg: float) -> str:
    if pr > move_avg:
        return "BUY"
    return "SELL/HOLD"



ticker_symbol = "RVNL.NS"

print(f"Fetching data for {ticker_symbol}....")

ticker = yf.Ticker(ticker_symbol)

history = ticker.history(period="100d")
print("Last 100 Days of Trading ")
latest_close=history['Close'].iloc[-1]
avg_value=history['Close'].mean()
signal = evaluate_price(latest_close, avg_value)
# print(f"Average Value of {ticker_symbol} is {avg_value:.2f} and Last Closed value is {latest_close:.2f} ")
# print(f"Signal: {signal}")
#
#
# print("Generating chart...")
#
# # 1. Plot the actual closing prices
# history['Close'].plot(title=f"{ticker_symbol} - 100 Day Trend", figsize=(10, 5), label="Close Price")
#
# # 2. Draw a red dashed line for your average!
# plt.axhline(y=avg_value, color='red', linestyle='--', label='100-Day Avg')
#
# # 3. Add a legend and show the window
# plt.legend()
# plt.show()


def generate_llm_prompt(ticker_p:str, current_price: float, avg_price:float, algo_signal: str) -> str:
    prompt =f"""
    SYSTEM: You are an expert quantitative AI trading assistant.
    MARKET DATA: 
    - Asset: {ticker_p}
    - Current Price: {current_price:.2f}
    - Moving Average: {avg_price:.2f}
    - Baseline Algorithmic Signal: {algo_signal}
    
    TASK: Based on the market data provided above, the stock is currently trading 
{'below' if current_price < avg_price else 'above'} its moving average. Provide a brief, 2-sentence analysis on whether you agree with the baseline algorithmic signal and why.
    """
    return prompt

final_prompt = generate_llm_prompt(ticker_symbol, latest_close, avg_value, signal)

print("\n--- 🤖 GENERATED LLM PROMPT ---")
print(final_prompt)