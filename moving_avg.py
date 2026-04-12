from datetime import datetime as dt
import time
import yfinance as yf

def evaluate_price(pr: float, move_avg: float) -> str:
    if pr > move_avg:
        return "BUY"
    return "SELL/HOLD"

signal = evaluate_price(105.0,103.0)
print(f"Price signal: {signal}")


prices = [101.0,105.0,107.0,103.0,108.0]
moving_average = 103.0
for price in prices :
    current_time = dt.now()
    print("Checking Price....")
    time.sleep(1)
    result = evaluate_price(price, moving_average)
    print(f"Time [{current_time}] Price {price} is : {result}")
    print("Checking Next Price....")