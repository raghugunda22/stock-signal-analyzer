import yfinance as yf
from google import genai

from real_data import calculate_rsi, evaluate_price
from real_data_RAG import get_latest_news
from config import client
import json

def generate_llm_prompt(ticker: str, current_price: float, avg_price: float, algo_signal: str, rsi: float,
                        news_context: str) -> str:
    # prompt = f"""-
    # SYSTEM: You are an expert quantitative AI trading assistant.
    #
    # MARKET DATA:
    # - Asset: {ticker}
    # - Current Price: {current_price:.2f}
    # - Moving Average: {avg_price:.2f}
    # - 14-Day RSI (Relative Strength Index): {rsi}
    # - Baseline Algorithmic Signal: {algo_signal}
    #
    # LATEST NEWS CONTEXT:
    # {news_context}
    #
    # TASK: Analyze the provided market data and recent news.
    # 1. Consider if the news indicates a bullish or bearish catalyst.
    # 2. Weigh this against the technicals (RSI of {rsi} and price vs moving average).
    # 3. Do you agree with the baseline algorithmic signal?
    #
    # Respond in STRICT JSON format with exactly three keys:
    # "decision" (Agree/Disagree),
    # "confidence" (0-100),
    # "reason" (A concise, 2-sentence explanation. YOU MUST EXPLICITLY CITE AT LEAST ONE NEWS HEADLINE IN YOUR REASONING).
    # """
    prompt = f"""
    You are an expert stock market analyst. I will provide you with a baseline algorithmic trading signal, technical indicators, and recent news headlines for a stock.

    Stock Ticker: {ticker}
    Algorithmic Signal: {algo_signal}
    Current Price: {current_price}
    Moving Average: {avg_price}
    14-Day RSI: {rsi}

    Recent News Headlines:
    {news_context}

    Task:
    Analyze the technical indicators alongside the sentiment of the recent news headlines. 
    Do you 'Agree' or 'Disagree' with the baseline algorithmic signal?

    Provide your response strictly in JSON format with the following keys:
    - "decision": "Agree" or "Disagree"
    - "confidence": an integer between 0 and 100
    - "reason": A brief explanation of why, referencing both technicals and news sentiment if applicable.
    """
    return prompt



watchlist = ["RVNL.NS", "INFY.NS", "IRFC.NS","LT.NS"]

for ticker_symbol in watchlist:
    print(f"\nFetching data for {ticker_symbol}....")

    try:
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(period="1mo")

        if history.empty:
            continue

        latest_close = history['Close'].iloc[-1]
        avg_value = history['Close'].mean()
        current_rsi = calculate_rsi(history)
        signal = evaluate_price(latest_close, avg_value)

        # 1. Fetch News
        recent_news = get_latest_news(ticker_symbol)
        print(f"\n📰 FETCHED NEWS FOR {ticker_symbol}:\n{recent_news}\n")

        # 2. Generate Prompt
        final_prompt = generate_llm_prompt(ticker_symbol, latest_close, avg_value, signal, current_rsi, recent_news)

        # 3. Call AI
        print("--- 🧠 SENDING TO AI FOR ANALYSIS ---")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=final_prompt
        )

        # 4. Print Verdict
        print(f"--- 🎯 AI VERDICT FOR {ticker_symbol} ---")
        raw_response = response.text
        cleaned_response = raw_response.replace("```json", "").replace("```", "").strip()

        try:
            verdict_json = json.loads(cleaned_response)
            print(f"--- 🎯 AI VERDICT FOR {ticker_symbol} ---")
            print(f"Decision: {verdict_json.get('decision')}")
            print(f"Confidence: {verdict_json.get('confidence')}")
            print(f"Reason: {verdict_json.get('reason')}")
        except json.JSONDecodeError:
            print(f"⚠️ Could not parse JSON for {ticker_symbol}. Raw output:\n{raw_response}")

    except Exception as e:
        print(f"❌ Failed to process {ticker_symbol}: {e}")