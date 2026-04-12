import yfinance as yf
from real_data import calculate_rsi, evaluate_price
from config import client


# def get_latest_news(ticker_symbol:str, count: int =3)-> str :
#     """Fetch and format the latest News Headline for the given Stock. """
#     ticker = yf.Ticker(ticker_symbol)
#     news_items = ticker.news
#
#     if not news_items:
#         return "No News Headlines Found"
#
#     formatted_news = ""
#
#     for i, item in enumerate(news_items[:count]):
#         title = item.get('title','No Title')
#         publisher = item.get('publisher','Unknown Publisher')
#         formatted_news += f"{i + 1}. {title}. {publisher}\n"
#         return formatted_news.strip()

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import ssl  # <-- NEW IMPORT


def get_latest_news(ticker_symbol: str, count: int = 3) -> str:
    """Fetches the latest news headlines using Google News RSS."""
    try:
        # Format the search query and hit the Google News India feed
        query = urllib.parse.quote(f"{ticker_symbol} stock")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

        # Google blocks standard python scripts, so we spoof a browser header
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

        # <-- NEW: Bypass SSL verification for this public feed -->
        ssl_context = ssl._create_unverified_context()

        # <-- NEW: Pass the context into urlopen -->
        with urllib.request.urlopen(req, context=ssl_context) as response:
            xml_data = response.read()

        # Parse the XML data
        root = ET.fromstring(xml_data)
        items = root.findall('.//item')

        if not items:
            return "No recent news available."

        formatted_news = ""
        for i, item in enumerate(items[:count]):
            title = item.find('title').text
            formatted_news += f"{i + 1}. {title}\n"

        return formatted_news.strip()

    except Exception as e:
        return f"Error fetching news: {e}"

def generate_llm_prompt(ticker: str, current_price: float, avg_price: float, algo_signal: str, rsi: float,
                        news_context: str) -> str:
    prompt = f"""
    SYSTEM: You are an expert quantitative AI trading assistant. 

    MARKET DATA: 
    - Asset: {ticker}
    - Current Price: {current_price:.2f}
    - Moving Average: {avg_price:.2f}
    - 14-Day RSI (Relative Strength Index): {rsi}
    - Baseline Algorithmic Signal: {algo_signal}

    LATEST NEWS CONTEXT:
    {news_context}

    TASK: Analyze the provided market data and recent news. 
    1. Consider if the news indicates a bullish or bearish catalyst.
    2. Weigh this against the technicals (RSI of {rsi} and price vs moving average).
    3. Do you agree with the baseline algorithmic signal? 

    Respond in STRICT JSON format with exactly three keys: 
    "decision" (Agree/Disagree), 
    "confidence" (0-100), 
    "reason" (A concise, 2-sentence explanation combining both technical and news analysis).
    """
    return prompt


def demo_fetch_and_prompt(ticker_symbol: str = "INFY.NS") -> str:
    """Helper function to fetch some recent history for a single ticker,
    compute the RSI and algorithmic signal, fetch news, and return the
    generated LLM prompt. This is safe to call from CLI for debugging.
    """
    print(f"Fetching data for {ticker_symbol}....")

    ticker = yf.Ticker(ticker_symbol)
    # Fetching 1 month of data so we have enough days to calculate a 14-day RSI
    history = ticker.history(period="1mo")

    latest_close = history['Close'].iloc[-1]
    avg_value = history['Close'].mean()

    # Calculate our new indicator
    current_rsi = calculate_rsi(history)

    signal = evaluate_price(latest_close, avg_value)

    # Fetch the news string
    recent_news = get_latest_news(ticker_symbol)

    # Pass it into the prompt generator (added as the last argument)
    final_prompt = generate_llm_prompt(ticker_symbol, latest_close, avg_value, signal, current_rsi, recent_news)
    return final_prompt


if __name__ == "__main__":
    # Simple demo when running this module directly
    prompt = demo_fetch_and_prompt()
    print(prompt)
