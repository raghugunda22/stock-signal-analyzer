# config.py
import os

# Try to import the Google GenAI client; make this defensive so the rest of
# the codebase can run even if the package or API key is missing.
try:
	from google import genai
	GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
	client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
except Exception:
	client = None


# Your watchlist in one place
WATCHLIST = ["IRFC.NS","RVNL.NS", "INFY.NS", "RELIANCE.NS", "TCS.NS"]

# Settings
LOOKBACK_PERIOD = "3mo"   # how much history to fetch
PREDICTION_DAYS = 5       # predict price direction N days ahead
