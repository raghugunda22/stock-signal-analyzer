"""Main runner for the ARAMB pipeline.

Steps performed:
1. Fetch historical data for the WATCHLIST
2. Add features using `features.add_features`
3. Train or load XGBoost model (via `ml_model`)
4. For each ticker: predict signal, fetch news, call Gemini (with safe fallback), save to DB
5. Print a summary / accuracy report
"""

from typing import Optional, Dict
import json
import traceback

import yfinance as yf
import pandas as pd

from config import WATCHLIST, LOOKBACK_PERIOD, PREDICTION_DAYS, client
from features import add_features
from ml_model import load_or_train, predict_signal
from real_data_RAG import get_latest_news, generate_llm_prompt
from database import init_db, save_signal, get_accuracy_report


def try_call_gemini(prompt: str) -> Optional[Dict]:
    """Attempt to call the configured Gemini client and parse a JSON response.
    This function is defensive: it tries a couple of common access patterns and
    falls back to None if anything goes wrong.
    """
    try:
        # Many genai client versions expose `responses.create` that returns a
        # structured object. We try to call it and then extract a textual
        # rendering to parse as JSON.
        resp = client.responses.create(input=prompt)

        # Try some common attributes to find the LLM text output
        text = None
        if hasattr(resp, "output_text"):
            text = resp.output_text
        elif hasattr(resp, "text"):
            text = resp.text
        else:
            # Dive into nested structures if present
            out = getattr(resp, "output", None)
            if out and isinstance(out, (list, tuple)):
                parts = []
                for item in out:
                    if isinstance(item, dict):
                        parts.append(item.get("content", ""))
                text = "\n".join(parts)

        if not text:
            text = str(resp)

        # Extract the first JSON object substring from the text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start:end+1]
            return json.loads(snippet)
    except Exception:
        # Don't crash the whole run if the LLM call fails; return None and
        # let the caller use a fallback heuristic.
        # Uncomment for debugging: print(traceback.format_exc())
        pass
    return None


def heuristic_gemini(algo_signal: str, rsi: float, news: str) -> Dict:
    """Very small heuristic to produce a Gemini-like decision when the API
    is unavailable.
    """
    lower = (news or "").lower()
    bullish_kw = ["upgrade", "beat", "profit", "positive", "gain", "gains", "launch", "partner", "acquire"]
    bearish_kw = ["downgrade", "loss", "lawsuit", "decline", "negative", "miss", "fall", "drop", "weak"]

    news_bull = any(k in lower for k in bullish_kw)
    news_bear = any(k in lower for k in bearish_kw)

    if news_bull and algo_signal == "BUY":
        decision = "Agree"
    elif news_bear and algo_signal == "SELL":
        decision = "Agree"
    elif not news_bull and not news_bear:
        # No clear news signal — default to agreeing with the model
        decision = "Agree"
    else:
        decision = "Disagree"

    reason = f"Technicals: 14-day RSI {rsi:.1f}. News indicates {'bullish' if news_bull else ('bearish' if news_bear else 'no clear catalyst')}."
    return {"decision": decision, "confidence": 60, "reason": reason}


def build_and_train_model() -> Optional[object]:
    """Fetch history for all tickers, build features, and load/train the model."""
    all_data = []
    for ticker in WATCHLIST:
        try:
            df = yf.Ticker(ticker).history(period=LOOKBACK_PERIOD)
            df = add_features(df)
            if not df.empty:
                # Keep the ticker as an identifier when combining
                df = df.copy()
                df['Ticker'] = ticker
                all_data.append(df)
        except Exception:
            print(f"Warning: failed to fetch or process {ticker}:\n{traceback.format_exc()}")

    if not all_data:
        print("No data available to train/load model.")
        return None

    combined = pd.concat(all_data, ignore_index=True)
    model = load_or_train(combined)
    return model


def run():
    print("Initializing DB...")
    init_db()

    print("Building or loading model (this may train if no model file exists)...")
    model = build_and_train_model()
    if model is None:
        print("Failed to obtain or train a model — exiting.")
        return

    # For each ticker, get fresh data, predict and ask Gemini
    for ticker in WATCHLIST:
        try:
            print(f"\n--- Processing {ticker} ---")
            df = yf.Ticker(ticker).history(period=LOOKBACK_PERIOD)
            df = add_features(df)
            if df.empty:
                print(f"No feature data for {ticker}, skipping.")
                continue

            xgb_result = predict_signal(model, df)
            latest_close = float(df['Close'].iloc[-1])
            current_rsi = float(df['RSI14'].iloc[-1])

            # Fetch recent news (may be a plain text string or an error message)
            news = get_latest_news(ticker)

            # Build prompt and try to call Gemini (with a fallback)
            prompt = generate_llm_prompt(ticker, latest_close, df['Close'].mean(), xgb_result['signal'], current_rsi, news)
            gemini_resp = try_call_gemini(prompt)
            if gemini_resp is None:
                gemini_result = heuristic_gemini(xgb_result['signal'], current_rsi, news)
            else:
                # Safe extraction — the RAG prompt requests keys: decision, confidence, reason
                gemini_result = {
                    'decision': gemini_resp.get('decision', 'Agree'),
                    'confidence': gemini_resp.get('confidence', 50),
                    'reason': gemini_resp.get('reason', '')
                }

            # Save to database
            save_signal(ticker, latest_close, xgb_result, gemini_result)

            # Print a human-friendly summary
            print(f"Ticker:       {ticker}")
            print(f"Close:        {latest_close:.2f}")
            print(f"XGB Signal:   {xgb_result['signal']} ({xgb_result['confidence']}% confidence)")
            print(f"Gemini:       {gemini_result['decision']} ({gemini_result.get('confidence')}%)")
            print(f"Reason:       {gemini_result.get('reason')}" )

        except Exception:
            print(f"Error while processing {ticker}:\n{traceback.format_exc()}")

    # Summary report
    report = get_accuracy_report()
    print("\n=== Summary ===")
    print(f"Total signals: {report['total']}")
    print(f"Agreements:    {report['agreements']}")
    print(f"Agreement %:   {report['percentage']}%")


if __name__ == "__main__":
    run()
