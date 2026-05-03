import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)
from config import WATCHLIST, client
from features import add_features, FEATURE_COLS
from database import save_signal, init_db, get_accuracy_report
from real_data_RAG import get_latest_news
import yfinance as yf
import json
import re
import time
import numpy as np
import pandas as pd
from ml_model import predict_signal, get_model
from lstm_model import predict_lstm, StockLSTM
from sklearn.preprocessing import StandardScaler

# print("FEATURE_COLS:", FEATURE_COLS)
# ── Setup ──────────────────────────────────────────
init_db()
model = get_model()

# Load LSTM
lstm = StockLSTM(input_size=7, hidden_size=128,
                 num_layers=2, num_classes=3)
lstm.load_state_dict(
    torch.load('models/lstm_stock.pt',
               map_location='cpu',
               weights_only=True)
)
lstm.eval()
print("📂 LSTM model loaded!")

# Fetch all data ONCE — reuse everywhere
print("📡 Fetching stock data...")
stock_data = {}
for ticker in WATCHLIST:
    df = yf.Ticker(ticker).history(period="2y")
    stock_data[ticker] = add_features(df)
    print(f"✅ Fetched {ticker}")

# Fit scaler using cached data
all_X = [stock_data[t][FEATURE_COLS].values for t in WATCHLIST]
scaler = StandardScaler()
scaler.fit(np.concatenate(all_X))
print("📊 Scaler fitted!")


# ── Helper functions ───────────────────────────────
def build_prompt(ticker, close_price, xgb_result, lstm_result, news):
    return f"""
    SYSTEM: You are an expert quantitative AI trading assistant.

    ASSET: {ticker}
    CURRENT PRICE: {close_price:.2f}

    ML MODEL SIGNALS:
    - XGBoost says: {xgb_result['signal']} ({xgb_result['confidence']}% confidence)
    - LSTM says:    {lstm_result['signal']} ({lstm_result['confidence']}% confidence)
    - They agree:   {xgb_result['signal'] == lstm_result['signal']}

    LATEST NEWS:
    {news}

    TASK: Explain WHY the signals make sense given the news.
    Do the models agree? Does news support or contradict?

    Respond in STRICT JSON:
    {{"decision": "Agree/Disagree", "confidence": 0-100, "reason": "2 sentences"}}
    """

def parse_gemini_response(raw_text):
    try:
        clean  = re.sub(r'```json|```', '', raw_text).strip()
        return json.loads(clean)
    except Exception:
        return {"decision": "Unknown", "confidence": 0, "reason": raw_text}


# ── Main loop ──────────────────────────────────────
for ticker in WATCHLIST:
    print(f"\n{'-'*50}")
    print(f"📈 Processing {ticker}")

    try:
        df          = stock_data[ticker]      # ← use cached data
        close_price = float(df['Close'].iloc[-1])

        xgb_result  = predict_signal(model, df)
        lstm_result = predict_lstm(lstm, scaler, df)

        print(f"XGBoost: {xgb_result['signal']} ({xgb_result['confidence']}%)")
        print(f"LSTM:    {lstm_result['signal']} ({lstm_result['confidence']}%)")

        news           = get_latest_news(ticker)
        gemini_prompt  = build_prompt(ticker, close_price,
                                      xgb_result, lstm_result, news)
        response       = client.models.generate_content(
            model='gemini-2.5-flash', contents=gemini_prompt)
        gemini_response = parse_gemini_response(response.text)

        save_signal(ticker, close_price, xgb_result, gemini_response)
        print(f"Gemini:  {gemini_response.get('decision')} — "
              f"{gemini_response.get('reason','')[:80]}")

        time.sleep(15)

    except Exception as e:
        print(f"❌ {ticker} failed: {e}")

report = get_accuracy_report()
print(f"\n📊 Total signals: {report['total']}")
print(f"📊 Agreement:     {report['percentage']}%")