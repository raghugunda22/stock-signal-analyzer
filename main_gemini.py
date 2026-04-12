from config import WATCHLIST,client        # what do you need from config?
from features import add_features     # what function?
from database import save_signal  # what functions?
from real_data_RAG import  get_latest_news
import  yfinance as yf
import json
import re
from ml_model import  predict_signal, get_model
from database import init_db,get_accuracy_report
import time

init_db()
model = get_model()


def build_prompt(ticker, close_price, xgb_result, news):
    return f"""
    SYSTEM: You are an expert quantitative AI trading assistant.

    ASSET: {ticker}
    CURRENT PRICE: {close_price:.2f}

    ML MODEL SIGNAL:
    - XGBoost says: {xgb_result['signal']}
    - Confidence:   {xgb_result['confidence']}

    LATEST NEWS:
    {news}

    TASK: Explain WHY the XGBoost signal makes sense given the news.
    Does the news support or contradict the signal?

    Respond in STRICT JSON:
    {{"decision": "Agree/Disagree", "confidence": 0-100, "reason": "2 sentences"}}
    """

def parse_gemini_response(raw_text):
    try:
        clean  = re.sub(r'```json|```', '', raw_text).strip()
        result = json.loads(clean)
        return result
    except Exception:
        return {"decision": "Unknown", "confidence": 0, "reason": raw_text}

for ticker in WATCHLIST:
    print(f"\n{'-'*50}")
    print(f"📈 Processing {ticker}")

    try:
        df= yf.Ticker(ticker).history(period="2y")
        df = add_features(df)
        close_price = df['Close'].iloc[-1]
        xgb_result=predict_signal(model,df)
        news = get_latest_news(ticker)
        gemini_prompt = build_prompt(ticker, close_price, xgb_result, news)
        response = client.models.generate_content(
            model='gemini-2.5-flash', contents=gemini_prompt)
        gemini_response = parse_gemini_response(response.text)

        save_signal(ticker,close_price,xgb_result,gemini_response)

        print(f"XGBoost: {xgb_result['signal']} ({xgb_result['confidence']}%)")
        print(f"Gemini:  {gemini_response.get('decision')} — {gemini_response.get('reason', '')[:80]}")

        time.sleep(15)

    except Exception as e:
        print(f"{ticker} failed:{e}")
report = get_accuracy_report()
print(f"\n📊 Total signals: {report['total']}")
print(f"📊 Agreement:     {report['percentage']}%")
