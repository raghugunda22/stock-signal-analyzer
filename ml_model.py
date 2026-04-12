import joblib
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score

from features import FEATURE_COLS

MODEL_PATH = Path("models/xgb_stock.pkl")
MODEL_PATH.parent.mkdir(exist_ok=True)

SIGNAL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}

def train_model(df):
    X = df[FEATURE_COLS]
    y = df['Target']

    # Time-aware split — 80% train, 20% test
    split   = int(len(X) * 0.80)
    X_train = X.iloc[:split]
    X_test  = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test  = y.iloc[split:]
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds,
                                target_names=['SELL', 'HOLD', 'BUY']))

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved → {MODEL_PATH}")

    return model

def load_or_train(df):
    if MODEL_PATH.exists():
        print("📂 Loading existing model...")
        return joblib.load(MODEL_PATH)
    print("🏋️  No model found — training fresh...")
    return train_model(df)

def predict_signal(model, df):
    # Take only the last row — today's data
    latest = df[FEATURE_COLS].iloc[[-1]]

    proba      = model.predict_proba(latest)[0]
    signal_idx = int(np.argmax(proba))
    confidence = float(proba[signal_idx])

    return {
        "signal":     SIGNAL_MAP[signal_idx],
        "confidence": round(confidence * 100, 1),
        "buy_prob":   round(float(proba[2]) * 100, 1),
        "hold_prob":  round(float(proba[1]) * 100, 1),
        "sell_prob":  round(float(proba[0]) * 100, 1),
    }
def get_model():
    import yfinance as yf
    import pandas as pd
    from config import WATCHLIST
    from features import add_features

    if MODEL_PATH.exists():
        print("📂 Loading existing model...")
        return joblib.load(MODEL_PATH)

    print("🏋️  Training fresh model...")
    all_data = []
    for ticker in WATCHLIST:
        df = yf.Ticker(ticker).history(period="2y")
        df = add_features(df)
        all_data.append(df)

    combined = pd.concat(all_data)
    return train_model(combined)

if __name__ == "__main__":
    import yfinance as yf
    import pandas as pd
    from features import add_features
    from config import WATCHLIST

    # Fetch all stocks and combine
    all_data = []
    for ticker in WATCHLIST:
        df = yf.Ticker(ticker).history(period="2y")
        df = add_features(df)
        all_data.append(df)

    # Combine all into one big DataFrame
    combined = pd.concat(all_data)
    print(f"Combined shape: {combined.shape}")
    print(f"Target counts:\n{combined['Target'].value_counts()}")

    model = train_model(combined)
    test_df = yf.Ticker("INFY.NS").history(period="2y")
    test_df = add_features(test_df)
    result = predict_signal(model, test_df)

    print(f"\nINFY Signal Today:")
    print(f"Signal:     {result['signal']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"BUY:        {result['buy_prob']}%")
    print(f"HOLD:       {result['hold_prob']}%")
    print(f"SELL:       {result['sell_prob']}%")


