import sqlite3
from datetime import datetime

DB_PATH = "stock_signals.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id              INTEGER PRIMARY KEY,
            ticker          TEXT,
            date            TEXT,
            close_price     REAL,
            xgb_signal      TEXT,
            xgb_confidence  REAL,
            gemini_decision TEXT,
            gemini_reason   TEXT,
            signals_agree   INTEGER,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_signal(ticker, close_price, xgb_result, gemini_result):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        INSERT INTO signals 
        (ticker, date, close_price, 
         xgb_signal, xgb_confidence,
         gemini_decision, gemini_reason, signals_agree)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        ticker,
        datetime.now().strftime('%Y-%m-%d'),
        close_price,
        xgb_result['signal'],
        xgb_result['confidence'],
        gemini_result.get('decision', ''),
        gemini_result.get('reason', ''),
        1 if gemini_result['decision'] == 'Agree' else 0
    ))
    conn.commit()
    conn.close()

def get_accuracy_report():
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute('''
        SELECT 
            COUNT(*)                 as total,
            SUM(signals_agree)       as agreements,
            ROUND(AVG(signals_agree) * 100, 1) as percentage
        FROM signals
    ''').fetchone()
    conn.close()
    return {
        'total':      row[0],
        'agreements': row[1],
        'percentage': row[2]
    }

if __name__ == "__main__":
    init_db()
    # # Test data — fake results to test saving
    # test_xgb = {'signal': 'BUY', 'confidence': 73.5}
    # test_gemini = {'decision': 'Agree', 'reason': 'RSI oversold with positive news'}
    #
    # save_signal('INFY.NS', 1278.30, test_xgb, test_gemini)
    # Read it back and print
    report = get_accuracy_report()
    print(f"Total signals: {report['total']}")
    print(f"Agreements:    {report['agreements']}")
    print(f"Agreement %:   {report['percentage']}%")
    print("Signal saved Successfully")

