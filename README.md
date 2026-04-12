# Stock Signal Analyzer

A hybrid ML + LLM system that generates Buy/Hold/Sell signals 
for Indian stocks using a trained XGBoost model and Google Gemini AI.

## What It Does

- Fetches real stock data from Yahoo Finance (yfinance)
- Engineers 7 technical features: RSI, MACD, Bollinger Bands, Returns
- Trains an XGBoost classifier on 2 years of historical data (80/20 split)
- Generates Buy/Hold/Sell signal with confidence score
- Fetches latest news headlines as RAG context
- Uses Gemini LLM to explain WHY the signal makes sense
- Saves both signals and agreement score to SQLite database

## Tech Stack

- Python, Pandas, NumPy
- XGBoost, Scikit-learn
- Google Gemini API
- yfinance, SQLite

## Model Performance

- Trained on 4 stocks, 1904 rows, 2 years of data
- Accuracy: 45% (vs 33% random baseline)
- HOLD F1: 0.56

## Stocks Covered

IRFC.NS, RVNL.NS, INFY.NS, RELIANCE.NS, TCS.NS