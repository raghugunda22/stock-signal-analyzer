# My Project Context

## What I'm building
Stock Market Signal Analyzer — hybrid XGBoost + Gemini LLM

## Background
Senior Java Backend Developer, transitioning to ML/AI Engineer
~3 weeks into Python/ML

## Completed files
- config.py      ✅ API keys, watchlist, settings
- features.py    ✅ RSI, MACD, BB_width, BB_pos, Returns, Target, FEATURE_COLS
- database.py    ✅ init_db(), save_signal(), get_accuracy_report()
- ml_model.py    ✅ train_model(), load_or_train(), predict_signal()

## Currently working on
- main.py ← NEXT FILE to build

## main.py needs to do
1. Fetch stock data (yfinance)
2. Add features (features.py)
3. XGBoost predicts (ml_model.py)
4. Fetch news (real_data_RAG.py)
5. Gemini explains WHY (config.py client)
6. Parse Gemini JSON response
7. Save both signals to SQLite (database.py)
8. Print summary

## Model performance
- 4 stocks, 2yr data, 1904 rows
- Accuracy: 45%, HOLD f1=0.56

## Roadmap
Month 1 ✅ → Month 2 (now) → Month 3 LSTM → 
Month 4 MLOps → Month 5 Deploy → Month 6 Jobs