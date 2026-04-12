import numpy as np




def add_features(df):
    close = df['Close']

    # RSI - you know exactly what each line does now!
    delta = close.diff()
    gain =delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs  = avg_gain / avg_loss
    df['RSI14'] = 100 - 100 / (1 + rs)
    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df['MACD'] =ema12 - ema26
    # Bollinger Bands
    middle = df["Close"].rolling(20).mean()
    upper = middle + (2 * df["Close"].rolling(20).std())
    lower = middle - (2 * df["Close"].rolling(20).std())

    df['BB_width'] = (upper - lower)/ middle
    df['BB_pos'] =(close-lower)/(upper-lower)

    df['Return_1d'] = close.pct_change(1)
    df['Return_5d'] = close.pct_change(5)
    df['Return_20d'] = close.pct_change(20)

    future_return = close.pct_change(5).shift(-5)
    df['Target'] = np.where(future_return > 0.015, 2,
                            np.where(future_return < -0.015, 0,
                                     1))

    return df.dropna()
FEATURE_COLS = [
    'RSI14',
    'MACD',
    'BB_width',
    'BB_pos',
    'Return_1d',
    'Return_5d',
    'Return_20d'
]