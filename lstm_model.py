import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from features import add_features, FEATURE_COLS
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def create_sequences(X, y,seq_length=20) :
    Xs , ys = [],[]

    for i in range(len(X) - seq_length):
        Xs.append(X[i : i + seq_length])
        ys.append([y[i +seq_length]])

    return np.array(Xs), np.array(ys).squeeze()


def train_lstm(model, loader, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in loader:
            # 1. Forward pass
            output = model(X_batch)

            # 2. Calculate loss
            loss = criterion(output, y_batch)

            # 3. Backward pass
            optimizer.zero_grad()
            loss.backward()

            # 4. Update weights
            optimizer.step()

            total_loss += loss.item()

        # Print every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch + 1}/{epochs} Loss: {avg_loss:.4f}")

    return model


def predict_lstm(model, scaler, df):
    # Get last 20 days of features
    X = df[FEATURE_COLS].values  # get feature columns
    X_scaled = scaler.transform(X)  # scale using fitted scaler

    # Take last 20 rows only
    X_seq = X_scaled[-20:]  # hint: last 20 rows

    # Reshape to [1, 20, 7] — batch of 1 sequence
    X_tensor = torch.tensor(
        X_seq.reshape(1, 20, 7),  # hint: 20 days, 7 features
        dtype=torch.float32)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(X_tensor)
        proba = torch.softmax(output, dim=1)[0].numpy()

    signal_idx = int(np.argmax(proba))
    SIGNAL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}

    return {
        "signal": SIGNAL_MAP[signal_idx],
        "confidence": round(float(proba[signal_idx]) * 100, 1),
        "sell_prob": round(float(proba[0]) * 100, 1),
        "hold_prob": round(float(proba[1]) * 100, 1),
        "buy_prob": round(float(proba[2]) * 100, 1),
    }

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    import yfinance as yf
    import os
    from features import add_features, FEATURE_COLS
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    from config import WATCHLIST

    # Step 1: Load ALL watchlist stocks
    all_X, all_y = [], []
    for ticker in WATCHLIST:
        df = yf.Ticker(ticker).history(period="2y")
        df = add_features(df)
        all_X.append(df[FEATURE_COLS].values)
        all_y.append(df['Target'].values.ravel())

    X_combined = np.concatenate(all_X)
    y_combined = np.concatenate(all_y)
    print(f"Combined: {X_combined.shape}")

    # Step 2: Scale features
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # Step 3: Create sequences
    Xs, ys = create_sequences(X_scaled, y_combined, seq_length=20)

    # Step 4: Time-aware split
    split           = int(len(Xs) * 0.80)
    X_train, X_test = Xs[:split], Xs[split:]
    y_train, y_test = ys[:split], ys[split:]
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Step 5: DataLoader — train only
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor  = torch.tensor(y_train, dtype=torch.long)
    dataset  = TensorDataset(X_tensor, y_tensor)
    loader   = DataLoader(dataset, batch_size=32, shuffle=False)

    # Step 6: Train LSTM
    model = StockLSTM(input_size=7, hidden_size=128,
                      num_layers=2, num_classes=3)
    model = train_lstm(model, loader, epochs=50)

    # Step 7: Evaluate on TEST set
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(X_test_tensor)
        preds  = torch.argmax(output, dim=1).numpy()

    print(classification_report(y_test, preds,
          target_names=['SELL', 'HOLD', 'BUY']))

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/lstm_stock.pt')
    print("✅ LSTM model saved!")
    # Test predict_lstm on INFY
    test_df = yf.Ticker("INFY.NS").history(period="3mo")
    test_df = add_features(test_df)

    result = predict_lstm(model, scaler, test_df)
    print(f"\nINFY LSTM Signal Today:")
    print(f"Signal:     {result['signal']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"BUY:        {result['buy_prob']}%")
    print(f"HOLD:       {result['hold_prob']}%")
    print(f"SELL:       {result['sell_prob']}%")