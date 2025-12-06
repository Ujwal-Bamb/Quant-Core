# models/train.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def make_features(df, lookback=20):
    X = pd.DataFrame(index=df.index)
    X['return_1'] = df['close'].pct_change(1)
    X['return_5'] = df['close'].pct_change(5)
    X['ma_5'] = df['close'].rolling(5).mean()
    X['ma_20'] = df['close'].rolling(20).mean()
    X['vol_10'] = df['close'].pct_change().rolling(10).std()
    X = X.fillna(0)
    return X

def make_targets(df, horizon=5):
    # binary: 1 if close in `horizon` bars ahead is higher
    target = (df['close'].shift(-horizon) > df['close']).astype(int)
    return target

def train_model(df, save_path="models/lgb_model.pkl"):
    X = make_features(df)
    y = make_targets(df)
    valid_idx = int(len(X)*0.8)
    X_train, X_test = X.iloc[:valid_idx], X.iloc[valid_idx:]
    y_train, y_test = y.iloc[:valid_idx], y.iloc[valid_idx:]
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)
    params = {
        "objective":"binary",
        "metric":"auc",
        "verbosity":-1,
        "boosting_type":"gbdt",
        "learning_rate":0.05,
        "num_leaves":31,
    }
    model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=200, early_stopping_rounds=20)
    joblib.dump(model, save_path)
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, preds)
    print("Saved model:", save_path, "AUC:", auc)
    return model

if __name__ == "__main__":
    import sys
    from tools.data_loader import download_ohlcv
    df = download_ohlcv("AAPL", period="60d", interval="5m")
    model = train_model(df, save_path="models/lgb_model.pkl")
