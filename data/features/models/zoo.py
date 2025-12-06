import lightgbm as lgb
import torch
import torch.nn as nn
import numpy as np

# --- 1. Tabular Model ---
class TabularModel:
    def __init__(self):
        self.model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

# --- 2. Sequence Model ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take last step
        return self.sigmoid(out)

# --- 3. Hybrid Ensemble ---
class HybridStrategy:
    def __init__(self):
        self.lgbm = TabularModel()
        self.lstm = None # Init in production
        self.regime_map = {0: 0.7, 1: 0.5, 2: 0.2} # Weight given to Trend model vs Mean Reversion

    def get_signal(self, tabular_feats, seq_feats, regime):
        # Mock logic for demo
        p_lgbm = self.lgbm.predict(tabular_feats)[0]
        # p_lstm = self.lstm(seq_feats).item() 
        
        # Regime gating
        weight = self.regime_map.get(regime, 0.5)
        
        final_prob = p_lgbm # Simplified
        return final_prob
