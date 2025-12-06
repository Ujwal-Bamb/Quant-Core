from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np

class RegimeDetector:
    def __init__(self):
        self.model = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
        self.fitted = False

    def fit(self, returns):
        # Features: Volatility and Returns
        data = pd.DataFrame()
        data['ret'] = returns
        data['vol'] = returns.rolling(20).std()
        data = data.dropna()
        
        self.model.fit(data)
        self.fitted = True
        
    def predict(self, recent_returns):
        if not self.fitted: return 0
        vol = recent_returns.std()
        ret = recent_returns.mean()
        # 0: Low Vol/Bull, 1: High Vol/Correction, 2: Crash
        return self.model.predict([[ret, vol]])[0]
