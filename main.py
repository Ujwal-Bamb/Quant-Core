import pandas as pd
import numpy as np
from data.synthetic import SyntheticMarket
from features.regimes import RegimeDetector
from models.zoo import TabularModel
from execution.backtester import EventDrivenBacktester

# --- Mock Strategy for Backtest ---
class SimpleStrategy:
    def __init__(self, model, regime_detector):
        self.model = model
        self.rd = regime_detector
        
    def evaluate(self, price, chain):
        # 1. Feature Engineering (Simplified)
        feats = np.array([[price, 0.2]]) # Price, Vol
        
        # 2. Regime
        regime = self.rd.predict(pd.Series([price])) # Mock series
        
        # 3. Prediction
        prob = self.model.predict(feats)[0]
        
        if prob > 0.6 and regime != 2: # Don't buy in crash
            return {'action': 'BUY_CALL'}
        return {'action': 'HOLD'}

def run_demo():
    print("🔵 [INIT] QuantCore System Startup...")
    
    # 1. Generate Training Data
    print("🔵 [DATA] Generating synthetic historical data...")
    market = SyntheticMarket()
    history = [market.get_tick() for _ in range(1000)]
    df_hist = pd.DataFrame(history, columns=['close'])
    df_hist['target'] = (df_hist['close'].shift(-5) > df_hist['close']).astype(int)
    df_hist.dropna(inplace=True)
    
    # 2. Train Models
    print("🔵 [TRAIN] Fitting Regime Detector & LightGBM...")
    rd = RegimeDetector()
    rd.fit(df_hist['close'].pct_change().dropna())
    
    model = TabularModel()
    model.train(df_hist[['close', 'target']].iloc[:, :-1], df_hist['target'])
    
    # 3. Setup Backtest
    print("🔵 [EXEC] Starting Event-Driven Backtest...")
    bt = EventDrivenBacktester()
    strategy = SimpleStrategy(model, rd)
    
    # Simulation Generator
    def market_feed():
        for _ in range(50):
            p = market.get_tick()
            chain = market.generate_option_chain(p, '2023-12-01')
            yield (pd.Timestamp.now(), p, chain)
            
    bt.run(market_feed(), strategy)
    
    print("✅ [DONE] System Demo Finished.")

if __name__ == "__main__":
    run_demo()
