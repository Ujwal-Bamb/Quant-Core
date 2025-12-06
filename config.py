import os

class Config:
    # System
    mode = os.getenv("MODE", "BACKTEST")  # BACKTEST, LIVE, PAPER
    
    # Risk
    MAX_DRAWDOWN_LIMIT = 0.15
    POSITION_LIMIT = 0.05  # 5% of equity per trade
    
    # Data
    TICKERS = ["SPY", "QQQ"]
    USE_SYNTHETIC = True  # Fallback if no API keys
    
    # APIs (Set via env vars)
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    
    # Model
    SEQ_LEN = 60  # Time steps for LSTM
    PREDICT_HORIZON = 5  # Bars ahead
