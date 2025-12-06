import pandas as pd

class EventDrivenBacktester:
    def __init__(self, initial_capital=100000):
        self.cash = initial_capital
        self.positions = {}
        self.history = []
        self.slippage = 0.01 # $0.01 per share/contract
        self.commission = 0.65 # $0.65 per contract

    def on_fill(self, symbol, quantity, price, side):
        cost = (price * quantity * 100) # Options multiplier
        fee = self.commission * quantity
        
        if side == 'BUY':
            total_cost = cost + fee + (self.slippage * quantity)
            if self.cash >= total_cost:
                self.cash -= total_cost
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        elif side == 'SELL':
            total_revenue = cost - fee - (self.slippage * quantity)
            self.cash += total_revenue
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity

    def run(self, market_data_feed, strategy):
        """
        market_data_feed: Generator yielding (timestamp, tick_data, chain)
        """
        print(f"Starting Backtest. Capital: ${self.cash}")
        
        for timestamp, tick, chain in market_data_feed:
            # 1. Update Portfolio Value
            # 2. Get Strategy Signal
            signal = strategy.evaluate(tick, chain)
            
            # 3. Execute
            if signal['action'] == 'BUY_CALL':
                # Find ATM call
                atm_call = chain.loc[abs(chain['strike'] - tick).idxmin()]
                self.on_fill(f"CALL_{atm_call['strike']}", 1, atm_call['ask'], 'BUY')
                
        print(f"Backtest Complete. Capital: ${self.cash}")
