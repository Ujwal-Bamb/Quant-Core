import numpy as np
import pandas as pd
from scipy.stats import norm

class SyntheticMarket:
    def __init__(self, start_price=400, vol=0.2):
        self.price = start_price
        self.vol = vol
        self.dt = 1/252/390  # 1 minute step

    def get_tick(self):
        # Geometric Brownian Motion
        drift = 0.0002 # Slight upward drift
        shock = np.random.normal(0, 1)
        self.price *= np.exp((drift - 0.5 * self.vol**2) * self.dt + self.vol * np.sqrt(self.dt) * shock)
        return self.price

    def generate_option_chain(self, price, date):
        strikes = np.linspace(price * 0.95, price * 1.05, 10)
        chain = []
        for K in strikes:
            # Simple Black Scholes for pricing
            d1 = (np.log(price/K) + (0.04 + 0.5*self.vol**2)*30/365) / (self.vol*np.sqrt(30/365))
            d2 = d1 - self.vol*np.sqrt(30/365)
            call_price = price*norm.cdf(d1) - K*np.exp(-0.04*30/365)*norm.cdf(d2)
            
            chain.append({
                'strike': round(K, 2),
                'expiry': '30D',
                'type': 'call',
                'bid': round(call_price * 0.99, 2),
                'ask': round(call_price * 1.01, 2),
                'iv': self.vol,
                'delta': round(norm.cdf(d1), 3)
            })
        return pd.DataFrame(chain)
