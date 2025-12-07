# create_trading_features.py
# Run this from your project root: python create_trading_features.py
import os
from textwrap import dedent

ROOT = os.path.abspath(os.path.dirname(__file__))

files = {
"scanners/stock_scanner.py": dedent("""
    # scanners/stock_scanner.py
    import yfinance as yf
    import pandas as pd
    import numpy as np

    NIFTY50_SAMPLE = [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","HDFC.NS","ICICIBANK.NS",
        "KOTAKBANK.NS","SBIN.NS","LT.NS","ITC.NS","AXISBANK.NS","HINDUNILVR.NS",
        "BHARTIARTL.NS","MARUTI.NS","BAJFINANCE.NS","ONGC.NS","BAJAJ-AUTO.NS","POWERGRID.NS"
    ]

    def download_candles(symbol, period="7d", interval="5m"):
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            raise RuntimeError(f"No data for {symbol}")
        df = df[['Open','High','Low','Close','Volume']].rename(
            columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}
        )
        df.index = pd.to_datetime(df.index)
        return df

    def compute_features(df):
        f = pd.DataFrame(index=df.index)
        f['close'] = df['close']
        f['ret1'] = df['close'].pct_change(1)
        f['ret5'] = df['close'].pct_change(5)
        f['ma5'] = df['close'].rolling(5).mean()
        f['ma20'] = df['close'].rolling(20).mean()
        f['atr'] = (df['high'] - df['low']).rolling(14).mean()
        f['vol'] = df['volume'].rolling(20).mean().fillna(0)
        f = f.fillna(0)
        return f

    def score_momentum(df_features):
        recent = df_features.iloc[-1]
        score = 0.0
        score += np.sign(recent['ret1']) * min(abs(recent['ret1']) * 100, 3)
        score += np.sign(recent['ret5']) * min(abs(recent['ret5']) * 50, 5)
        score += 3.0 if recent['ma5'] > recent['ma20'] else -2.0
        if recent['vol'] > df_features['vol'].median():
            score += 1.0
        score -= min(recent['atr'] / max(recent['close'], 1) * 100, 3)
        return score

    def scan_universe(universe=None, period="7d", interval="5m", top_n=10):
        universe = universe or NIFTY50_SAMPLE
        results = []
        for s in universe:
            try:
                df = download_candles(s, period=period, interval=interval)
                feats = compute_features(df)
                score = score_momentum(feats)
                last = feats['close'].iloc[-1]
                atr = feats['atr'].iloc[-1]
                results.append({"symbol": s, "score": score, "price": last, "atr": atr})
            except Exception:
                continue
        dfres = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
        return dfres.head(top_n)
"""),

"options/option_utils.py": dedent("""
    # options/option_utils.py
    import requests
    import pandas as pd
    import numpy as np
    import math
    from datetime import date, timedelta, datetime
    from scipy.stats import norm

    # ---------- Black-Scholes ----------
    def bs_price(S, K, r, sigma, tau, is_call=True):
        if tau <= 0 or sigma <= 0:
            return max(0.0, (S - K) if is_call else (K - S))
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
        d2 = d1 - sigma * math.sqrt(tau)
        if is_call:
            return S * norm.cdf(d1) - K * math.exp(-r * tau) * norm.cdf(d2)
        else:
            return K * math.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def bs_greeks(S, K, r, sigma, tau, is_call=True):
        if tau <= 0 or sigma <= 0:
            return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
        d2 = d1 - sigma * math.sqrt(tau)
        delta = norm.cdf(d1) if is_call else (norm.cdf(d1) - 1)
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(tau))
        vega = S * norm.pdf(d1) * math.sqrt(tau)
        theta = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(tau)) - r * K * math.exp(-r * tau) * (norm.cdf(d2) if is_call else -norm.cdf(-d2))
        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}

    # ---------- NSE option chain fetch (best-effort, free) ----------
    def fetch_option_chain_nse(symbol):
        """
        Try to fetch NSE option-chain for index symbol like NIFTY/BANKNIFTY via public NSE endpoint.
        If it fails, fallback to synthetic chain.
        Returns: DataFrame and underlying price
        """
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol.replace('NIFTY','NIFTY')}"
        headers = {
            "User-Agent":"Mozilla/5.0",
            "Accept-Language":"en-US,en;q=0.9",
            "Accept":"application/json, text/plain, */*",
            "Referer":"https://www.nseindia.com/",
        }
        session = requests.Session()
        try:
            # visit home to get cookies
            session.get("https://www.nseindia.com", headers=headers, timeout=5)
            r = session.get(url, headers=headers, timeout=10)
            data = r.json()
            records = data.get('records', {})
            underlying = records.get('underlyingValue', None)
            data_list = []
            expiry_dates = records.get('expiryDates', []) or []
            for rec in records.get('data', []):
                strike = rec.get('strikePrice')
                ce = rec.get('CE')
                pe = rec.get('PE')
                if ce:
                    ce_row = {
                        "strike": strike, "expiry": ce.get('expiryDate'),
                        "type": "CE", "lastPrice": ce.get('lastPrice') or ce.get('underlyingValue') or 0,
                        "bid": ce.get('bidprice') or 0, "ask": ce.get('askPrice') or 0,
                        "openInterest": ce.get('openInterest') or 0, "iv": ce.get('impliedVolatility') or 0
                    }
                    data_list.append(ce_row)
                if pe:
                    pe_row = {
                        "strike": strike, "expiry": pe.get('expiryDate'),
                        "type": "PE", "lastPrice": pe.get('lastPrice') or 0,
                        "bid": pe.get('bidprice') or 0, "ask": pe.get('askPrice') or 0,
                        "openInterest": pe.get('openInterest') or 0, "iv": pe.get('impliedVolatility') or 0
                    }
                    data_list.append(pe_row)
            df = pd.DataFrame(data_list)
            if df.empty:
                raise ValueError("Empty DF from NSE")
            return df, underlying
        except Exception:
            # fallback: synthetic chain around a proxy price (use numpy)
            return synthetic_chain_fallback()

    def synthetic_chain_fallback():
        S = 20000.0
        expiries = [date.today() + timedelta(days=7)]
        strikes = np.arange(int(S*0.8/100)*100, int(S*1.2/100)*100 + 100, 100)
        rows = []
        for ex in expiries:
            tau = (ex - date.today()).days / 365.0
            for K in strikes:
                m = abs(K - S)/S
                iv = 0.18 + m*0.5
                for t in ['CE','PE']:
                    lp = bs_price(S, K, 0.06, iv, tau, is_call=(t=='CE'))
                    rows.append({
                        "strike": K, "expiry": ex.isoformat(), "type": t,
                        "lastPrice": lp, "bid": lp*0.99, "ask": lp*1.01,
                        "openInterest": int(1000*(1-m)), "iv": iv
                    })
        import pandas as pd
        return pd.DataFrame(rows), S

    def suggest_strike_for_direction(underlying_price, option_chain_df, direction="bullish"):
        import pandas as pd
        df = option_chain_df.copy()
        df['expiry_dt'] = pd.to_datetime(df['expiry'])
        df['days'] = (df['expiry_dt'].dt.date - pd.Timestamp.now().date()).dt.days
        df['tau'] = (df['days'] / 365.0).clip(lower=1e-6)
        # compute greeks for each row (fast, vectorized)
        deltas = []
        for i, r in df.iterrows():
            g = bs_greeks(underlying_price, r['strike'], 0.06, r.get('iv', 0.25), r['tau'], is_call=(r['type']=='CE'))
            deltas.append(g['delta'])
        df['delta'] = deltas
        want_type = 'CE' if direction=='bullish' else 'PE'
        cand = df[df['type']==want_type]
        cand = cand[(cand['openInterest']>50) & (cand['iv'] < 2.0)]
        if cand.empty:
            return None
        target_low, target_high = (0.25, 0.6)
        cand['score_delta'] = (cand['delta'].abs() - (target_low+target_high)/2).abs()
        cand = cand.sort_values(['score_delta','openInterest'], ascending=[True,False])
        best = cand.iloc[0].to_dict()
        return best

    def format_option_suggestion(opt):
        if opt is None:
            return None
        return {
            "strike": int(opt['strike']),
            "type": opt['type'],
            "expiry": str(opt['expiry']),
            "lastPrice": float(opt.get('lastPrice',0)),
            "delta": float(opt.get('delta',0)),
            "iv": float(opt.get('iv',0)),
            "oi": int(opt.get('openInterest',0))
        }
"""),

"signals/entry_rules.py": dedent("""
    # signals/entry_rules.py
    import pandas as pd

    def atr(high, low, close, n=14):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(n).mean()

    def entry_stop_target_from_atr(latest_price, latest_atr, risk_fraction=0.01, capital=100000):
        stop_loss_amount = 1.5 * latest_atr
        target_amount = 2.0 * stop_loss_amount
        if stop_loss_amount <= 0:
            return None
        position_value = (risk_fraction * capital) / stop_loss_amount
        qty = max(int(position_value / latest_price), 0)
        entry = latest_price
        stop = entry - stop_loss_amount
        target = entry + target_amount
        return {
            "entry": round(entry,2),
            "stop": round(stop,2),
            "target": round(target,2),
            "qty": int(qty),
            "risk_amount": round(risk_fraction * capital, 2),
            "stop_loss_points": round(stop_loss_amount, 4)
        }

    def option_position_sizing(option_premium, risk_fraction=0.01, capital=100000, lot_size=1):
        risk_money = risk_fraction * capital
        if option_premium <= 0:
            return 0
        contracts = int(risk_money // (option_premium * lot_size))
        return max(contracts, 0)
"""),

"serving/dashboard.py": dedent("""
    # serving/dashboard.py
    import os, sys
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    import streamlit as st
    import pandas as pd
    import numpy as np
    import time

    # Import modules we created
    from scanners.stock_scanner import scan_universe, download_candles, compute_features
    from options.option_utils import fetch_option_chain_nse, suggest_strike_for_direction, format_option_suggestion
    from signals.entry_rules import entry_stop_target_from_atr, option_position_sizing

    st.set_page_config(page_title="Quant-Core Trading Dashboard", layout="wide")
    st.title("🚀 Quant-Core Indian Market Toolkit (MVP)")

    # Sidebar controls
    st.sidebar.header("Controls")
    symbol = st.sidebar.text_input("Symbol (ex: NIFTY, BANKNIFTY or RELIANCE.NS)", value="NIFTY")
    universe_choice = st.sidebar.multiselect("Universe", options=["NIFTY50"], default=["NIFTY50"])
    interval = st.sidebar.selectbox("Interval", ["1m","5m","15m","30m","1h"], index=2)
    period = st.sidebar.selectbox("Period", ["1d","5d","7d","30d"], index=1)

    # Scanner
    st.sidebar.markdown("### Scanner (intraday / scalp)")
    if st.sidebar.button("Run Scanner"):
        with st.spinner("Scanning..."):
            from scanners.stock_scanner import NIFTY50_SAMPLE
            universe = NIFTY50_SAMPLE if "NIFTY50" in universe_choice else None
            res = scan_universe(universe=universe, period="5d", interval=interval, top_n=15)
            st.session_state['scan_results'] = res
            st.success("Scanner finished")

    # Load chart data for symbol
    @st.cache_data(ttl=30)
    def load_df(sym, per, intr):
        try:
            return download_candles(sym, period=per, interval=intr)
        except Exception as e:
            st.error("Data load failed: " + str(e))
            return pd.DataFrame()

    df = load_df(symbol, period, interval)

    # Layout: left = chart, right = scanner & option suggestions
    left, right = st.columns([2,1])

    with left:
        st.subheader(f"{symbol} Price (last {len(df)} bars)")
        if not df.empty:
            st.line_chart(df['close'])
        else:
            st.info("No data for symbol; try e.g. RELIANCE.NS or NIFTY")

        st.markdown("### Option chain (fetch & suggestion)")
        if st.button("Fetch Option Chain & Suggest Strike"):
            with st.spinner("Fetching chain..."):
                oc_df, underlying = fetch_option_chain_nse(symbol if symbol.upper() in ['NIFTY','BANKNIFTY'] else symbol)
                st.session_state['oc'] = oc_df
                st.session_state['underlying'] = underlying
                st.success("Fetched option chain (or synthetic fallback)")

        if 'oc' in st.session_state:
            oc = st.session_state['oc']
            st.write("Underlying price (approx):", st.session_state.get('underlying'))
            st.dataframe(oc.head(40))

            # Let user choose direction
            dir_choice = st.selectbox("Direction for option suggestion", ["bullish","bearish"])
            best = suggest_strike_for_direction(st.session_state.get('underlying', 0), oc, direction=dir_choice)
            st.write("Suggested strike:", format_option_suggestion(best))
            if best:
                prem = float(best.get('lastPrice',0))
                contracts = option_position_sizing(prem, lot_size=15)
                st.write("Suggested contracts (lot 15 assumed):", contracts)

    with right:
        st.subheader("Top Intraday Picks")
        if 'scan_results' in st.session_state:
            st.dataframe(st.session_state['scan_results'])
            for i, row in st.session_state['scan_results'].iterrows():
                s = row['symbol']
                st.markdown(f\"\"\"**{s}** — Score: {row['score']:.2f} — Price: {row['price']:.2f} — ATR: {row['atr']:.3f}\"\"\")
                if st.button(f"Suggest option for {s}", key=f"opt_{s}"):
                    with st.spinner("Fetching option chain..."):
                        oc_df, underlying_price = fetch_option_chain_nse(s)
                        direction = "bullish" if row['score'] > 0 else "bearish"
                        opt = suggest_strike_for_direction(underlying_price, oc_df, direction=direction)
                        st.write("Option suggestion:", format_option_suggestion(opt))
                        try:
                            dfx = download_candles(s, period="7d", interval=interval)
                            feats = compute_features(dfx)
                            latest_atr = float(feats['atr'].iloc[-1])
                            est = entry_stop_target_from_atr(float(row['price']), latest_atr)
                            st.write("Entry / Stop / Target:", est)
                            if opt:
                                prem = float(opt['lastPrice'])
                                st.write("Suggested contracts (lot 15):", option_position_sizing(prem))
                        except Exception as e:
                            st.error("Failed to compute entry/size: " + str(e))
        else:
            st.info("Run the scanner from the sidebar to list top intraday picks")

    st.caption("Quant-Core — Demo MVP. Replace free endpoints with broker API for production.")
""")
}

# create directories & files
for path, content in files.items():
    full = os.path.join(ROOT, path)
    d = os.path.dirname(full)
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
        print("Created folder:", d)
    with open(full, "w", encoding="utf8") as f:
        f.write(content)
    print("Wrote file:", full)

print("\\nALL FILES CREATED.")
print("Next steps:")
print("1) Activate your venv and install missing packages if any:")
print("   pip install yfinance scipy pandas numpy requests streamlit")
print("   (you probably already installed most of them)")
print("2) Run Streamlit:")
print("   streamlit run serving/dashboard.py")
print("\\nIf you want, I can also:")
print(" - plug in AngelOne or Zerodha API code to replace free NSE calls")
print(" - add NewsAPI sentiment integration (requires API key)")
