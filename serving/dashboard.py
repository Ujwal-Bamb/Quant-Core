import os
import sys
from datetime import datetime

# -------------------------
# Make project importable
# -------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Our helper modules (from create_trading_features.py)
from scanners.stock_scanner import (
    scan_universe,
    download_candles,
    compute_features,
    NIFTY50_SAMPLE,
)
from options.option_utils import synthetic_chain, suggest_strike
from signals.entry_rules import entry_stop_target

# ML imports (LightGBM if available, else RandomForest)
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

from sklearn.ensemble import RandomForestClassifier

# ==========================================
# BASIC PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Quant-Core Indian Trading AI", layout="wide")
st.title("🚀 Quant-Core Indian Trading AI (MVP)")

st.markdown(
    "Paper-trading only. This is **not** investment advice. "
    "Use for education, testing and strategy research."
)

# ==========================================
# HELPERS
# ==========================================

@st.cache_data(ttl=60)
def load_ohlcv(symbol: str, period: str = "30d", interval: str = "5m") -> pd.DataFrame:
    """Load OHLCV candles via yfinance."""
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        return df
    df = df.rename(columns=str.lower)
    return df[["open", "high", "low", "close", "volume"]]


@st.cache_resource(show_spinner=False)
def train_price_model(symbol: str, period: str = "30d", interval: str = "5m", horizon: int = 5):
    """Train a simple classification model: will price be higher after N bars?"""
    df = load_ohlcv(symbol, period=period, interval=interval)
    if df.empty:
        raise RuntimeError("No data to train for symbol: " + symbol)

    feats = compute_features(df)
    # future return over horizon bars
    future_close = df["close"].shift(-horizon)
    ret = (future_close / df["close"] - 1.0)
    y = (ret > 0).astype(int)

    # align
    df_train = pd.concat([feats, y.rename("y")], axis=1).dropna()
    if df_train.empty:
        raise RuntimeError("Not enough data after feature engineering.")

    X = df_train.drop(columns=["y"])
    y = df_train["y"]

    if HAS_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=120,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )

    model.fit(X, y)
    meta = {
        "symbol": symbol,
        "period": period,
        "interval": interval,
        "horizon": horizon,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "feature_cols": list(X.columns),
    }
    return model, meta


def predict_latest_prob(model, meta, df_raw: pd.DataFrame):
    feats = compute_features(df_raw)
    X = feats[meta["feature_cols"]].iloc[[-1]]
    proba = model.predict_proba(X)[0, 1]
    return float(proba)


def backtest_threshold(
    model,
    meta,
    df_raw: pd.DataFrame,
    prob_threshold: float = 0.55,
    fee_bp: float = 1.0,
):
    """Very simple long/flat backtest on model probabilities."""
    feats = compute_features(df_raw)
    X = feats[meta["feature_cols"]].copy()
    X = X.dropna()
    common_idx = X.index.intersection(df_raw.index)
    X = X.loc[common_idx]
    close = df_raw["close"].loc[common_idx]

    probs = model.predict_proba(X)[:, 1]
    signal = (probs > prob_threshold).astype(int)  # 1 = long, 0 = flat
    ret = close.pct_change().fillna(0.0)

    strat_ret = signal * ret
    # transaction cost when signal changes (entry or exit)
    trades = (signal.diff().abs().fillna(0.0) > 0).astype(int)
    fee = trades * (fee_bp / 10000.0)
    strat_ret = strat_ret - fee

    equity = (1.0 + strat_ret).cumprod()
    total_return = equity.iloc[-1] - 1.0
    sharpe = np.nan
    if strat_ret.std() > 0:
        # approx: 252 trading days * ~78 five-minute bars per day
        sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252 * 78)
    summary = {
        "total_return": float(total_return),
        "sharpe": float(sharpe) if np.isfinite(sharpe) else None,
        "trades": int(trades.sum()),
        "equity": equity,
    }
    return summary


# ==========================================
# SIDEBAR – GLOBAL CONTROLS
# ==========================================
st.sidebar.header("Global Settings")

default_symbol = st.sidebar.text_input("Symbol (e.g. RELIANCE.NS or NIFTY)", value="RELIANCE.NS")
intraday_interval = st.sidebar.selectbox("Intraday interval", ["5m", "15m", "30m"], index=0)
intraday_period = st.sidebar.selectbox("History period", ["5d", "30d", "60d"], index=1)

# ==========================================
# TABS
# ==========================================
tab_overview, tab_options, tab_ai, tab_backtest = st.tabs(
    [
        "📊 Market Overview",
        "📈 Options & Strikes",
        "🤖 AI Prediction",
        "📉 Backtest & Performance",
    ]
)

# ==========================================
# TAB 1 – MARKET OVERVIEW (SCANNER + CHART)
# ==========================================
with tab_overview:
    st.subheader("Market Overview & Intraday Scanner")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"**Spot chart – {default_symbol} ({intraday_interval})**")
        df_spot = load_ohlcv(default_symbol, period=intraday_period, interval=intraday_interval)
        if df_spot.empty:
            st.warning("No data for this symbol. Try RELIANCE.NS, TCS.NS, SBIN.NS etc.")
        else:
            st.line_chart(df_spot["close"])
            st.caption(f"Bars: {len(df_spot)}")

    with col2:
        st.markdown("**Intraday Scanner (NIFTY sample)**")
        if st.button("Run Scanner"):
            with st.spinner("Scanning universe for momentum & trend..."):
                scan_df = scan_universe(NIFTY50_SAMPLE)
                st.session_state["scan_df"] = scan_df
        if "scan_df" in st.session_state:
            st.dataframe(st.session_state["scan_df"].reset_index(drop=True))
            if not st.session_state["scan_df"].empty:
                best_row = st.session_state["scan_df"].iloc[0]
                st.markdown(
                    f"**Top pick now:** `{best_row['symbol']}` – "
                    f"Score: `{best_row['score']}` – Price: `{best_row['price']:.2f}`"
                )
        else:
            st.info("Click **Run Scanner** to see intraday candidates.")


# ==========================================
# TAB 2 – OPTIONS & STRIKES (LIKE OPTION CHAIN PICKER)
# ==========================================
with tab_options:
    st.subheader("Options & Strike Suggestion")

    st.markdown(
        "For real NSE option chain, later we can plug broker / NSE APIs. "
        "For now this uses a **synthetic chain** around an underlying price you choose."
    )

    col1, col2 = st.columns(2)

    with col1:
        underlying_price = st.number_input(
            "Underlying price (index or stock)", min_value=1.0, value=20000.0, step=50.0
        )
        if st.button("Generate Synthetic Option Chain"):
            chain, px = synthetic_chain(underlying_price)
            st.session_state["opt_chain"] = chain
            st.session_state["opt_underlying"] = px

        if "opt_chain" in st.session_state:
            st.dataframe(st.session_state["opt_chain"].head(40))
        else:
            st.info("Click **Generate Synthetic Option Chain** to view CE/PE strikes.")

    with col2:
        st.markdown("### AI-style Strike Suggestion")

        direction = st.selectbox("View for direction", ["bullish (CE)", "bearish (PE)"])
        dir_key = "bullish" if direction.startswith("bullish") else "bearish"

        if "opt_chain" in st.session_state:
            best = suggest_strike(st.session_state["opt_chain"], direction=dir_key)
            st.write("**Suggested Option:**", best)

            if best:
                price = float(best.get("lastPrice", 0))
                levels = entry_stop_target(underlying_price)
                st.write("**Spot-based Entry / Stop / Target**")
                st.json(levels)
                st.caption(
                    "You can adapt these levels to option price (e.g., same % distance from spot)."
                )
        else:
            st.info("First generate the chain on the left side.")


# ==========================================
# TAB 3 – AI PREDICTION (LIVE MODEL)
# ==========================================
with tab_ai:
    st.subheader("AI Prediction – Will Price Go Up?")

    st.markdown(
        "This trains a simple ML model on intraday candles of a symbol and predicts the "
        "probability that price will be **higher after N bars**."
    )

    horizon = st.slider("Prediction horizon (bars ahead)", min_value=3, max_value=30, value=10)

    if st.button("Train / Retrain Model"):
        with st.spinner("Downloading data & training model..."):
            try:
                model, meta = train_price_model(
                    default_symbol,
                    period=intraday_period,
                    interval=intraday_interval,
                    horizon=horizon,
                )
                st.session_state["ai_model"] = model
                st.session_state["ai_meta"] = meta
                st.success(
                    f"Model trained on {default_symbol} – horizon {horizon} bars. "
                    f"Trained at {meta['trained_at']}."
                )
            except Exception as e:
                st.error("Training failed: " + str(e))

    if "ai_model" in st.session_state and "ai_meta" in st.session_state:
        model = st.session_state["ai_model"]
        meta = st.session_state["ai_meta"]
        df_latest = load_ohlcv(default_symbol, period=intraday_period, interval=intraday_interval)
        if df_latest.empty:
            st.warning("No data to predict.")
        else:
            prob = predict_latest_prob(model, meta, df_latest)
            last_price = df_latest["close"].iloc[-1]
            st.metric(
                "Latest price",
                f"{last_price:.2f}",
            )
            st.metric(
                f"Prob(price up in {meta['horizon']} bars)",
                f"{prob:.1%}",
            )

            st.markdown("### AI suggestion")
            if prob > 0.6:
                st.success("Bias: **BUY / CALL side** (model sees upside edge).")
            elif prob < 0.4:
                st.error("Bias: **SELL / PUT side** (model sees downside edge).")
            else:
                st.info("Bias: **NO CLEAR EDGE** (avoid overtrading).")
    else:
        st.info("Train the model first using the button above.")


# ==========================================
# TAB 4 – BACKTEST (USING THE AI SIGNALS)
# ==========================================
with tab_backtest:
    st.subheader("Backtest – AI Strategy on Intraday Data")

    st.markdown(
        "Uses the trained AI model to simulate a simple long/flat strategy. "
        "Entry when probability > threshold, exit to cash when below."
    )

    threshold = st.slider("Entry threshold (prob up)", min_value=0.50, max_value=0.80, step=0.01, value=0.55)

    if (
        "ai_model" in st.session_state
        and "ai_meta" in st.session_state
    ):
        model = st.session_state["ai_model"]
        meta = st.session_state["ai_meta"]
        df_bt = load_ohlcv(default_symbol, period=intraday_period, interval=intraday_interval)
        if df_bt.empty:
            st.warning("No data to backtest.")
        else:
            with st.spinner("Running backtest..."):
                res = backtest_threshold(model, meta, df_bt, prob_threshold=threshold)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Return", f"{res['total_return']*100:.2f}%")
            col2.metric("Sharpe (rough)", "N/A" if res["sharpe"] is None else f"{res['sharpe']:.2f}")
            col3.metric("Trades", res["trades"])

            st.markdown("### Equity Curve")
            eq_df = pd.DataFrame({"equity": res["equity"]})
            st.line_chart(eq_df["equity"])
    else:
        st.info("Train the model in the **AI Prediction** tab first.")

# ==========================================
# FOOTER
# ==========================================
st.caption(
    "Quant-Core demo – For education only. "
    "For real trading like Angel One, you must connect a broker API, handle risk, slippage, and regulations."
)
