# serving/dashboard.py
import os, sys, time, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.synthetic import SyntheticMarket
from models.zoo import TabularModel
from features.regimes import RegimeDetector

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Quant-Core Dashboard", layout="wide")
st.title("🚀 Quant-Core AI Trading Dashboard (MVP)")

market = SyntheticMarket()
regime_detector = RegimeDetector()

# Initialize history with 60 ticks if not present
if "history" not in st.session_state:
    st.session_state.history = [market.get_tick() for _ in range(60)]
if "model" not in st.session_state:
    # lightweight model instance cached in session
    try:
        st.session_state.model = TabularModel()
    except Exception:
        st.session_state.model = None

# Sidebar controls
col1, col2 = st.sidebar.columns(2)
if col1.button("Generate 1 Tick"):
    st.session_state.history.append(market.get_tick())
if col2.button("Generate 100 Ticks"):
    st.session_state.history.extend([market.get_tick() for _ in range(100)])
if st.sidebar.button("Reset History"):
    st.session_state.history = [market.get_tick() for _ in range(60)]

predict = st.sidebar.button("Run Prediction")

# Live UI
tab1, tab2 = st.tabs(["📊 Market Chart", "🤖 Model Output"])

with tab1:
    st.subheader("Market Price History (latest first shown)")
    df = pd.DataFrame(st.session_state.history, columns=["price"])
    st.line_chart(df["price"])

with tab2:
    st.subheader("AI Prediction / Signal")
    st.write(f"History length: {len(st.session_state.history)}")
    if predict:
        history = st.session_state.history
        if len(history) < 60:
            st.warning("Need at least 60 ticks")
        else:
            df_hist = pd.DataFrame(history, columns=["close"])
            df_hist["target"] = (df_hist["close"].shift(-5) > df_hist["close"]).astype(int)
            df_hist = df_hist.dropna()
            # train simple model (TabularModel from your repo)
            model = st.session_state.model or TabularModel()
            try:
                model.train(df_hist[["close"]], df_hist["target"])
                latest = df_hist["close"].iloc[-1]
                p = float(model.predict([[latest]])[0])
            except Exception as e:
                st.error("Model train/predict failed: " + str(e))
                p = 0.5

            regime = None
            try:
                regime = regime_detector.predict(df_hist["close"].pct_change().dropna())
            except Exception:
                regime = "N/A"

            st.metric("Latest Price", f"${latest:,.2f}")
            st.metric("Prob Up", f"{p:.2%}")
            st.metric("Regime", str(regime))
            action = "HOLD"
            if p > 0.6:
                action = "BUY"
            elif p < 0.4:
                action = "SELL"
            st.markdown(f"### Recommendation: **{action}**")

            # show last few rows
            st.dataframe(df_hist.tail(10))

st.caption("Quant-Core Streamlit MVP — Synthetic data")
