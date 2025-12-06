# ============================================
# FIX IMPORT PATHS FOR LOCAL PACKAGE STRUCTURE
# ============================================
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
print(">>> PROJECT ROOT ADDED:", ROOT)

# Normal imports now work
from data.synthetic import SyntheticMarket
from models.zoo import TabularModel
from features.regimes import RegimeDetector

# ============================================
# STREAMLIT STARTS HERE
# ============================================

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Quant-Core Dashboard", layout="wide")
st.title("🚀 Quant-Core AI Trading Dashboard")

market = SyntheticMarket()
regime_detector = RegimeDetector()

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
if st.sidebar.button("Generate Tick"):
    st.session_state.history.append(market.get_tick())

predict = st.sidebar.button("Run Prediction")

# Tabs
tab1, tab2 = st.tabs(["📊 Market Chart", "🤖 Model Output"])

# -------------------------
# MARKET CHART
# -------------------------
with tab1:
    st.subheader("Market Price History")
    if len(st.session_state.history) == 0:
        st.info("Click Generate Tick to start")
    else:
        df = pd.DataFrame(st.session_state.history, columns=["price"])
        st.line_chart(df["price"])

# -------------------------
# AI MODEL OUTPUT
# -------------------------
with tab2:
    st.subheader("AI Prediction")
    history = st.session_state.history

    if predict:
        if len(history) < 60:
            st.warning("Need at least 60 ticks")
        else:
            df_hist = pd.DataFrame(history, columns=["close"])
            df_hist["target"] = (df_hist["close"].shift(-5) > df_hist["close"]).astype(int)
            df_hist = df_hist.dropna()

            model = TabularModel()
            model.train(df_hist[["close"]], df_hist["target"])

            latest = df_hist["close"].iloc[-1]
            p = model.predict([[latest]])[0]

            regime = regime_detector.predict(df_hist["close"].pct_change().dropna())

            st.metric("Latest Price", f"${latest:,.2f}")
            st.metric("Prob Up", f"{p:.2%}")
            st.metric("Regime", str(regime))

            if p > 0.6:
                st.success("BUY CALL")
            else:
                st.info("HOLD")

st.caption("Quant-Core Streamlit Demo")
