# ============================================
# FIX IMPORT PATHS FOR LOCAL PACKAGE STRUCTURE
# ============================================
import os
import sys

# Add project root so "data", "models", "features" import correctly
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

# =====================================================
# AUTO-GENERATE INITIAL PRICE HISTORY (60 ticks minimum)
# =====================================================
if "history" not in st.session_state or len(st.session_state.history) < 60:
    st.session_state.history = [market.get_tick() for _ in range(60)]

# Sidebar buttons
if st.sidebar.button("Generate Tick"):
    st.session_state.history.append(market.get_tick())

predict = st.sidebar.button("Run Prediction")

# Main UI Tabs
tab1, tab2 = st.tabs(["📊 Market Chart", "🤖 Model Output"])

# -------------------------
# MARKET CHART TAB
# -------------------------
with tab1:
    st.subheader("Market Price History")
    df = pd.DataFrame(st.session_state.history, columns=["price"])
    st.line_chart(df["price"])

# -------------------------
# MODEL OUTPUT TAB
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

            # Train model
            model = TabularModel()
            model.train(df_hist[["close"]], df_hist["target"])

            latest = df_hist["close"].iloc[-1]
            p = model.predict([[latest]])[0]

            # Regime detection
            regime = regime_detector.predict(df_hist["close"].pct_change().dropna())

            # Display results
            st.metric("Latest Price", f"${latest:,.2f}")
            st.metric("Prob Up", f"{p:.2%}")
            st.metric("Regime", str(regime))

            if p > 0.6:
                st.success("BUY CALL")
            else:
                st.info("HOLD")

st.caption("Quant-Core Streamlit Demo")

