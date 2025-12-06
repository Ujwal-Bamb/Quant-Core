# ============================================================
# FIX PATHS SO STREAMLIT CAN IMPORT data/, models/, features/
# ============================================================
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

print("ROOT DIR:", ROOT_DIR)

# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np

from data.synthetic import SyntheticMarket
from models.zoo import TabularModel
from features.regimes import RegimeDetector


# ============================================================
# STREAMLIT PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Quant-Core AI Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.markdown("<h1 style='text-align:center;'>🚀 Quant-Core AI Trading Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#ccc;'>Synthetic Market • AI Predictions • Trading Signals</p>", unsafe_allow_html=True)


# ============================================================
# INITIALIZE COMPONENTS
# ============================================================
market = SyntheticMarket()
rd = RegimeDetector()

# Streamlit session state
if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("⚙️ Controls")

simulate = st.sidebar.button("📈 Generate Price Tick")
predict = st.sidebar.button("🤖 Run Prediction")


# ============================================================
# HANDLE PRICE SIMULATION
# ============================================================
if simulate:
    price = market.get_tick()
    st.session_state.history.append(price)

history = st.session_state.history


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📊 Market Chart", "🤖 AI Model Output", "ℹ️ System Info"])


# ============================================================
# TAB 1 — MARKET CHART
# ============================================================
with tab1:
    st.header("📊 Market Price History")

    if len(history) == 0:
        st.info("Click **Generate Price Tick** to begin.")
    else:
        df = pd.DataFrame(history, columns=["Price"])
        st.line_chart(df["Price"])


# ============================================================
# TAB 2 — AI MODEL PREDICTIONS
# ============================================================
with tab2:
    st.header("🤖 AI Trading Signals")

    if predict and len(history) > 60:

        # Build training DataFrame
        df_hist = pd.DataFrame(history, columns=['close'])
        df_hist["target"] = (df_hist['close'].shift(-5) > df_hist['close']).astype(int)
        df_hist = df_hist.dropna()

        # Train model (simple LightGBM model)
        model = TabularModel()
        model.train(df_hist[['close']], df_hist['target'])

        latest_price = history[-1]
        probability_up = model.predict([[latest_price]])[0]

        # Regime detection
        regime = rd.predict(df_hist["close"].pct_change().dropna())

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Latest Price", f"${latest_price:,.2f}")

        with col2:
            st.metric("Probability Up", f"{probability_up:.2%}")

        with col3:
            st.metric("Market Regime", str(regime))

        st.write("---")

        # Trading signal
        if probability_up > 0.60:
            st.success("📈 **BUY CALL** — Model expects bullish movement.")
        else:
            st.info("⏸ **HOLD** — No strong prediction.")


    else:
        st.info("Generate **at least 60 ticks** before running predictions.")


# ============================================================
# TAB 3 — SYSTEM INFO
# ============================================================
with tab3:
    st.header("ℹ️ About Quant-Core")
    st.write("""
        Quant-Core is a modular AI trading system featuring:

        - 🧠 LightGBM, LSTM, Transformer models  
        - 📉 Synthetic market generator  
        - 🔍 Regime detection (GMM / volatility clustering)  
        - 💹 Backtesting engine  
        - 🌐 FastAPI inference server  
        - 📊 Streamlit dashboard  

        This dashboard demonstrates the synthetic simulation and prediction pipeline.
    """)

st.write("---")
st.caption("Quant-Core — Streamlit Edition (Demo Mode)")
