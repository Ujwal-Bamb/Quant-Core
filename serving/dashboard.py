# ============================================================
# ABSOLUTELY MUST BE FIRST — FIX PYTHON PATH FOR STREAMLIT
# ============================================================

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

print(">>> ROOT DIR LOADED FOR IMPORTS:", ROOT_DIR)


# ============================================================
# NOW SAFE TO IMPORT OTHER LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

# Your project modules (will now work)
from data.synthetic import SyntheticMarket
from models.zoo import TabularModel
from features.regimes import RegimeDetector


# ============================================================
# STREAMLIT UI SETUP
# ============================================================

st.set_page_config(
    page_title="Quant-Core AI Trading Dashboard",
    layout="wide",
)

st.title("🚀 Quant-Core AI Trading Dashboard")


# ============================================================
# INITIALIZE MARKET + MODEL COMPONENTS
# ============================================================

market = SyntheticMarket()
regime_detector = RegimeDetector()

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("⚙️ Controls")

if st.sidebar.button("📈 Generate Price Tick"):
    price = market.get_tick()
    st.session_state.history.append(price)

run_prediction = st.sidebar.button("🤖 Run AI Prediction")


# ============================================================
# TABS LAYOUT
# ============================================================

tab1, tab2, tab3 = st.tabs(["📊 Market Chart", "🤖 AI Model Output", "ℹ️ About"])


# ============================================================
# TAB 1 — PRICE CHART
# ============================================================

with tab1:
    st.subheader("📊 Market Price History")

    if len(st.session_state.history) == 0:
        st.info("Click **Generate Price Tick** to start simulation.")
    else:
        df = pd.DataFrame(st.session_state.history, columns=["Price"])
        st.line_chart(df["Price"])


# ============================================================
# TAB 2 — PREDICTION ENGINE
# ============================================================

with tab2:
    st.subheader("🤖 AI Prediction Model")

    history = st.session_state.history

    if run_prediction:

        if len(history) < 60:
            st.warning("Need **at least 60 ticks** before running AI model.")
        else:
            df_hist = pd.DataFrame(history, columns=["close"])
            df_hist["target"] = (df_hist["close"].shift(-5) > df_hist["close"]).astype(int)
            df_hist = df_hist.dropna()

            model = TabularModel()
            model.train(df_hist[["close"]], df_hist["target"])

            latest = df_hist["close"].iloc[-1]
            prob = model.predict([[latest]])[0]

            regime = regime_detector.predict(df_hist["close"].pct_change().dropna())

            col1, col2, col3 = st.columns(3)

            col1.metric("Latest Price", f"${latest:,.2f}")
            col2.metric("Probability Up", f"{prob:.2%}")
            col3.metric("Market Regime", str(regime))

            st.write("---")

            if prob > 0.60:
                st.success("📈 **BUY CALL** — Model is bullish!")
            else:
                st.info("⏸ **HOLD** — No strong bullish signal.")

    else:
        st.info("Click **Run AI Prediction** to generate signals.")


# ============================================================
# TAB 3 — ABOUT SYSTEM
# ============================================================

with tab3:
    st.subheader("ℹ️ About Quant-Core")

    st.write("""
    Quant-Core is a modular AI trading research system built with:

    - 📈 Synthetic market generator  
    - 🧠 LightGBM AI prediction model  
    - 🔍 Market regime classifier  
    - ⏱ Backtesting engine  
    - 🌐 FastAPI + Streamlit interface  

    This dashboard is running in **demo mode** using synthetic data.
    """)

st.write("---")
st.caption("Quant-Core — AI Trading Framework (Demo Mode)")
