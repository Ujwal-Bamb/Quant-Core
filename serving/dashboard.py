import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

# ============================================================
# FIX PYTHON PATH SO STREAMLIT CAN IMPORT YOUR PROJECT MODULES
# ============================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

print("LOADED ROOT DIR:", ROOT_DIR)

# Now imports will work correctly
from data.synthetic import SyntheticMarket
from models.zoo import TabularModel
from features.regimes import RegimeDetector


# ============================================================
# STREAMLIT UI SETUP
# ============================================================

st.set_page_config(
    page_title="Quant-Core AI Dashboard",
    layout="wide",
)

st.title("🚀 Quant-Core AI Trading Dashboard")


# ============================================================
# INITIALISE MODELS
# ============================================================

market = SyntheticMarket()
regime_detector = RegimeDetector()

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Controls")

if st.sidebar.button("Generate Price Tick"):
    price = market.get_tick()
    st.session_state.history.append(price)

if st.sidebar.button("Run Prediction"):
    st.session_state.run_prediction = True


# ============================================================
# MAIN CHART TAB
# ============================================================

tab1, tab2, tab3 = st.tabs(["📈 Market Chart", "🤖 AI Model", "ℹ About"])


# ====================== TAB 1 ===============================
with tab1:
    st.subheader("Market Price History")

    if len(st.session_state.history) == 0:
        st.info("Click 'Generate Price Tick' to begin simulation.")
    else:
        df = pd.DataFrame(st.session_state.history, columns=["Price"])
        st.line_chart(df["Price"])


# ====================== TAB 2 ===============================
with tab2:
    st.subheader("AI Prediction Engine")

    history = st.session_state.history

    if len(history) < 60:
        st.warning("Need at least **60 ticks** before running prediction.")
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
        col3.metric("Regime", str(regime))

        st.write("---")

        if prob > 0.6:
            st.success("📈 BUY CALL — Bullish signal detected!")
        else:
            st.info("⏸ HOLD — No strong directional signal.")


# ====================== TAB 3 ===============================
with tab3:
    st.subheader("About Quant-Core")
    st.write("""
    Quant-Core is an AI-driven modular trading framework featuring:

    - Synthetic market engine  
    - LightGBM + ML prediction models  
    - Market regime detection  
    - Backtesting engine  
    - FastAPI inference server  
    - Streamlit dashboard  

    This dashboard runs entirely in **demo mode** using synthetic price data.
    """)

st.write("---")
st.caption("Quant-Core AI (Demo Build)")
