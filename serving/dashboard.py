# -------------------------------------------------------
# FIX: Add project root to Python path for module imports
# -------------------------------------------------------
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

# Print for debugging (optional)
print("ROOT DIR:", ROOT_DIR)

# -------------------------------------------------------
# Imports
# -------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np

from data.synthetic import SyntheticMarket
from models.zoo import TabularModel
from features.regimes import RegimeDetector

# -------------------------------------------------------
# Streamlit Page Settings
# -------------------------------------------------------
st.set_page_config(
    page_title="Quant-Core Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (Website-like UI)
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    h1, h2, h3 { color: #fafafa !important; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Title
# -------------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🚀 Quant-Core AI Trading Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#ccc;'>Synthetic Market • AI Predictions • Regime Detection</p>", unsafe_allow_html=True)

# -------------------------------------------------------
# Initialize Objects
# -------------------------------------------------------
market = SyntheticMarket()
rd = RegimeDetector()

# Sidebar Controls
st.sidebar.header("Controls")
simulate = st.sidebar.button("📈 Generate Price Tick")
predict = st.sidebar.button("🤖 Run Prediction")

# Session State
if "history" not in st.session_state:
    st.session_state.history = []

# Add synthetic price
if simulate:
    price = market.get_tick()
    st.session_state.history.append(price)

# DataFrame
history = st.session_state.history

# -------------------------------------------------------
# Dashboard Tabs
# -------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Market Chart", "🤖 AI Model Output", "ℹ️ System Info"])

# ---------------- TAB 1: Chart --------------------------
with tab1:
    st.header("Market Price Chart")

    if len(history) == 0:
        st.info("Click 'Generate Price Tick' to begin simulation.")
    else:
        df = pd.DataFrame(history, columns=["Price"])
        st.line_chart(df["Price"])

# ---------------- TAB 2: Prediction ---------------------
with tab2:
    st.header("AI Trading Signal")

    if predict and len(history) > 60:

        # Prepare training data
        df_hist = pd.DataFrame(history, columns=['close'])
        df_hist["target"] = (df_hist['close'].shift(-5) > df_hist['close']).astype(int)
        df_hist = df_hist.dropna()

        # Train TabularModel
        model = TabularModel()
        model.train(df_hist[['close']], df_hist['target'])

        # Predict
        latest_price = history[-1]
        probability_up = model.predict([[latest_price]])[0]
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
            st.success("📈 BUY CALL — Model expects bullish movement")
        else:
            st.info("⏸ HOLD — No strong signal")

    else:
        st.info("Generate at least 60 price ticks to run predictions.")

# ---------------- TAB 3: Info ---------------------------
with tab3:
    st.header("About Quant-Core")
    st.write("""
        Quant-Core is a modular AI-driven trading framework featuring:

        - 🧠 Machine learning (LightGBM / LSTM / Transformers)
        - 📊 Synthetic market generation (GBM)
        - 🔍 Regime classification (GMM/HMM)
        - 💹 Backtesting engine
        - 🌐 FastAPI inference server
        - 📈 Streamlit dashboard

        This dashboard demonstrates real-time synthetic market simulation and
        AI-powered trading signals.
    """)

st.write("---")
st.caption("Quant-Core Research Dashboard — Streamlit Edition")
