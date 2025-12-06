# ============================================================
# WINDOWS-PROOF IMPORT FIX — Loads modules by absolute path
# ============================================================
import os
import sys
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "synthetic.py")
MODELS_PATH = os.path.join(ROOT, "models", "zoo.py")
FEATURES_PATH = os.path.join(ROOT, "features", "regimes.py")

def load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

synthetic = load_module("synthetic", DATA_PATH)
zoo = load_module("zoo", MODELS_PATH)
regimes = load_module("regimes", FEATURES_PATH)

SyntheticMarket = synthetic.SyntheticMarket
TabularModel = zoo.TabularModel
RegimeDetector = regimes.RegimeDetector


# ============================================================
# STREAMLIT IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np


# ============================================================
# STREAMLIT PAGE CONFIG
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

# Session state for price history
if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("⚙️ Controls")
simulate = st.sidebar.button("📈 Generate Price Tick")
predict = st.sidebar.button("🤖 Run Prediction")


# ============================================================
# HANDLE PRICE TICK SIMULATION
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
        st.info("Click **Generate Price Tick** to begin simulation.")
    else:
        df = pd.DataFrame(history, columns=["Price"])
        st.line_chart(df["Price"])


# ============================================================
# TAB 2 — AI MODEL PREDICTION
# ============================================================
with tab2:
    st.header("🤖 AI Trading Signals")

    if predict and len(history) > 60:

        # Build training dataset
        df_hist = pd.DataFrame(history, columns=['close'])
        df_hist["target"] = (df_hist['close'].shift(-5) > df_hist['close']).astype(int)
        df_hist = df_hist.dropna()

        # Train model
        model = TabularModel()
        model.train(df_hist[['close']], df_hist['target'])

        # Latest values
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

        # Trading decision
        if probability_up > 0.60:
            st.success("📈 **BUY CALL** — Model expects bullish movement!")
        else:
            st.info("⏸ **HOLD** — No strong signal.")

    else:
        st.info("Generate **at least 60 ticks** before running prediction.")


# ============================================================
# TAB 3 — SYSTEM INFO
# ============================================================
with tab3:
    st.header("ℹ️ About Quant-Core")
    st.write("""
        Quant-Core is a modular AI trading system featuring:

        - 🧠 ML models (LightGBM, LSTM, Transformers)
        - 📉 Synthetic market generator (GBM)
        - 🔍 Regime detector (GMM/HMM-style)
        - 💹 Event-driven backtester
        - 🌐 FastAPI model server
        - 📊 Streamlit monitoring dashboard

        This dashboard demonstrates the simulation + prediction pipeline.
    """)

st.write("---")
st.caption("Quant-Core — Streamlit Edition (Demo Mode)")

