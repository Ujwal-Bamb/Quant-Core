import os
print(">>> RUNNING FROM:", os.getcwd())

import os
import sys
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

print(">>> ROOT:", ROOT)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load synthetic.py manually
synthetic = load(os.path.join(ROOT, "data", "synthetic.py"), "synthetic")
# Load zoo.py manually
zoo = load(os.path.join(ROOT, "models", "zoo.py"), "zoo")
# Load regimes.py manually
regimes = load(os.path.join(ROOT, "features", "regimes.py"), "regimes")

SyntheticMarket = synthetic.SyntheticMarket
TabularModel = zoo.TabularModel
RegimeDetector = regimes.RegimeDetector


# ============================================================
# STREAMLIT STARTS HERE
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Quant-Core Dashboard", layout="wide")
st.title("🚀 Quant-Core AI Trading Dashboard")

market = SyntheticMarket()
regime_detector = RegimeDetector()

if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
if st.sidebar.button("Generate Tick"):
    st.session_state.history.append(market.get_tick())

predict = st.sidebar.button("Run Prediction")

# Tabs
tab1, tab2 = st.tabs(["📊 Market Chart", "🤖 Model Output"])

with tab1:
    st.subheader("Market Price History")
    if len(st.session_state.history) == 0:
        st.info("Click Generate Tick to start")
    else:
        df = pd.DataFrame(st.session_state.history, columns=["price"])
        st.line_chart(df["price"])

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
