import os
import sys

# Add project root to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import pandas as pd
import numpy as np
from data.synthetic import SyntheticMarket
from models.zoo import TabularModel
from features.regimes import RegimeDetector

st.set_page_config(page_title="Quant-Core Dashboard", layout="wide")

st.title("📈 Quant-Core Options Trading Dashboard")

# ---------------------------------
# Initialize system
# ---------------------------------
market = SyntheticMarket()
rd = RegimeDetector()

if "price_history" not in st.session_state:
    st.session_state.price_history = []

if "position" not in st.session_state:
    st.session_state.position = None  # store active trade

# Sidebar
st.sidebar.header("Controls")
run_tick = st.sidebar.button("Generate Tick")
run_predict = st.sidebar.button("Run ML Prediction")

# ---------------------------------
# Step 1: Generate market tick
# ---------------------------------
if run_tick:
    price = market.get_tick()
    st.session_state.price_history.append(price)

if len(st.session_state.price_history) == 0:
    st.info("Click 'Generate Tick' to start simulation.")
    st.stop()

df = pd.DataFrame(st.session_state.price_history, columns=["price"])
df["returns"] = df["price"].pct_change()

st.subheader("📊 Price History")
st.line_chart(df["price"])

# ---------------------------------
# Step 2: Generate option chain (synthetic)
# ---------------------------------
latest_price = st.session_state.price_history[-1]
chain = market.generate_option_chain(latest_price, "2024-01-01")

st.subheader("📝 Option Chain (ATM Highlighted)")
chain_display = chain.copy()
chain_display["mid"] = (chain_display["bid"] + chain_display["ask"]) / 2

# Identify ATM call
atm_idx = (chain_display["strike"] - latest_price).abs().idxmin()
atm_option = chain_display.loc[atm_idx]

st.dataframe(chain_display)

st.write(f"🎯 **ATM Call Selected:** Strike {atm_option['strike']} | Mid ${atm_option['mid']:.2f}")

# ---------------------------------
# Step 3: ML Prediction + Signal
# ---------------------------------
st.subheader("🤖 Trading Signal")

if run_predict and len(df) > 60:
    model = TabularModel()

    df_hist = df.copy()
    df_hist["target"] = (df_hist["price"].shift(-5) > df_hist["price"]).astype(int)
    df_hist = df_hist.dropna()

    model.train(df_hist[["price"]], df_hist["target"])

    prob = model.predict([[latest_price]])[0]
    regime = rd.predict(df_hist["returns"].dropna())

    st.metric("Latest Price", f"${latest_price:.2f}")
    st.metric("Prob Up", f"{prob:.2%}")
    st.metric("Regime", regime)

    buy_signal = (prob > 0.6 and regime != 2)

    if st.session_state.position is None:  
        if buy_signal:
            st.success("🔥 BUY SIGNAL TRIGGERED — Buying ATM Call")

            st.session_state.position = {
                "strike": atm_option["strike"],
                "entry_price": atm_option["ask"],
                "type": "CALL",
                "qty": 1
            }
        else:
            st.info("HOLD — No Buy Signal")
    else:
        st.info("Already in a position → checking sell conditions...")

        # Compute live P&L
        current_mid = atm_option["mid"]
        entry = st.session_state.position["entry_price"]
        pnl_pct = (current_mid - entry) / entry * 100

        st.metric("Position P/L %", f"{pnl_pct:.2f}%")

        if pnl_pct > 20:
            st.success("🚀 SELL SIGNAL — Profit Target Hit! Closing Position.")
            st.session_state.position = None

        elif pnl_pct < -10:
            st.error("⚠️ STOP LOSS — Selling to prevent more loss.")
            st.session_state.position = None

else:
    st.info("Generate ticks → then click Run Prediction (requires >60 ticks).")

# ---------------------------------
# Display active position
# ---------------------------------
st.write("---")
st.subheader("📦 Current Position")

if st.session_state.position:
    st.json(st.session_state.position)
else:
    st.info("No active positions.")

st.write("---")
st.caption("Quant-Core Live Options Signal Simulator")
