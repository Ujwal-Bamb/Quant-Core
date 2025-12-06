import streamlit as st
import pandas as pd
import numpy as np
import requests
from data.synthetic import SyntheticMarket
from models.zoo import TabularModel
from features.regimes import RegimeDetector

st.set_page_config(page_title="Quant-Core Dashboard", layout="wide")

st.title("📈 Quant-Core Live Trading Dashboard")

# -------------------------------
# MODEL SETUP (Simple for demo)
# -------------------------------
market = SyntheticMarket()
rd = RegimeDetector()

st.sidebar.header("Controls")
run_simulation = st.sidebar.button("Generate Synthetic Price Tick")
run_model = st.sidebar.button("Run Prediction")

# Session state
if "price_history" not in st.session_state:
    st.session_state.price_history = []

# ---------------------------------
# Generate Synthetic Market Tick
# ---------------------------------
if run_simulation:
    new_price = market.get_tick()
    st.session_state.price_history.append(new_price)

# Convert to DataFrame
if len(st.session_state.price_history) > 0:
    df = pd.DataFrame(st.session_state.price_history, columns=["price"])
    df["returns"] = df["price"].pct_change()

    st.subheader("📊 Price History")
    st.line_chart(df["price"])

# ---------------------------------
# Prediction Example
# ---------------------------------
if run_model and len(st.session_state.price_history) > 60:
    # Simple model use
    model = TabularModel()

    # Fake training
    df_hist = pd.DataFrame(st.session_state.price_history, columns=['close'])
    df_hist["target"] = (df_hist["close"].shift(-5) > df_hist["close"]).astype(int)
    df_hist = df_hist.dropna()

    model.train(df_hist[["close"]], df_hist["target"])

    # Run prediction on latest tick
    latest_price = st.session_state.price_history[-1]
    prob = model.predict([[latest_price]])[0]

    regime = rd.predict(df_hist["close"].pct_change().dropna())

    st.subheader("🤖 Model Output")
    st.metric("Latest Price", f"${latest_price:,.2f}")
    st.metric("Probability Up", f"{prob:.2%}")
    st.metric("Regime", str(regime))

    if prob > 0.6:
        st.success("Signal: BUY CALL")
    else:
        st.info("Signal: HOLD")

else:
    st.info("Need at least 60 price ticks to run model.")

st.write("---")
st.caption("Quant-Core Research Dashboard — Synthetic Demo Mode")
