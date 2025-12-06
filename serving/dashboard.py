# serving/dashboard.py
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.data_loader import download_ohlcv
from models.train import train_model, make_features
from models.backtest import generate_signals, backtest
import joblib

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Quant-Core Trading MVP", layout="wide")
st.title("Quant-Core Trading MVP")

# Controls
symbol = st.sidebar.text_input("Symbol", value="AAPL")
interval = st.sidebar.selectbox("Interval", ["1m","5m","15m","1h"], index=1)
period = st.sidebar.selectbox("Period", ["7d","30d","60d","120d"], index=2)

if st.sidebar.button("Load Data & Train"):
    with st.spinner("Downloading ..."):
        df = download_ohlcv(symbol, period=period, interval=interval)
    st.session_state['df'] = df
    with st.spinner("Training model ..."):
        model = train_model(df, save_path="models/lgb_model.pkl")
    st.success("Trained and saved model (models/lgb_model.pkl)")

if "df" not in st.session_state:
    try:
        st.session_state['df'] = download_ohlcv(symbol, period="30d", interval="5m")
    except Exception as e:
        st.error("Data load failed: " + str(e))
        st.stop()

df = st.session_state['df']
st.subheader(f"{symbol} price (last {len(df)} bars)")
st.line_chart(df['close'])

# Run predict/backtest
if st.sidebar.button("Run Predict & Backtest"):
    with st.spinner("Predicting..."):
        sig, probs = generate_signals(df, model_path="models/lgb_model.pkl")
        res = backtest(df, sig)
    st.metric("Total Return", f"{res['total_return']:.2f}")
    st.metric("Sharpe", f"{res['sharpe']:.2f}")
    st.subheader("Latest signal & prob")
    st.write("Signal:", sig.iloc[-1], "Prob up:", float(probs[-1]))
    st.subheader("Equity Curve")
    st.line_chart(res['equity'])

st.caption("Paper trading only. Always test before going live.")
