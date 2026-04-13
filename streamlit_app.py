import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

# =========================
# CONFIG
# =========================
BINANCE_URL = "https://api.binance.com/api/v3/klines"

# =========================
# FETCH DATA
# =========================
def get_binance_data(symbol="BTCUSDT", interval="1m", limit=100):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = requests.get(BINANCE_URL, params=params).json()
    df = pd.DataFrame(data)
    df = df.iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["close"] = df["close"].astype(float)
    return df

# =========================
# INDICATORS
# =========================
def calculate_indicators(df):
    df["ema_fast"] = df["close"].ewm(span=9).mean()
    df["ema_slow"] = df["close"].ewm(span=21).mean()

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["signal"] = df["macd"].ewm(span=9).mean()

    return df

# =========================
# DECISION LOGIC
# =========================
def get_signal(df):
    last = df.iloc[-1]

    if np.isnan(last["rsi"]):
        return "IGNORE"

    bullish = (
        last["ema_fast"] > last["ema_slow"]
        and last["macd"] > last["signal"]
        and 50 < last["rsi"] < 70
    )

    bearish = (
        last["ema_fast"] < last["ema_slow"]
        and last["macd"] < last["signal"]
        and 30 < last["rsi"] < 50
    )

    if bullish:
        return "BUY YES"
    elif bearish:
        return "BUY NO"
    else:
        return "IGNORE"

# =========================
# UI
# =========================
st.set_page_config(page_title="Kalshi 15m Bot", layout="centered")

st.title("Kalshi 15-Min Signal Bot")
st.write("BTC & ETH Signals (BUY YES / BUY NO / IGNORE)")

refresh = st.slider("Refresh every (seconds)", 5, 60, 10)

btc_df = get_binance_data("BTCUSDT")
btc_df = calculate_indicators(btc_df)
btc_signal = get_signal(btc_df)

eth_df = get_binance_data("ETHUSDT")
eth_df = calculate_indicators(eth_df)
eth_signal = get_signal(eth_df)

st.subheader("BTC Signal")
st.write(f"### {btc_signal}")

st.subheader("ETH Signal")
st.write(f"### {eth_signal}")

st.caption("Only trade between minute 2–10 of each 15-min market")

time.sleep(refresh)
st.rerun()