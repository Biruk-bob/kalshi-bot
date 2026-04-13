import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

# Public market-data-only Binance base URL
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"


def get_binance_data(symbol="BTCUSDT", interval="1m", limit=100):
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    try:
        response = requests.get(BINANCE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        # Binance kline response should be a list of lists
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"Unexpected Binance response: {data}")

        df = pd.DataFrame(
            data,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna().reset_index(drop=True)
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"Network/API error while fetching {symbol}: {e}")
        return pd.DataFrame()

    except ValueError as e:
        st.error(f"Data error while fetching {symbol}: {e}")
        return pd.DataFrame()


def calculate_indicators(df):
    if df.empty or len(df) < 30:
        return df

    df = df.copy()

    df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    return df


def get_signal(df):
    if df.empty or len(df) < 30:
        return "IGNORE"

    last = df.iloc[-1]

    if pd.isna(last["rsi"]) or pd.isna(last["ema_fast"]) or pd.isna(last["ema_slow"]) or pd.isna(last["macd"]) or pd.isna(last["signal"]):
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
    if bearish:
        return "BUY NO"
    return "IGNORE"


st.set_page_config(page_title="Kalshi 15m Bot", layout="centered")

st.title("Kalshi 15-Min Signal Bot")
st.write("BTC & ETH Signals (BUY YES / BUY NO / IGNORE)")

refresh = st.slider("Refresh every (seconds)", 5, 60, 10)

btc_df = get_binance_data("BTCUSDT")
eth_df = get_binance_data("ETHUSDT")

if not btc_df.empty:
    btc_df = calculate_indicators(btc_df)
    btc_signal = get_signal(btc_df)
else:
    btc_signal = "IGNORE"

if not eth_df.empty:
    eth_df = calculate_indicators(eth_df)
    eth_signal = get_signal(eth_df)
else:
    eth_signal = "IGNORE"

st.subheader("BTC Signal")
st.write(f"### {btc_signal}")

st.subheader("ETH Signal")
st.write(f"### {eth_signal}")

st.caption("Only trade between minute 2–10 of each 15-min market")

time.sleep(refresh)
st.rerun()
