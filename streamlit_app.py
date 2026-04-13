import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
BINANCE_BASE = "https://data-api.binance.vision"

ENTRY_START_SEC = 120   # do nothing in first 2 minutes
ENTRY_END_SEC = 480     # no new entries after minute 8
DEFAULT_REFRESH = 10


# ----------------------------
# Basic helpers
# ----------------------------
def now_utc():
    return datetime.now(timezone.utc)


def parse_iso_z(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def kalshi_get(path, params=None):
    try:
        r = requests.get(f"{KALSHI_BASE}{path}", params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def binance_klines(symbol="BTCUSDT", interval="1m", limit=180):
    try:
        r = requests.get(
            f"{BINANCE_BASE}/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()

        if not isinstance(data, list) or not data:
            return pd.DataFrame()

        df = pd.DataFrame(
            data,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ],
        )

        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.dropna().reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


# ----------------------------
# Kalshi market selection
# ----------------------------
def get_open_markets(limit=1000):
    """
    Pull open markets. Uses pagination cursor if present.
    """
    all_markets = []
    cursor = None

    for _ in range(10):
        params = {"status": "open", "limit": limit}
        if cursor:
            params["cursor"] = cursor

        data = kalshi_get("/markets", params=params)
        if not data or "markets" not in data:
            break

        batch = data.get("markets", [])
        all_markets.extend(batch)

        cursor = data.get("cursor")
        if not cursor or not batch:
            break

    return all_markets


def extract_target_price(text):
    """
    Tries to extract a price target from market title text.
    """
    if not text:
        return None

    patterns = [
        r"above\s+\$?([0-9,]+(?:\.[0-9]+)?)",
        r"over\s+\$?([0-9,]+(?:\.[0-9]+)?)",
        r"at\s+or\s+above\s+\$?([0-9,]+(?:\.[0-9]+)?)",
        r"greater\s+than\s+\$?([0-9,]+(?:\.[0-9]+)?)",
        r"under\s+\$?([0-9,]+(?:\.[0-9]+)?)",
        r"below\s+\$?([0-9,]+(?:\.[0-9]+)?)",
    ]

    lower = text.lower()
    for p in patterns:
        m = re.search(p, lower)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except Exception:
                return None
    return None


def asset_match_score(asset, market):
    """
    Score markets for BTC or ETH using title/ticker text.
    """
    text = " ".join([
        str(market.get("title", "")),
        str(market.get("subtitle", "")),
        str(market.get("yes_sub_title", "")),
        str(market.get("ticker", "")),
        str(market.get("event_ticker", "")),
    ]).lower()

    score = 0

    if asset == "BTC":
        if "bitcoin" in text:
            score += 5
        if "btc" in text:
            score += 5
        if "kxbtc" in text:
            score += 8
    elif asset == "ETH":
        if "ethereum" in text:
            score += 5
        if "ether" in text:
            score += 3
        if "eth" in text:
            score += 5
        if "kxeth" in text:
            score += 8

    if "15 min" in text or "15m" in text or "15-minute" in text:
        score += 5

    if "crypto" in text:
        score += 1

    return score


def choose_live_market_for_asset(asset):
    now = now_utc()
    markets = get_open_markets()

    candidates = []

    for m in markets:
        score = asset_match_score(asset, m)
        if score <= 0:
            continue

        open_time = m.get("open_time")
        close_time = m.get("close_time")
        if not open_time or not close_time:
            continue

        try:
            open_dt = parse_iso_z(open_time)
            close_dt = parse_iso_z(close_time)
        except Exception:
            continue

        if not (open_dt <= now < close_dt):
            continue

        seconds_left = int((close_dt - now).total_seconds())
        if seconds_left <= 0 or seconds_left > 15 * 60 + 120:
            continue

        title = m.get("title") or m.get("yes_sub_title") or m.get("subtitle") or m.get("ticker", "")
        target = extract_target_price(title)

        candidates.append({
            "score": score,
            "ticker": m.get("ticker", ""),
            "title": title,
            "target": target,
            "open_dt": open_dt,
            "close_dt": close_dt,
            "yes_bid": to_float(m.get("yes_bid_dollars")),
            "yes_ask": to_float(m.get("yes_ask_dollars")),
            "no_bid": to_float(m.get("no_bid_dollars")),
            "no_ask": to_float(m.get("no_ask_dollars")),
        })

    if not candidates:
        return None

    # Best score first, then the one closing soonest
    candidates.sort(key=lambda x: (-x["score"], x["close_dt"]))
    return candidates[0]


# ----------------------------
# Indicators
# ----------------------------
def add_indicators(df):
    if df.empty or len(df) < 60:
        return df

    df = df.copy()

    df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    denom = (high14 - low14).replace(0, np.nan)
    df["stoch_k"] = 100 * (df["close"] - low14) / denom
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    return df


def estimate_up_probability(df):
    if df.empty or len(df) < 60:
        return None, "Not enough Binance data"

    last = df.iloc[-1]
    prev = df.iloc[-2]

    needed = ["ema_fast", "ema_slow", "rsi", "macd", "macd_signal", "stoch_k", "stoch_d", "atr", "close"]
    if any(pd.isna(last[c]) for c in needed):
        return None, "Indicators incomplete"

    bull = 0
    bear = 0
    reasons = []

    if last["ema_fast"] > last["ema_slow"]:
        bull += 1
        reasons.append("EMA bullish")
    elif last["ema_fast"] < last["ema_slow"]:
        bear += 1
        reasons.append("EMA bearish")

    if last["macd"] > last["macd_signal"] and last["macd"] >= prev["macd"]:
        bull += 1
        reasons.append("MACD bullish")
    elif last["macd"] < last["macd_signal"] and last["macd"] <= prev["macd"]:
        bear += 1
        reasons.append("MACD bearish")

    if 54 <= last["rsi"] <= 66:
        bull += 1
        reasons.append("RSI bullish zone")
    elif 34 <= last["rsi"] <= 46:
        bear += 1
        reasons.append("RSI bearish zone")

    if last["stoch_k"] > last["stoch_d"] and 35 <= last["stoch_k"] <= 78:
        bull += 1
        reasons.append("Stochastic bullish")
    elif last["stoch_k"] < last["stoch_d"] and 22 <= last["stoch_k"] <= 65:
        bear += 1
        reasons.append("Stochastic bearish")

    atr_pct = (last["atr"] / last["close"]) if last["close"] else 0
    vol_ok = atr_pct >= 0.0008
    if vol_ok:
        reasons.append("Volatility OK")

    if bull >= 4 and bear == 0 and vol_ok:
        return 0.66, ", ".join(reasons)
    if bull >= 3 and bear == 0 and vol_ok:
        return 0.60, ", ".join(reasons)
    if bear >= 4 and bull == 0 and vol_ok:
        return 0.34, ", ".join(reasons)
    if bear >= 3 and bull == 0 and vol_ok:
        return 0.40, ", ".join(reasons)

    return 0.50, "No strong directional edge"


# ----------------------------
# Kalshi price logic
# ----------------------------
def market_implied_up_probability(m):
    yes_bid = m.get("yes_bid")
    yes_ask = m.get("yes_ask")
    no_bid = m.get("no_bid")
    no_ask = m.get("no_ask")

    vals = []

    if yes_bid is not None and yes_ask is not None:
        vals.append((yes_bid + yes_ask) / 2.0)

    if no_bid is not None and no_ask is not None:
        vals.append(1.0 - ((no_bid + no_ask) / 2.0))

    if vals:
        return float(sum(vals) / len(vals))

    if yes_ask is not None:
        return yes_ask
    if yes_bid is not None:
        return yes_bid

    return None


# ----------------------------
# Locked one-decision-per-market
# ----------------------------
def get_decision_key(asset, market):
    return f"{asset}:{market['ticker']}"


def decide_for_market(asset, market, df):
    now = now_utc()

    elapsed = int((now - market["open_dt"]).total_seconds())
    if elapsed < ENTRY_START_SEC:
        return {
            "signal": "WAIT",
            "reason": "Waiting for first 2 minutes to pass",
            "locked": False,
        }

    if elapsed > ENTRY_END_SEC:
        return {
            "signal": "IGNORE THIS MARKET",
            "reason": "No entry after minute 8",
            "locked": True,
        }

    model_up, model_reason = estimate_up_probability(df)
    if model_up is None:
        return {
            "signal": "IGNORE THIS MARKET",
            "reason": model_reason,
            "locked": True,
        }

    market_up = market_implied_up_probability(market)
    if market_up is None:
        return {
            "signal": "IGNORE THIS MARKET",
            "reason": "Kalshi market price unavailable",
            "locked": True,
        }

    edge = model_up - market_up

    if model_up >= 0.60 and edge >= 0.08:
        return {
            "signal": "BUY YES",
            "reason": f"{model_reason} | Model UP={model_up:.0%}, Kalshi UP={market_up:.0%}",
            "locked": True,
        }

    if model_up <= 0.40 and (-edge) >= 0.08:
        return {
            "signal": "BUY NO",
            "reason": f"{model_reason} | Model DOWN={1-model_up:.0%}, Kalshi DOWN={1-market_up:.0%}",
            "locked": True,
        }

    return {
        "signal": "IGNORE THIS MARKET",
        "reason": f"No strong edge | Model UP={model_up:.0%}, Kalshi UP={market_up:.0%}",
        "locked": True,
    }


def get_locked_decision(asset, market, df):
    if "locked_decisions" not in st.session_state:
        st.session_state.locked_decisions = {}

    key = get_decision_key(asset, market)

    # Return existing lock for this exact market ticker
    if key in st.session_state.locked_decisions:
        return st.session_state.locked_decisions[key]

    result = decide_for_market(asset, market, df)

    if result["locked"]:
        st.session_state.locked_decisions[key] = result

    return result


# ----------------------------
# Rendering
# ----------------------------
def render_asset(asset, binance_symbol):
    market = choose_live_market_for_asset(asset)

    st.subheader(asset)

    if market is None:
        st.write("### NO LIVE MARKET FOUND")
        st.caption("Could not find an active Kalshi 15-minute market for this asset.")
        return

    df = binance_klines(binance_symbol, "1m", 180)
    df = add_indicators(df)
    result = get_locked_decision(asset, market, df)

    now = now_utc()
    left = max(0, int((market["close_dt"] - now).total_seconds()))
    mins = left // 60
    secs = left % 60

    st.write(f"**Current market:** {market['title']}")
    if market["target"] is not None:
        st.write(f"**Target price:** {market['target']:,.2f}")
    st.write(f"**Kalshi ticker:** `{market['ticker']}`")
    st.write(
        f"**Market window:** {market['open_dt'].strftime('%H:%M:%S')} - "
        f"{market['close_dt'].strftime('%H:%M:%S')} UTC"
    )
    st.write(f"**Signal:** ### {result['signal']}")
    st.caption(result["reason"])
    st.caption("LOCKED" if result["locked"] else "NOT LOCKED YET")
    st.caption(f"Time left: {mins:02d}:{secs:02d}")

    st.caption(
        f"Kalshi prices — YES bid/ask: {market['yes_bid']} / {market['yes_ask']} | "
        f"NO bid/ask: {market['no_bid']} / {market['no_ask']}"
    )


# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="Kalshi 15m Locked Signal Bot", layout="centered")

st.title("Kalshi 15-Min Locked Signal Bot")
st.write("One decision only per exact live Kalshi market")

refresh = st.slider("Refresh every (seconds)", 5, 60, DEFAULT_REFRESH)

left_col, right_col = st.columns(2)

with left_col:
    render_asset("BTC", "BTCUSDT")

with right_col:
    render_asset("ETH", "ETHUSDT")

time.sleep(refresh)
st.rerun()
