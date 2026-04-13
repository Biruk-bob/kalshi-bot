import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time

# ----------------------------
# CONFIG
# ----------------------------
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"

ENTRY_START_SEC = 120   # wait first 2 min
ENTRY_END_SEC = 480     # stop new entries after 8 min
REFRESH_DEFAULT = 10

# ----------------------------
# TIME HELPERS
# ----------------------------
def now_utc():
    return datetime.now(timezone.utc)

def parse_iso(dt_str):
    # Handles Kalshi ISO timestamps like 2026-04-13T15:30:00Z
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))

def current_15m_window(now=None):
    if now is None:
        now = now_utc()
    minute_block = (now.minute // 15) * 15
    start = now.replace(minute=minute_block, second=0, microsecond=0)
    end = start + timedelta(minutes=15)
    return start, end

def fmt_window(start_dt, end_dt):
    return f"{start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')} UTC"

def secs_since_open(now, start_dt):
    return max(0, int((now - start_dt).total_seconds()))

def secs_to_close(now, end_dt):
    return max(0, int((end_dt - now).total_seconds()))

# ----------------------------
# KALSHI
# ----------------------------
def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def kalshi_get(path, params=None):
    try:
        r = requests.get(f"{KALSHI_BASE}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def get_open_markets_closing_soon(limit=200, within_minutes=20):
    now = now_utc()
    max_close_ts = int((now + timedelta(minutes=within_minutes)).timestamp())
    params = {
        "status": "open",
        "max_close_ts": max_close_ts,
        "limit": limit,
        "mve_filter": "exclude",
    }
    data = kalshi_get("/markets", params=params)
    if not data or "markets" not in data:
        return []
    return data["markets"]

def get_event(event_ticker):
    data = kalshi_get(f"/events/{event_ticker}")
    if not data or "event" not in data:
        return None
    return data["event"]

def asset_from_event_and_market(event_title, market):
    text = f"{event_title} {market.get('yes_sub_title','')} {market.get('ticker','')}".lower()
    if "bitcoin" in text or "btc" in text:
        return "BTC"
    if "ether" in text or "ethereum" in text or "eth" in text:
        return "ETH"
    return None

def human_market_name(event, market):
    event_title = event.get("title", "Unnamed Kalshi market")
    yes_sub = market.get("yes_sub_title", "")
    if yes_sub:
        return f"{event_title} — {yes_sub}"
    return event_title

def choose_current_asset_market(asset):
    """
    Pick the best currently-open Kalshi market for BTC or ETH that closes soon.
    Prefer markets whose close time matches the current 15m cycle most closely.
    """
    now = now_utc()
    markets = get_open_markets_closing_soon(limit=300, within_minutes=20)

    candidates = []
    for m in markets:
        event_ticker = m.get("event_ticker")
        if not event_ticker:
            continue
        event = get_event(event_ticker)
        if not event:
            continue

        detected_asset = asset_from_event_and_market(event.get("title", ""), m)
        if detected_asset != asset:
            continue

        close_time = m.get("close_time")
        open_time = m.get("open_time")
        if not close_time or not open_time:
            continue

        try:
            close_dt = parse_iso(close_time)
            open_dt = parse_iso(open_time)
        except Exception:
            continue

        if not (open_dt <= now <= close_dt):
            continue

        # Prefer the market nearest to current window end
        _, current_end = current_15m_window(now)
        distance = abs((close_dt - current_end).total_seconds())

        candidates.append((distance, close_dt, open_dt, event, m))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]))
    _, close_dt, open_dt, event, market = candidates[0]

    return {
        "asset": asset,
        "event": event,
        "market": market,
        "open_dt": open_dt,
        "close_dt": close_dt,
        "market_name": human_market_name(event, market),
        "yes_bid": safe_float(market.get("yes_bid_dollars")),
        "yes_ask": safe_float(market.get("yes_ask_dollars")),
        "no_bid": safe_float(market.get("no_bid_dollars")),
        "no_ask": safe_float(market.get("no_ask_dollars")),
        "ticker": market.get("ticker", ""),
    }

# ----------------------------
# BINANCE
# ----------------------------
def get_binance_data(symbol="BTCUSDT", interval="1m", limit=120):
    try:
        r = requests.get(
            BINANCE_URL,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

        if not isinstance(data, list) or not data:
            return pd.DataFrame()

        df = pd.DataFrame(
            data,
            columns=[
                "open_time","open","high","low","close","volume",
                "close_time","quote_asset_volume","number_of_trades",
                "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
            ],
        )
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.dropna().reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()

def add_indicators(df):
    if df.empty or len(df) < 60:
        return df

    df = df.copy()

    # EMA
    df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Stochastic
    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    denom = (high14 - low14).replace(0, np.nan)
    df["stoch_k"] = 100 * (df["close"] - low14) / denom
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ATR
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    return df

# ----------------------------
# SIGNAL ENGINE
# ----------------------------
def market_implied_up_probability(market_info):
    yes_bid = market_info.get("yes_bid")
    yes_ask = market_info.get("yes_ask")
    no_bid = market_info.get("no_bid")
    no_ask = market_info.get("no_ask")

    mids = []
    if yes_bid is not None and yes_ask is not None:
        mids.append((yes_bid + yes_ask) / 2)
    if no_bid is not None and no_ask is not None:
        mids.append(1 - ((no_bid + no_ask) / 2))
    if mids:
        return sum(mids) / len(mids)

    if yes_bid is not None:
        return yes_bid
    if yes_ask is not None:
        return yes_ask
    return None

def estimate_up_probability(df):
    if df.empty or len(df) < 60:
        return None, "Not enough price data"

    last = df.iloc[-1]
    prev = df.iloc[-2]

    needed = ["ema_fast","ema_slow","rsi","macd","macd_signal","stoch_k","stoch_d","atr","close"]
    if any(pd.isna(last[c]) for c in needed):
        return None, "Indicators incomplete"

    bullish_score = 0
    bearish_score = 0
    reasons = []

    # Trend
    if last["ema_fast"] > last["ema_slow"]:
        bullish_score += 1
        reasons.append("EMA bullish")
    elif last["ema_fast"] < last["ema_slow"]:
        bearish_score += 1
        reasons.append("EMA bearish")

    # MACD
    if last["macd"] > last["macd_signal"] and last["macd"] >= prev["macd"]:
        bullish_score += 1
        reasons.append("MACD bullish")
    elif last["macd"] < last["macd_signal"] and last["macd"] <= prev["macd"]:
        bearish_score += 1
        reasons.append("MACD bearish")

    # RSI
    if 54 <= last["rsi"] <= 66:
        bullish_score += 1
        reasons.append("RSI bullish zone")
    elif 34 <= last["rsi"] <= 46:
        bearish_score += 1
        reasons.append("RSI bearish zone")

    # Stochastic
    if last["stoch_k"] > last["stoch_d"] and 35 <= last["stoch_k"] <= 78:
        bullish_score += 1
        reasons.append("Stoch bullish")
    elif last["stoch_k"] < last["stoch_d"] and 22 <= last["stoch_k"] <= 65:
        bearish_score += 1
        reasons.append("Stoch bearish")

    # Volatility filter
    atr_pct = (last["atr"] / last["close"]) if last["close"] else 0
    enough_vol = atr_pct >= 0.0008
    if enough_vol:
        reasons.append("Volatility sufficient")

    # Convert strict rule set into estimated probability
    if bullish_score >= 4 and bearish_score == 0 and enough_vol:
        return 0.66, ", ".join(reasons)
    if bullish_score >= 3 and bearish_score == 0 and enough_vol:
        return 0.60, ", ".join(reasons)
    if bearish_score >= 4 and bullish_score == 0 and enough_vol:
        return 0.34, ", ".join(reasons)
    if bearish_score >= 3 and bullish_score == 0 and enough_vol:
        return 0.40, ", ".join(reasons)

    return 0.50, "No strong directional edge"

def decide_signal(market_info, df, now):
    open_dt = market_info["open_dt"]
    close_dt = market_info["close_dt"]

    elapsed = secs_since_open(now, open_dt)
    remaining = secs_to_close(now, close_dt)

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

    est_prob_up, model_reason = estimate_up_probability(df)
    if est_prob_up is None:
        return {
            "signal": "IGNORE THIS MARKET",
            "reason": model_reason,
            "locked": True,
        }

    implied = market_implied_up_probability(market_info)
    if implied is None:
        return {
            "signal": "IGNORE THIS MARKET",
            "reason": "Kalshi price unavailable",
            "locked": True,
        }

    # Conservative required edge
    edge = est_prob_up - implied
    if est_prob_up >= 0.60 and edge >= 0.08:
        return {
            "signal": "BUY YES",
            "reason": f"{model_reason} | Model UP={est_prob_up:.0%}, Market UP={implied:.0%}",
            "locked": True,
        }

    if est_prob_up <= 0.40 and (-edge) >= 0.08:
        return {
            "signal": "BUY NO",
            "reason": f"{model_reason} | Model DOWN={1-est_prob_up:.0%}, Market DOWN={1-implied:.0%}",
            "locked": True,
        }

    return {
        "signal": "IGNORE THIS MARKET",
        "reason": f"No strong edge | Model UP={est_prob_up:.0%}, Market UP={implied:.0%}",
        "locked": True,
    }

# ----------------------------
# LOCK PER MARKET
# ----------------------------
def market_lock_key(asset, market_ticker):
    return f"{asset}:{market_ticker}"

def get_locked_market_decision(asset, market_info, df):
    if "locked_market_signals" not in st.session_state:
        st.session_state.locked_market_signals = {}

    key = market_lock_key(asset, market_info["ticker"])

    # keep only recent locks
    active_keys = set()
    for a in ["BTC", "ETH"]:
        mi = st.session_state.get(f"active_market_{a}")
        if mi and "ticker" in mi:
            active_keys.add(market_lock_key(a, mi["ticker"]))
    st.session_state.locked_market_signals = {
        k: v for k, v in st.session_state.locked_market_signals.items() if k in active_keys or k == key
    }

    if key in st.session_state.locked_market_signals:
        return st.session_state.locked_market_signals[key]

    now = now_utc()
    result = decide_signal(market_info, df, now)

    if result["locked"]:
        st.session_state.locked_market_signals[key] = result

    return result

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Kalshi 15m Locked Signal Bot", layout="centered")

st.title("Kalshi 15-Min Locked Signal Bot")
st.write("One decision only for each live BTC/ETH Kalshi market")

refresh = st.slider("Refresh every (seconds)", 5, 60, REFRESH_DEFAULT)

def render_asset(asset, symbol):
    market_info = choose_current_asset_market(asset)
    st.session_state[f"active_market_{asset}"] = market_info

    if not market_info:
        st.subheader(asset)
        st.write("### NO LIVE MARKET FOUND")
        st.caption("Could not find a current Kalshi market for this asset")
        return

    df = get_binance_data(symbol=symbol, interval="1m", limit=120)
    df = add_indicators(df)

    result = get_locked_market_decision(asset, market_info, df)

    now = now_utc()
    left = secs_to_close(now, market_info["close_dt"])
    mins = left // 60
    secs = left % 60

    st.subheader(asset)
    st.write(f"**Current market:** {market_info['market_name']}")
    st.write(f"**Kalshi ticker:** `{market_info['ticker']}`")
    st.write(f"**Market window:** {market_info['open_dt'].strftime('%H:%M')} - {market_info['close_dt'].strftime('%H:%M')} UTC")
    st.write(f"**Signal:** ### {result['signal']}")
    st.caption(result["reason"])
    st.caption("LOCKED" if result["locked"] else "NOT LOCKED YET")
    st.caption(f"Time left: {mins:02d}:{secs:02d}")

    yes_bid = market_info.get("yes_bid")
    yes_ask = market_info.get("yes_ask")
    no_bid = market_info.get("no_bid")
    no_ask = market_info.get("no_ask")
    st.caption(f"Kalshi prices — YES bid/ask: {yes_bid} / {yes_ask} | NO bid/ask: {no_bid} / {no_ask}")

col1, col2 = st.columns(2)

with col1:
    render_asset("BTC", "BTCUSDT")

with col2:
    render_asset("ETH", "ETHUSDT")

time.sleep(refresh)
st.rerun()
