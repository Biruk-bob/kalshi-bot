import ast
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ----------------------------
# CONFIG
# ----------------------------
GAMMA_BASE = "https://gamma-api.polymarket.com"
BINANCE_BASE = "https://data-api.binance.vision"

ENTRY_START_SEC = 120   # wait first 2 minutes
ENTRY_END_SEC = 480     # no new entries after minute 8
REFRESH_DEFAULT = 10
HTTP_TIMEOUT = 20


# ----------------------------
# BASIC HELPERS
# ----------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    text = str(value).strip()
    try:
        if text.endswith("Z"):
            text = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def get_first_nonempty(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, "", [], {}):
            return d[k]
    return default


def parse_jsonish_list(value: Any) -> List[Any]:
    """
    Polymarket Gamma often returns list-like fields as JSON strings.
    This safely turns them into Python lists.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            try:
                import json
                parsed = json.loads(text)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []

    return []


def polymarket_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    try:
        r = requests.get(f"{GAMMA_BASE}{path}", params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ----------------------------
# BINANCE DATA
# ----------------------------
def get_binance_klines(symbol: str = "BTCUSDT", interval: str = "1m", limit: int = 180) -> pd.DataFrame:
    try:
        r = requests.get(
            f"{BINANCE_BASE}/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=HTTP_TIMEOUT,
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

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.dropna().reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 60:
        return df

    df = df.copy()

    # EMAs
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


def estimate_up_probability(df: pd.DataFrame) -> Tuple[Optional[float], str]:
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

    # Trend
    if last["ema_fast"] > last["ema_slow"]:
        bull += 1
        reasons.append("EMA bullish")
    elif last["ema_fast"] < last["ema_slow"]:
        bear += 1
        reasons.append("EMA bearish")

    # Momentum
    if last["macd"] > last["macd_signal"] and last["macd"] >= prev["macd"]:
        bull += 1
        reasons.append("MACD bullish")
    elif last["macd"] < last["macd_signal"] and last["macd"] <= prev["macd"]:
        bear += 1
        reasons.append("MACD bearish")

    # RSI zone
    if 54 <= last["rsi"] <= 66:
        bull += 1
        reasons.append("RSI bullish zone")
    elif 34 <= last["rsi"] <= 46:
        bear += 1
        reasons.append("RSI bearish zone")

    # Stochastic timing
    if last["stoch_k"] > last["stoch_d"] and 35 <= last["stoch_k"] <= 78:
        bull += 1
        reasons.append("Stochastic bullish")
    elif last["stoch_k"] < last["stoch_d"] and 22 <= last["stoch_k"] <= 65:
        bear += 1
        reasons.append("Stochastic bearish")

    # Volatility filter
    atr_pct = (last["atr"] / last["close"]) if last["close"] else 0
    vol_ok = atr_pct >= 0.0008
    if vol_ok:
        reasons.append("Volatility OK")

    # Very conservative mapping
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
# POLYMARKET MARKET DISCOVERY
# ----------------------------
def get_active_markets(limit: int = 100, max_pages: int = 8) -> List[Dict[str, Any]]:
    """
    Uses the public Gamma markets endpoint with offset paging.
    """
    all_rows: List[Dict[str, Any]] = []

    for page in range(max_pages):
        offset = page * limit
        data = polymarket_get(
            "/markets",
            params={
                "active": "true",
                "closed": "false",
                "limit": limit,
                "offset": offset,
            },
        )

        if not isinstance(data, list) or not data:
            break

        all_rows.extend(data)

        if len(data) < limit:
            break

    return all_rows


def build_market_text(m: Dict[str, Any]) -> str:
    pieces = [
        str(get_first_nonempty(m, ["question", "title", "slug"], "")),
        str(get_first_nonempty(m, ["description"], "")),
        str(get_first_nonempty(m, ["slug"], "")),
    ]
    return " ".join(pieces).strip().lower()


def normalize_market(m: Dict[str, Any]) -> Dict[str, Any]:
    question = get_first_nonempty(m, ["question", "title", "slug"], "Unknown market")
    slug = get_first_nonempty(m, ["slug"], "")
    market_id = str(get_first_nonempty(m, ["id", "conditionId", "questionID"], slug or question))

    start_dt = parse_dt(get_first_nonempty(m, ["startDate", "start_date", "createdAt", "created_at"]))
    end_dt = parse_dt(get_first_nonempty(m, ["endDate", "end_date", "umaEndDate", "uma_end_date"]))

    outcomes = parse_jsonish_list(m.get("outcomes"))
    outcome_prices = [safe_float(x) for x in parse_jsonish_list(m.get("outcomePrices"))]

    # Pad if malformed
    if len(outcome_prices) < len(outcomes):
        outcome_prices += [None] * (len(outcomes) - len(outcome_prices))

    outcome_map = {}
    for i, name in enumerate(outcomes):
        outcome_map[str(name)] = outcome_prices[i] if i < len(outcome_prices) else None

    text = build_market_text(m)

    return {
        "id": market_id,
        "question": str(question),
        "slug": str(slug),
        "start_dt": start_dt,
        "end_dt": end_dt,
        "outcomes": [str(x) for x in outcomes],
        "outcome_prices": outcome_prices,
        "outcome_map": outcome_map,
        "raw": m,
        "text": text,
    }


def looks_like_target_market(asset: str, text: str) -> bool:
    text = text.lower()

    if asset == "BTC":
        asset_ok = ("bitcoin" in text) or ("btc" in text)
    else:
        asset_ok = ("ethereum" in text) or ("ether" in text) or ("eth" in text)

    short_window_ok = (
        ("15 minute" in text)
        or ("15-minute" in text)
        or ("15m" in text)
        or ("15 min" in text)
    )

    direction_ok = ("up or down" in text) or ("up/down" in text) or ("price to beat" in text)

    return asset_ok and (short_window_ok or direction_ok)


def choose_current_polymarket(asset: str) -> Optional[Dict[str, Any]]:
    now = now_utc()
    markets = [normalize_market(m) for m in get_active_markets()]

    candidates: List[Dict[str, Any]] = []

    for m in markets:
        if not looks_like_target_market(asset, m["text"]):
            continue

        end_dt = m["end_dt"]
        if end_dt is None:
            continue

        if end_dt <= now:
            continue

        start_dt = m["start_dt"]
        if start_dt is not None and start_dt > now:
            continue

        seconds_left = int((end_dt - now).total_seconds())

        # We only care about the current/near-current short market
        if seconds_left > 20 * 60:
            continue

        m["seconds_left"] = seconds_left
        m["elapsed"] = int((now - start_dt).total_seconds()) if start_dt else None
        candidates.append(m)

    if not candidates:
        return None

    # Pick the live market ending soonest
    candidates.sort(key=lambda x: x["seconds_left"])
    return candidates[0]


# ----------------------------
# POLYMARKET OUTCOME LOGIC
# ----------------------------
def get_up_down_labels(market: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    outcomes = market.get("outcomes", [])
    lower_map = {o.lower(): o for o in outcomes}

    up_label = None
    down_label = None

    for key in ["up", "yes"]:
        if key in lower_map:
            up_label = lower_map[key]
            break

    for key in ["down", "no"]:
        if key in lower_map:
            down_label = lower_map[key]
            break

    if up_label is None and len(outcomes) >= 1:
        up_label = outcomes[0]
    if down_label is None and len(outcomes) >= 2:
        down_label = outcomes[1]

    return up_label, down_label


def get_market_up_probability(market: Dict[str, Any]) -> Optional[float]:
    up_label, _ = get_up_down_labels(market)
    if not up_label:
        return None

    price = market["outcome_map"].get(up_label)
    if price is None:
        return None

    try:
        p = float(price)
        # Gamma prices are typically 0.00–1.00, but if some feed is 0–100 we normalize.
        if p > 1.0:
            p = p / 100.0
        return p
    except Exception:
        return None


def decision_text_for_outcome(market: Dict[str, Any], bullish: bool) -> str:
    up_label, down_label = get_up_down_labels(market)
    if bullish:
        return f"BUY {up_label.upper() if up_label else 'UP'}"
    return f"BUY {down_label.upper() if down_label else 'DOWN'}"


# ----------------------------
# LOCKED DECISION PER EXACT MARKET
# ----------------------------
def decide_for_market(market: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    elapsed = market.get("elapsed")

    # If start time isn't present, we fall back to "evaluate now"
    if elapsed is not None:
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

    market_up = get_market_up_probability(market)
    if market_up is None:
        return {
            "signal": "IGNORE THIS MARKET",
            "reason": "Polymarket live outcome price unavailable",
            "locked": True,
        }

    edge = model_up - market_up

    if model_up >= 0.60 and edge >= 0.08:
        return {
            "signal": decision_text_for_outcome(market, bullish=True),
            "reason": f"{model_reason} | Model UP={model_up:.0%}, Market UP={market_up:.0%}",
            "locked": True,
        }

    if model_up <= 0.40 and (-edge) >= 0.08:
        return {
            "signal": decision_text_for_outcome(market, bullish=False),
            "reason": f"{model_reason} | Model DOWN={1-model_up:.0%}, Market DOWN={1-market_up:.0%}",
            "locked": True,
        }

    return {
        "signal": "IGNORE THIS MARKET",
        "reason": f"No strong edge | Model UP={model_up:.0%}, Market UP={market_up:.0%}",
        "locked": True,
    }


def locked_key(asset: str, market: Dict[str, Any]) -> str:
    return f"{asset}:{market['id']}"


def get_locked_decision(asset: str, market: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    if "locked_decisions" not in st.session_state:
        st.session_state.locked_decisions = {}

    key = locked_key(asset, market)

    if key in st.session_state.locked_decisions:
        return st.session_state.locked_decisions[key]

    result = decide_for_market(market, df)

    if result["locked"]:
        st.session_state.locked_decisions[key] = result

    return result


# ----------------------------
# DISPLAY HELPERS
# ----------------------------
def extract_price_to_beat(text: str) -> Optional[float]:
    patterns = [
        r"above\s+\$?([0-9,]+(?:\.[0-9]+)?)",
        r"below\s+\$?([0-9,]+(?:\.[0-9]+)?)",
        r"over\s+\$?([0-9,]+(?:\.[0-9]+)?)",
        r"under\s+\$?([0-9,]+(?:\.[0-9]+)?)",
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


def render_asset(asset: str, symbol: str) -> None:
    market = choose_current_polymarket(asset)

    st.subheader(asset)

    if market is None:
        st.write("### NO LIVE MARKET FOUND")
        st.caption("Could not find an active Polymarket 15-minute market for this asset.")
        return

    df = get_binance_klines(symbol=symbol, interval="1m", limit=180)
    df = add_indicators(df)
    result = get_locked_decision(asset, market, df)

    up_label, down_label = get_up_down_labels(market)
    market_up = get_market_up_probability(market)
    target = extract_price_to_beat(market["question"])

    st.write(f"**Current market:** {market['question']}")
    if target is not None:
        st.write(f"**Price to beat / target:** {target:,.2f}")
    st.write(f"**Market ID:** `{market['id']}`")
    if market["slug"]:
        st.write(f"**Slug:** `{market['slug']}`")

    if market["start_dt"] and market["end_dt"]:
        st.write(
            f"**Market window:** {market['start_dt'].strftime('%H:%M:%S')} - "
            f"{market['end_dt'].strftime('%H:%M:%S')} UTC"
        )
    elif market["end_dt"]:
        st.write(f"**Ends at:** {market['end_dt'].strftime('%H:%M:%S')} UTC")

    st.write(f"**Outcomes:** {up_label} / {down_label}")
    st.write(f"**Signal:** ### {result['signal']}")
    st.caption(result["reason"])
    st.caption("LOCKED" if result["locked"] else "NOT LOCKED YET")

    if market["seconds_left"] is not None:
        mins = market["seconds_left"] // 60
        secs = market["seconds_left"] % 60
        st.caption(f"Time left: {mins:02d}:{secs:02d}")

    if market_up is not None:
        st.caption(f"Live market UP price: {market_up:.0%}")

    if market["outcomes"] and market["outcome_prices"]:
        pairs = []
        for name in market["outcomes"]:
            price = market["outcome_map"].get(name)
            if price is not None:
                p = float(price)
                if p > 1.0:
                    p = p / 100.0
                pairs.append(f"{name}: {p:.0%}")
        if pairs:
            st.caption("Outcome prices — " + " | ".join(pairs))


# ----------------------------
# APP
# ----------------------------
st.set_page_config(page_title="Polymarket 15m Locked Signal Bot", layout="centered")

st.title("Polymarket 15-Min Locked Signal Bot")
st.write("One decision only per exact live Polymarket market")

refresh = st.slider("Refresh every (seconds)", 5, 60, REFRESH_DEFAULT)

left_col, right_col = st.columns(2)

with left_col:
    render_asset("BTC", "BTCUSDT")

with right_col:
    render_asset("ETH", "ETHUSDT")

time.sleep(refresh)
st.rerun()
