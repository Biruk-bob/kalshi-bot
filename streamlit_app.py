import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
GAMMA_BASE = "https://gamma-api.polymarket.com"
BINANCE_BASE = "https://data-api.binance.vision"

ENTRY_START_SEC = 120   # wait first 2 minutes
ENTRY_END_SEC = 480     # stop new entries after minute 8
REFRESH_DEFAULT = 10
HTTP_TIMEOUT = 20


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


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


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            try:
                import ast
                parsed = ast.literal_eval(s)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
    return []


def pm_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    try:
        r = requests.get(f"{GAMMA_BASE}{path}", params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_first(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, "", [], {}):
            return d[k]
    return default


# -------------------------------------------------
# BINANCE
# -------------------------------------------------
def get_binance_klines(symbol: str, interval: str = "1m", limit: int = 180) -> pd.DataFrame:
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


# -------------------------------------------------
# POLYMARKET DISCOVERY
# -------------------------------------------------
def get_active_events(limit: int = 100, max_pages: int = 8) -> List[Dict[str, Any]]:
    all_events: List[Dict[str, Any]] = []

    for page in range(max_pages):
        offset = page * limit
        data = pm_get(
            "/events",
            params={
                "active": "true",
                "closed": "false",
                "limit": limit,
                "offset": offset,
            },
        )

        if not isinstance(data, list) or not data:
            break

        all_events.extend(data)

        if len(data) < limit:
            break

    return all_events


def event_text(event: Dict[str, Any]) -> str:
    parts = [
        str(get_first(event, ["title", "question", "slug"], "")),
        str(get_first(event, ["subtitle", "description"], "")),
        str(get_first(event, ["slug"], "")),
    ]
    return " ".join(parts).lower()


def market_text(market: Dict[str, Any]) -> str:
    parts = [
        str(get_first(market, ["question", "title", "slug"], "")),
        str(get_first(market, ["description"], "")),
        str(get_first(market, ["slug"], "")),
    ]
    return " ".join(parts).lower()


def asset_match(asset: str, text: str) -> bool:
    text = text.lower()
    if asset == "BTC":
        return ("btc" in text) or ("bitcoin" in text)
    return ("eth" in text) or ("ether" in text) or ("ethereum" in text)


def is_15m_updown(text: str) -> bool:
    text = text.lower()
    return (
        ("15 minute" in text)
        or ("15-minute" in text)
        or ("15m" in text)
        or ("15 min" in text)
        or ("up or down" in text)
        or ("up/down" in text)
    )


def normalize_event_market(event: Dict[str, Any], market: Dict[str, Any]) -> Dict[str, Any]:
    question = str(get_first(market, ["question", "title"], get_first(event, ["title"], "Unknown market")))
    slug = str(get_first(market, ["slug"], get_first(event, ["slug"], "")))

    start_dt = parse_dt(get_first(market, ["startDate", "start_date"], get_first(event, ["startDate", "start_date"])))
    end_dt = parse_dt(get_first(market, ["endDate", "end_date"], get_first(event, ["endDate", "end_date"])))

    outcomes = to_list(market.get("outcomes"))
    prices = [safe_float(x) for x in to_list(market.get("outcomePrices"))]

    if len(prices) < len(outcomes):
        prices += [None] * (len(outcomes) - len(prices))

    outcome_map = {}
    for i, name in enumerate(outcomes):
        outcome_map[str(name)] = prices[i] if i < len(prices) else None

    return {
        "id": str(get_first(market, ["id", "conditionId"], slug or question)),
        "question": question,
        "slug": slug,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "outcomes": [str(x) for x in outcomes],
        "outcome_prices": prices,
        "outcome_map": outcome_map,
        "event_title": str(get_first(event, ["title"], "")),
        "event_slug": str(get_first(event, ["slug"], "")),
    }


def choose_current_polymarket(asset: str) -> Optional[Dict[str, Any]]:
    now = now_utc()
    events = get_active_events()

    candidates: List[Dict[str, Any]] = []

    for event in events:
        etext = event_text(event)
        if not asset_match(asset, etext):
            continue
        if not is_15m_updown(etext):
            continue

        markets = event.get("markets", [])
        if not isinstance(markets, list) or not markets:
            continue

        for market in markets:
            mtext = market_text(market) + " " + etext
            if not asset_match(asset, mtext):
                continue
            if not is_15m_updown(mtext):
                continue

            nm = normalize_event_market(event, market)

            if nm["end_dt"] is None:
                continue
            if nm["end_dt"] <= now:
                continue
            if nm["start_dt"] is not None and nm["start_dt"] > now:
                continue

            seconds_left = int((nm["end_dt"] - now).total_seconds())
            if seconds_left > 20 * 60:
                continue

            elapsed = int((now - nm["start_dt"]).total_seconds()) if nm["start_dt"] else None

            nm["seconds_left"] = seconds_left
            nm["elapsed"] = elapsed
            candidates.append(nm)

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["seconds_left"])
    return candidates[0]


# -------------------------------------------------
# POLYMARKET PRICING
# -------------------------------------------------
def get_yes_no_labels(market: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    outcomes = market.get("outcomes", [])
    lower_map = {o.lower(): o for o in outcomes}

    yes_label = lower_map.get("yes")
    no_label = lower_map.get("no")

    if yes_label is None and len(outcomes) >= 1:
        yes_label = outcomes[0]
    if no_label is None and len(outcomes) >= 2:
        no_label = outcomes[1]

    return yes_label, no_label


def get_market_yes_probability(market: Dict[str, Any]) -> Optional[float]:
    yes_label, _ = get_yes_no_labels(market)
    if not yes_label:
        return None

    p = market["outcome_map"].get(yes_label)
    if p is None:
        return None

    p = float(p)
    if p > 1.0:
        p = p / 100.0
    return p


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


# -------------------------------------------------
# DECISION ENGINE
# -------------------------------------------------
def decide_for_market(market: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    elapsed = market.get("elapsed")

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

    market_yes = get_market_yes_probability(market)
    if market_yes is None:
        return {
            "signal": "IGNORE THIS MARKET",
            "reason": "Polymarket live YES price unavailable",
            "locked": True,
        }

    edge = model_up - market_yes

    if model_up >= 0.60 and edge >= 0.08:
        return {
            "signal": "BUY YES",
            "reason": f"{model_reason} | Model UP={model_up:.0%}, Market YES={market_yes:.0%}",
            "locked": True,
        }

    if model_up <= 0.40 and (-edge) >= 0.08:
        return {
            "signal": "BUY NO",
            "reason": f"{model_reason} | Model DOWN={1-model_up:.0%}, Market NO={1-market_yes:.0%}",
            "locked": True,
        }

    return {
        "signal": "IGNORE THIS MARKET",
        "reason": f"No strong edge | Model UP={model_up:.0%}, Market YES={market_yes:.0%}",
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


# -------------------------------------------------
# UI
# -------------------------------------------------
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

    yes_label, no_label = get_yes_no_labels(market)
    market_yes = get_market_yes_probability(market)
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

    st.write(f"**Outcomes:** {yes_label} / {no_label}")
    st.write(f"**Signal:** ### {result['signal']}")
    st.caption(result["reason"])
    st.caption("LOCKED" if result["locked"] else "NOT LOCKED YET")

    if market.get("seconds_left") is not None:
        mins = market["seconds_left"] // 60
        secs = market["seconds_left"] % 60
        st.caption(f"Time left: {mins:02d}:{secs:02d}")

    if market_yes is not None:
        st.caption(f"Live YES price: {market_yes:.0%}")

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


st.set_page_config(page_title="Polymarket 15m Locked Signal Bot", layout="centered")

st.title("Polymarket 15-Min Locked Signal Bot")
st.write("One decision only per exact live Polymarket market")

refresh = st.slider("Refresh every (seconds)", 5, 60, REFRESH_DEFAULT)

col1, col2 = st.columns(2)

with col1:
    render_asset("BTC", "BTCUSDT")

with col2:
    render_asset("ETH", "ETHUSDT")

time.sleep(refresh)
st.rerun()
