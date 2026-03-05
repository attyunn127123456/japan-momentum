"""Core momentum scoring engine - pure math, no LLM"""
import numpy as np
import pandas as pd


def calculate_rs_score(stock_prices: pd.Series, nikkei_prices: pd.Series) -> float:
    """
    Relative Strength vs Nikkei 225.
    Weighted multi-period return difference.
    """
    periods = [(63, 0.40), (126, 0.30), (252, 0.20), (21, 0.10)]
    score = 0.0
    for days, weight in periods:
        if len(stock_prices) >= days + 1 and len(nikkei_prices) >= days + 1:
            s_ret = (stock_prices.iloc[-1] / stock_prices.iloc[-days] - 1) * 100
            n_ret = (nikkei_prices.iloc[-1] / nikkei_prices.iloc[-days] - 1) * 100
            score += (s_ret - n_ret) * weight
    return score


def calculate_momentum_score(
    prices: pd.Series,
    volumes: pd.Series,
    nikkei_prices: pd.Series,
    as_of: int = 0,  # 0 = today, -N = N days ago (for backtesting)
) -> dict:
    """
    Calculate full momentum score for a stock.
    
    Returns dict with sub-scores and total.
    Uses data up to index `as_of` from end (0=latest, -21=21 days ago etc.)
    """
    # Slice to as_of
    if as_of < 0:
        p = prices.iloc[:as_of]
        v = volumes.iloc[:as_of]
        n = nikkei_prices.iloc[:as_of]
    else:
        p = prices
        v = volumes
        n = nikkei_prices

    result = {
        "return_5_25d": 0.0,
        "volume_acceleration": 0.0,
        "green_day_ratio": 0.0,
        "rs_acceleration": 0.0,
        "rs_score": 0.0,
        "total": 0.0,
        "valid": False,
    }

    if len(p) < 30:
        return result

    # --- 1. return_5_25d: cumulative return from day -25 to day -3 ---
    # Excludes last 3 days to filter single-day pumps
    if len(p) >= 26:
        ret_5_25d = (p.iloc[-3] / p.iloc[-26] - 1) * 100
        result["return_5_25d"] = ret_5_25d
    
    # Normalized: cap at ±40%
    ret_norm = np.clip(result["return_5_25d"] / 40.0, -1.0, 1.0)

    # --- 2. volume_acceleration: weekly volume trend ---
    # Compare avg volume of last 4 weeks (W4=most recent)
    vol_acc = 0.0
    if len(v) >= 20:
        w1 = v.iloc[-20:-15].mean()
        w2 = v.iloc[-15:-10].mean()
        w3 = v.iloc[-10:-5].mean()
        w4 = v.iloc[-5:].mean()
        weeks = [w1, w2, w3, w4]
        # Count how many consecutive increases
        increases = sum(1 for i in range(1, 4) if weeks[i] > weeks[i-1])
        vol_acc = increases / 3.0  # 0, 0.33, 0.67, or 1.0
        result["volume_acceleration"] = vol_acc

    # --- 3. green_day_ratio: up-days in last 25 trading days ---
    if len(p) >= 26:
        daily_ret = p.iloc[-25:].pct_change().dropna()
        green_ratio = (daily_ret > 0).sum() / len(daily_ret)
        result["green_day_ratio"] = float(green_ratio)

    # --- 4. RS score and RS acceleration ---
    rs_now = calculate_rs_score(p, n)
    result["rs_score"] = rs_now

    # RS acceleration: compare RS now vs RS 21 days ago
    if len(p) >= 52 and len(n) >= 52:
        rs_21d_ago = calculate_rs_score(p.iloc[:-21], n.iloc[:-21])
        rs_accel = rs_now - rs_21d_ago
        # Normalize: delta of +-10% relative to nikkei = full score
        rs_accel_norm = np.clip(rs_accel / 10.0, -1.0, 1.0)
        result["rs_acceleration"] = float(rs_accel_norm)

    # --- Total score (0-100 scale) ---
    total = (
        ret_norm * 0.40
        + result["volume_acceleration"] * 0.30
        + result["green_day_ratio"] * 0.20
        + result["rs_acceleration"] * 0.10
    )
    # Scale to 0-100
    result["total"] = round((total + 1) / 2 * 100, 2)
    result["valid"] = True

    return result


def apply_filters(ticker: str, prices: pd.Series, volumes: pd.Series, market_cap: float) -> tuple:
    """
    Returns (passes, reason_if_fails)
    """
    if len(prices) < 210:
        return False, "insufficient_data"

    # Market cap filter (20B JPY)
    if market_cap and market_cap < 20_000_000_000:
        return False, "small_cap"

    # Single-day pump filter: >10% gain in last 3 days
    if len(prices) >= 4:
        recent_ret = (prices.iloc[-1] / prices.iloc[-4] - 1) * 100
        if recent_ret > 10:
            return False, "recent_pump"

    # Trend filter: 50d MA > 200d MA
    if len(prices) >= 200:
        ma50 = prices.iloc[-50:].mean()
        ma200 = prices.iloc[-200:].mean()
        if ma50 < ma200:
            return False, "downtrend"

    return True, ""
