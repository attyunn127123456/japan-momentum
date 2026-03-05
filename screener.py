"""Daily momentum screener for TSE Prime stocks"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from stocks import TICKERS, BENCHMARK
from momentum import calculate_momentum_score, apply_filters

CACHE_DIR = Path("data/cache")
RESULTS_DIR = Path("results")
CACHE_MAX_AGE_HOURS = 6


def get_cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('^', '_')}.parquet"


def load_cached(ticker: str) -> pd.DataFrame | None:
    path = get_cache_path(ticker)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > CACHE_MAX_AGE_HOURS * 3600:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def fetch_ticker(ticker: str, period: str = "3y") -> pd.DataFrame | None:
    cached = load_cached(ticker)
    if cached is not None:
        return cached
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(get_cache_path(ticker))
        return df
    except Exception as e:
        print(f"  [WARN] {ticker}: {e}")
        return None


def get_market_cap(ticker: str) -> float:
    try:
        info = yf.Ticker(ticker).fast_info
        return getattr(info, "market_cap", 0) or 0
    except Exception:
        return 0


def score_ticker(ticker: str, nikkei_prices: pd.Series) -> dict | None:
    df = fetch_ticker(ticker)
    if df is None or len(df) < 60:
        return None

    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    prices = df["Close"].dropna()
    volumes = df["Volume"].dropna()

    market_cap = get_market_cap(ticker)
    passes, reason = apply_filters(ticker, prices, volumes, market_cap)
    if not passes:
        return None

    scores = calculate_momentum_score(prices, volumes, nikkei_prices)
    if not scores["valid"]:
        return None

    return {
        "ticker": ticker,
        "name": ticker,  # will enrich later if needed
        "score": scores["total"],
        "rs_score": round(scores["rs_score"], 2),
        "return_5_25d": round(scores["return_5_25d"], 2),
        "volume_acceleration": round(scores["volume_acceleration"], 2),
        "green_day_ratio": round(scores["green_day_ratio"], 2),
        "rs_acceleration": round(scores["rs_acceleration"], 2),
        "price": round(float(prices.iloc[-1]), 2),
        "market_cap": market_cap,
    }


def run_screener(top_n: int = 20) -> list[dict]:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching Nikkei 225...")
    nikkei_df = fetch_ticker(BENCHMARK)
    if nikkei_df is None:
        raise RuntimeError("Failed to fetch Nikkei 225")

    if isinstance(nikkei_df.columns, pd.MultiIndex):
        nikkei_df.columns = nikkei_df.columns.get_level_values(0)
    nikkei_prices = nikkei_df["Close"].dropna()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scoring {len(TICKERS)} stocks...")
    results = []
    with ThreadPoolExecutor(max_workers=8) as exe:
        futures = {exe.submit(score_ticker, t, nikkei_prices): t for t in TICKERS}
        done = 0
        for f in as_completed(futures):
            done += 1
            r = f.result()
            if r:
                results.append(r)
            if done % 20 == 0:
                print(f"  {done}/{len(TICKERS)} processed, {len(results)} valid...")

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:top_n]

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    out_path = RESULTS_DIR / f"{today}.json"
    with open(out_path, "w") as f:
        json.dump({"date": today, "results": results, "top": top}, f, ensure_ascii=False, indent=2)
    # Also save as latest.json
    with open(RESULTS_DIR / "latest.json", "w") as f:
        json.dump({"date": today, "results": results, "top": top}, f, ensure_ascii=False, indent=2)

    print(f"\n=== TOP {top_n} モメンタム銘柄 ({today}) ===")
    for i, r in enumerate(top, 1):
        print(f"{i:2}. {r['ticker']:10} スコア:{r['score']:5.1f} | RS:{r['rs_score']:+.1f} | "
              f"中期リターン:{r['return_5_25d']:+.1f}% | 出来高加速:{r['volume_acceleration']:.2f}")

    return top


if __name__ == "__main__":
    run_screener()
