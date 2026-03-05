"""Daily momentum screener using J-Quants V2 API"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import os

from jquants import get_daily_quotes_code
from momentum import calculate_momentum_score, apply_filters
from universe import get_top_liquid_tickers

RESULTS_DIR = Path("results")
API_KEY = os.environ.get("JQUANTS_API_KEY", "cph3PdiF8zxH9GxClcFfShcJdSUzuNpV9ho_zMPm4a8")


def get_nikkei(start: str, end: str) -> pd.Series:
    all_data = []
    params = {"from": start, "to": end}
    while True:
        r = requests.get(
            "https://api.jquants.com/v2/indices/bars/daily",
            headers={"x-api-key": API_KEY},
            params={"from": start, "to": end, "code": "0028"}  # TOPIX
        )
        if not r.ok:
            # フォールバック: yfinance
            import yfinance as yf
            df = yf.download("^N225", start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df["Close"].dropna()
        d = r.json()
        all_data.extend(d.get("data", []))
        if not d.get("pagination_key"):
            break
        params["pagination_key"] = d["pagination_key"]

    df = pd.DataFrame(all_data)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()["C"].dropna()


def score_ticker(code: str, start: str, end: str, nikkei: pd.Series, sector: str = "") -> dict:
    try:
        df = get_daily_quotes_code(code, start, end)
    except Exception:
        return None

    if df is None or len(df) < 60:
        return None

    prices = df["AdjC"].dropna() if "AdjC" in df else df.get("C", pd.Series()).dropna()
    volumes = df["AdjVo"].dropna() if "AdjVo" in df else df.get("Vo", pd.Series()).dropna()
    if len(prices) < 60:
        return None

    n_aligned = nikkei.reindex(prices.index, method="ffill").dropna()
    passes, _ = apply_filters(code, prices, volumes, market_cap=0)
    if not passes:
        return None

    scores = calculate_momentum_score(prices, volumes, n_aligned)
    if not scores["valid"]:
        return None

    return {
        "ticker": f"{code}.T",
        "code": code,
        "sector": sector,
        "score": scores["total"],
        "rs_score": round(scores["rs_score"], 2),
        "return_5_25d": round(scores["return_5_25d"], 2),
        "volume_acceleration": round(scores["volume_acceleration"], 2),
        "green_day_ratio": round(scores["green_day_ratio"], 2),
        "rs_acceleration": round(scores["rs_acceleration"], 2),
        "price": round(float(prices.iloc[-1]), 2),
    }


def run_screener(top_n: int = 20, universe_size: int = 500) -> list[dict]:
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] ユニバース選定中（売買代金上位{universe_size}）...")
    codes = get_top_liquid_tickers(universe_size)

    # セクター情報取得
    from jquants import get_master
    master = get_master()
    sector_map = dict(zip(master["Code"].astype(str), master["S17Nm"]))

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 日経/TOPIX取得中...")
    nikkei = get_nikkei(start, end)
    print(f"  {len(nikkei)}日分")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] {len(codes)}銘柄スコア計算中...")
    results = []
    with ThreadPoolExecutor(max_workers=12) as exe:
        futures = {exe.submit(score_ticker, c, start, end, nikkei, sector_map.get(c, "")): c for c in codes}
        done = 0
        for f in as_completed(futures):
            done += 1
            r = f.result()
            if r:
                results.append(r)
            if done % 100 == 0:
                print(f"  {done}/{len(codes)} 処理済, {len(results)} 有効")

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:top_n]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    for fname in [f"{today}.json", "latest.json"]:
        with open(RESULTS_DIR / fname, "w") as f:
            json.dump({"date": today, "results": results, "top": top}, f, ensure_ascii=False, indent=2)

    print(f"\n=== TOP {top_n} モメンタム銘柄 ({today}) ===")
    for i, r in enumerate(top, 1):
        print(f"{i:2}. {r['code']:6} [{r['sector'][:8]:8}] スコア:{r['score']:5.1f} | "
              f"RS:{r['rs_score']:+5.1f} | 中期:{r['return_5_25d']:+5.1f}% | 出来高:{r['volume_acceleration']:.2f}")

    return top


if __name__ == "__main__":
    run_screener()
