"""J-Quants API V2 client"""
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

API_KEY = os.environ.get("JQUANTS_API_KEY", "cph3PdiF8zxH9GxClcFfShcJdSUzuNpV9ho_zMPm4a8")
BASE_URL = "https://api.jquants.com/v2"
CACHE_DIR = Path("data/cache_v2")


def _get(endpoint: str, params: dict = None) -> dict:
    headers = {"x-api-key": API_KEY}
    r = requests.get(f"{BASE_URL}{endpoint}", headers=headers, params=params or {})
    r.raise_for_status()
    return r.json()


def get_master() -> pd.DataFrame:
    """全上場銘柄マスタ（市場・セクター情報付き）"""
    d = _get("/equities/master")
    return pd.DataFrame(d["data"])


def get_daily_quotes_date(date: str) -> pd.DataFrame:
    """指定日の全銘柄株価（date: YYYY-MM-DD）"""
    all_data = []
    params = {"date": date}
    while True:
        d = _get("/equities/bars/daily", params)
        all_data.extend(d.get("data", []))
        pk = d.get("pagination_key")
        if not pk:
            break
        params = {"date": date, "pagination_key": pk}
    df = pd.DataFrame(all_data)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def get_daily_quotes_code(code: str, start: str, end: str) -> pd.DataFrame:
    """個別銘柄の期間株価（code: 4桁, start/end: YYYY-MM-DD）"""
    cache_path = CACHE_DIR / f"{code}_{start}_{end}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < 6 * 3600:
        return pd.read_parquet(cache_path)

    all_data = []
    params = {"code": code, "from": start, "to": end}
    while True:
        d = _get("/equities/bars/daily", params)
        all_data.extend(d.get("data", []))
        pk = d.get("pagination_key")
        if not pk:
            break
        params = {**params, "pagination_key": pk}

    df = pd.DataFrame(all_data)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        df.to_parquet(cache_path)
    return df


def fetch_all_history(tickers: list, start: str, end: str, max_workers: int = 8) -> dict:
    """
    全銘柄の株価を日付一括取得でまとめて取得する（高速）。
    Returns: dict[code -> DataFrame(index=Date, cols=AdjC,AdjVo,...)]
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 日付リストを生成
    dates = pd.date_range(start, end, freq="B")  # Business days
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    print(f"J-Quants: {len(date_strs)}営業日分のデータを取得中...")
    all_rows = []

    def fetch_date(dt):
        try:
            df = get_daily_quotes_date(dt)
            return df
        except Exception as e:
            print(f"  [WARN] {dt}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(fetch_date, dt): dt for dt in date_strs}
        done = 0
        for f in futures:
            r = f.result()
            if r is not None and not r.empty:
                all_rows.append(r)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(date_strs)} 日付完了")

    if not all_rows:
        return {}

    combined = pd.concat(all_rows, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])

    # tickersのコード形式に合わせる（5803 or 5803.T）
    ticker_codes = {t.replace(".T", "") for t in tickers}
    combined = combined[combined["Code"].isin(ticker_codes)]

    result = {}
    for code, grp in combined.groupby("Code"):
        df = grp.set_index("Date").sort_index()
        result[code] = df

    print(f"J-Quants: {len(result)}銘柄のデータ取得完了")
    return result
