"""J-Quants API V2 client — full endpoint support"""
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

API_KEY = os.environ.get("JQUANTS_API_KEY", "cph3PdiF8zxH9GxClcFfShcJdSUzuNpV9ho_zMPm4a8")
BASE_URL = "https://api.jquants.com/v2"
CACHE_DIR = Path("data/cache_v2")


def _get(endpoint: str, params: dict = None, timeout: int = 30) -> dict:
    headers = {"x-api-key": API_KEY}
    r = requests.get(f"{BASE_URL}{endpoint}", headers=headers, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _get_all(endpoint: str, params: dict = None) -> list:
    """Paginated fetch — returns all rows."""
    all_data = []
    p = dict(params or {})
    while True:
        d = _get(endpoint, p)
        all_data.extend(d.get("data", []))
        pk = d.get("pagination_key")
        if not pk:
            break
        p["pagination_key"] = pk
    return all_data


# ── 銘柄マスタ ──────────────────────────────────────
def get_master() -> pd.DataFrame:
    """全上場銘柄マスタ（市場・セクター情報付き）"""
    d = _get("/equities/master")
    return pd.DataFrame(d["data"])


# ── 株価 ────────────────────────────────────────────
def get_daily_quotes_date(date: str) -> pd.DataFrame:
    """指定日の全銘柄株価（date: YYYY-MM-DD）"""
    all_data = _get_all("/equities/bars/daily", {"date": date})
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

    all_data = _get_all("/equities/bars/daily", {"code": code, "from": start, "to": end})
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
    from concurrent.futures import ThreadPoolExecutor

    dates = pd.date_range(start, end, freq="B")
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    print(f"J-Quants: {len(date_strs)}営業日分のデータを取得中...")
    all_rows = []

    def fetch_date(dt):
        try:
            return get_daily_quotes_date(dt)
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

    ticker_codes = {t.replace(".T", "") for t in tickers}
    combined = combined[combined["Code"].isin(ticker_codes)]

    result = {}
    for code, grp in combined.groupby("Code"):
        df = grp.set_index("Date").sort_index()
        result[code] = df

    print(f"J-Quants: {len(result)}銘柄のデータ取得完了")
    return result


# ── 財務情報（V2: /fins/summary） ───────────────────
def get_fins_summary(code: str = None, date_from: str = None, date_to: str = None) -> pd.DataFrame:
    """
    財務サマリー（EPS・売上・純利益・BPS・ROE等）。
    code指定で個別銘柄、省略で全銘柄。date_from/date_toで期間指定。
    """
    params = {}
    if code:
        params["code"] = code
    if date_from:
        params["from"] = date_from
    if date_to:
        params["to"] = date_to
    data = _get_all("/fins/summary", params)
    df = pd.DataFrame(data)
    if not df.empty and "DiscDate" in df.columns:
        df["DiscDate"] = pd.to_datetime(df["DiscDate"], errors="coerce")
    return df


# ── 財務諸表詳細（BS/PL/CF） ────────────────────────
def get_fins_details(code: str) -> pd.DataFrame:
    """財務諸表の詳細（BS/PL/CF）。個別銘柄指定。"""
    data = _get_all("/fins/details", {"code": code})
    return pd.DataFrame(data)


# ── 決算発表予定（V2: /equities/earnings-calendar） ──
def get_earnings_calendar(date_from: str = None, date_to: str = None) -> pd.DataFrame:
    """決算発表予定日一覧。"""
    params = {}
    if date_from:
        params["from"] = date_from
    if date_to:
        params["to"] = date_to
    data = _get_all("/equities/earnings-calendar", params)
    df = pd.DataFrame(data)
    if not df.empty and "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


# ── 配当金情報 ──────────────────────────────────────
def get_fins_dividend(code: str = None) -> pd.DataFrame:
    """配当金情報。code省略で全銘柄。"""
    params = {}
    if code:
        params["code"] = code
    data = _get_all("/fins/dividend", params)
    return pd.DataFrame(data)


# ── TOPIX指数 ───────────────────────────────────────
def get_topix(date_from: str = None, date_to: str = None) -> pd.DataFrame:
    """TOPIX四本値。"""
    params = {}
    if date_from:
        params["from"] = date_from
    if date_to:
        params["to"] = date_to
    data = _get_all("/indices/bars/daily/topix", params)
    df = pd.DataFrame(data)
    if not df.empty and "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


# ── 投資部門別売買動向 ──────────────────────────────
def get_investor_types(date_from: str = None, section: str = "TSEPrime") -> pd.DataFrame:
    """投資部門別売買動向。"""
    params = {"section": section}
    if date_from:
        params["from"] = date_from
    data = _get_all("/equities/investor-types", params)
    df = pd.DataFrame(data)
    if not df.empty and "PublishedDate" in df.columns:
        df["PublishedDate"] = pd.to_datetime(df["PublishedDate"], errors="coerce")
    return df


# ── 市場データ補助 ──────────────────────────────────
def get_margin_interest(date_from: str = None) -> pd.DataFrame:
    """信用取引週末残高。"""
    params = {}
    if date_from:
        params["from"] = date_from
    data = _get_all("/markets/margin-interest", params)
    return pd.DataFrame(data)


def get_short_ratio(date_from: str = None) -> pd.DataFrame:
    """業種別空売り比率。"""
    params = {}
    if date_from:
        params["from"] = date_from
    data = _get_all("/markets/short-ratio", params)
    return pd.DataFrame(data)


def get_market_calendar() -> pd.DataFrame:
    """取引カレンダー。"""
    data = _get_all("/markets/calendar")
    return pd.DataFrame(data)
