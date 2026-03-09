"""
J-Quants V2 ファンダメンタル・需給データ取得 & キャッシュ。
data/fundamentals/ 以下にparquetで保存。
"""
import time
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from jquants import (
    get_fins_summary,
    get_fins_details,
    get_fins_dividend,
    get_earnings_calendar,
    get_investor_types,
    get_margin_interest,
    get_short_ratio,
    get_topix,
)

CACHE_DIR = Path("data/fundamentals")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# キャッシュ有効期間 (秒)
CACHE_TTL = 86400  # 24h


def _cache_valid(path: Path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL


# ── 財務サマリー（全銘柄） ──────────────────────────
def fetch_fins_summary(codes: list, start: str = "2022-01-01") -> pd.DataFrame:
    """全銘柄の財務サマリー取得。銘柄ごとにAPI呼び出し。"""
    out = CACHE_DIR / "fins_summary.parquet"
    if _cache_valid(out):
        df = pd.read_parquet(out)
        print(f"fins_summary: キャッシュ使用 ({len(df)}行)")
        return df

    print(f"fins_summary 取得中... ({len(codes)}銘柄)")
    all_data = []
    for i, code in enumerate(codes):
        try:
            df_code = get_fins_summary(code=code)
            if not df_code.empty:
                all_data.append(df_code)
        except Exception as e:
            if "403" in str(e) or "429" in str(e):
                print(f"  [WARN] {code}: {e}")
                time.sleep(1)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(codes)}", flush=True)
        time.sleep(0.05)  # rate limit: 500req/min → 0.12s safe

    if not all_data:
        print("fins_summary: データ取得できず")
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    df.to_parquet(out)
    print(f"fins_summary: {len(df)}行保存")
    return df


# ── 決算発表予定 ────────────────────────────────────
def fetch_earnings_calendar(date_from: str = None, date_to: str = None) -> pd.DataFrame:
    """決算発表予定日（イベント駆動戦略用）"""
    out = CACHE_DIR / "earnings_calendar.parquet"
    if _cache_valid(out):
        df = pd.read_parquet(out)
        print(f"earnings_calendar: キャッシュ使用 ({len(df)}行)")
        return df

    print("earnings_calendar 取得中...")
    try:
        df = get_earnings_calendar(date_from=date_from, date_to=date_to)
        if not df.empty:
            df.to_parquet(out)
            print(f"earnings_calendar: {len(df)}行保存")
        return df
    except Exception as e:
        print(f"earnings_calendar エラー: {e}")
        return pd.DataFrame()


# ── 配当情報 ────────────────────────────────────────
def fetch_dividend(codes: list) -> pd.DataFrame:
    """配当金情報。銘柄ごとに取得。"""
    out = CACHE_DIR / "dividend.parquet"
    if _cache_valid(out):
        df = pd.read_parquet(out)
        print(f"dividend: キャッシュ使用 ({len(df)}行)")
        return df

    print(f"dividend 取得中... ({len(codes)}銘柄)")
    all_data = []
    for i, code in enumerate(codes):
        try:
            df_code = get_fins_dividend(code=code)
            if not df_code.empty:
                all_data.append(df_code)
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(codes)}", flush=True)
        time.sleep(0.05)

    if not all_data:
        print("dividend: データ取得できず")
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    # Mixed-type columns → string for safe parquet serialization
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)
    df.to_parquet(out)
    print(f"dividend: {len(df)}行保存")
    return df


# ── 信用取引残高 ────────────────────────────────────
def fetch_margin_interest(start: str = "2022-01-01") -> pd.DataFrame:
    out = CACHE_DIR / "margin_interest.parquet"
    if _cache_valid(out):
        print("margin_interest: キャッシュ使用")
        return pd.read_parquet(out)

    print("margin_interest 取得中...")
    try:
        df = get_margin_interest(date_from=start)
        if not df.empty:
            df.to_parquet(out)
            print(f"margin_interest: {len(df)}行保存")
        return df
    except Exception as e:
        print(f"margin_interest エラー: {e}")
        return pd.DataFrame()


# ── 投資部門別売買 ──────────────────────────────────
def fetch_investor_types(start: str = "2022-01-01") -> pd.DataFrame:
    out = CACHE_DIR / "investor_types.parquet"
    if _cache_valid(out):
        print("investor_types: キャッシュ使用")
        return pd.read_parquet(out)

    print("investor_types 取得中...")
    try:
        df = get_investor_types(date_from=start)
        if not df.empty:
            df.to_parquet(out)
            print(f"investor_types: {len(df)}行保存")
        return df
    except Exception as e:
        print(f"investor_types エラー: {e}")
        return pd.DataFrame()


# ── 空売り比率 ──────────────────────────────────────
def fetch_short_ratio(start: str = "2022-01-01") -> pd.DataFrame:
    out = CACHE_DIR / "short_ratio.parquet"
    if _cache_valid(out):
        print("short_ratio: キャッシュ使用")
        return pd.read_parquet(out)

    print("short_ratio 取得中...")
    try:
        df = get_short_ratio(date_from=start)
        if not df.empty:
            df.to_parquet(out)
            print(f"short_ratio: {len(df)}行保存")
        return df
    except Exception as e:
        print(f"short_ratio エラー: {e}")
        return pd.DataFrame()


# ── TOPIX指数 ───────────────────────────────────────
def fetch_topix(start: str = "2022-01-01") -> pd.DataFrame:
    out = CACHE_DIR / "topix.parquet"
    if _cache_valid(out):
        print("topix: キャッシュ使用")
        return pd.read_parquet(out)

    print("topix 取得中...")
    try:
        df = get_topix(date_from=start)
        if not df.empty:
            df.to_parquet(out)
            print(f"topix: {len(df)}行保存")
        return df
    except Exception as e:
        print(f"topix エラー: {e}")
        return pd.DataFrame()


# ── メイン ──────────────────────────────────────────
if __name__ == "__main__":
    try:
        from universe import get_top_liquid_tickers
        codes = get_top_liquid_tickers(500)
    except Exception:
        # fallback: マスタから取得
        from jquants import get_master
        master = get_master()
        codes = master["Code"].tolist()[:500]

    print(f"対象: {len(codes)}銘柄")

    fetch_fins_summary(codes)
    fetch_earnings_calendar()
    fetch_dividend(codes)
    fetch_margin_interest()
    fetch_investor_types()
    fetch_short_ratio()
    fetch_topix()

    print("全データ取得完了")
    Path("data/fundamentals/fetch_done.json").write_text(
        json.dumps({"at": datetime.now().isoformat(), "codes": len(codes)})
    )
