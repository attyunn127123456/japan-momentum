"""
OHLCVデータをローカルにキャッシュする。
初回: 全銘柄の全期間を取得
更新: 最終取得日以降の差分だけ取得
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from jquants import get_daily_quotes_code

CACHE_DIR = Path("data/ohlcv_cache")
META_PATH = Path("data/ohlcv_meta.json")
API_KEY = os.environ.get("JQUANTS_API_KEY", "cph3PdiF8zxH9GxClcFfShcJdSUzuNpV9ho_zMPm4a8")


def get_cache_path(code: str) -> Path:
    return CACHE_DIR / f"{code}.parquet"


def load_cache(code: str) -> pd.DataFrame:
    p = get_cache_path(code)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def save_cache(code: str, df: pd.DataFrame):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(get_cache_path(code))


def get_last_date(code: str) -> str:
    df = load_cache(code)
    if df.empty:
        return None
    return df.index.max().strftime("%Y-%m-%d")


def get_first_date(code: str) -> str:
    df = load_cache(code)
    if df.empty:
        return None
    return df.index.min().strftime("%Y-%m-%d")


def fetch_and_cache(code: str, start: str, end: str) -> bool:
    """1銘柄のデータを取得してキャッシュ。既存データがあれば差分のみ取得。"""
    last = get_last_date(code)
    if last:
        # 差分のみ（最終日の翌日から）
        fetch_start = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        if fetch_start > end:
            return True  # 既に最新
    else:
        fetch_start = start

    try:
        new_df = get_daily_quotes_code(code, fetch_start, end)
        if new_df is None or new_df.empty:
            return True
        # 既存データとマージ
        existing = load_cache(code)
        if not existing.empty:
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        else:
            combined = new_df
        save_cache(code, combined)
        return True
    except Exception as e:
        print(f"  [WARN] {code}: {e}")
        return False


def backfill_one(code: str, backfill_to: str, retries: int = 3) -> bool:
    """1銘柄のデータを backfill_to 以前まで遡って取得してキャッシュに追記。"""
    import time as _time
    first = get_first_date(code)
    if first is None:
        # キャッシュがない場合は通常取得
        return fetch_and_cache(code, backfill_to, datetime.now().strftime("%Y-%m-%d"))

    if first <= backfill_to:
        return True  # 既に充分古いデータがある

    # backfill_to から既存キャッシュ開始日の前日まで取得
    fetch_end = (datetime.strptime(first, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    if fetch_end < backfill_to:
        return True

    for attempt in range(retries):
        try:
            new_df = get_daily_quotes_code(code, backfill_to, fetch_end)
            if new_df is None or new_df.empty:
                return True
            existing = load_cache(code)
            if not existing.empty:
                combined = pd.concat([new_df, existing])
                combined = combined[~combined.index.duplicated(keep='last')].sort_index()
            else:
                combined = new_df
            save_cache(code, combined)
            return True
        except Exception as e:
            err_str = str(e)
            if "429" in err_str and attempt < retries - 1:
                wait = 2 ** attempt + 1  # 2, 3秒
                _time.sleep(wait)
                continue
            print(f"  [WARN backfill] {code}: {e}")
            return False
    return False


def backfill_cache(codes: list, backfill_to: str = "2020-01-01", workers: int = 4):
    """既存キャッシュを backfill_to まで遡って取得する。
    レート制限対策で workers=4 がデフォルト。
    """
    print(f"OHLCVバックフィル: {len(codes)}銘柄 → {backfill_to}まで遡る")
    done = errors = 0
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(backfill_one, c, backfill_to): c for c in codes}
        for f in as_completed(futures):
            done += 1
            if not f.result():
                errors += 1
            if done % 50 == 0:
                print(f"  バックフィル {done}/{len(codes)} 完了 (エラー:{errors})")
    print(f"バックフィル完了: {done}銘柄, エラー:{errors}")


def update_cache(codes: list, start: str = "2022-01-01", end: str = None, workers: int = 16):
    """全銘柄のキャッシュを更新"""
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    print(f"OHLCVキャッシュ更新: {len(codes)}銘柄 ({start} → {end})")
    done = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(fetch_and_cache, c, start, end): c for c in codes}
        for f in as_completed(futures):
            done += 1
            if not f.result():
                errors += 1
            if done % 50 == 0:
                print(f"  {done}/{len(codes)} 完了 (エラー:{errors})")
    print(f"完了: {done}銘柄, エラー:{errors}")

    # メタ情報更新
    META_PATH.write_text(json.dumps({
        "updated": datetime.now().isoformat(),
        "codes": codes,
        "start": start,
        "end": end,
        "count": len(codes)
    }, ensure_ascii=False, indent=2))


def read_ohlcv(code: str, start: str = None, end: str = None) -> pd.DataFrame:
    """キャッシュからOHLCVを読む（API不要）"""
    df = load_cache(code)
    if df.empty:
        return df
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]
    return df


if __name__ == "__main__":
    import argparse
    from universe import get_top_liquid_tickers

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--n", type=int, default=500, help="対象銘柄数")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    codes = get_top_liquid_tickers(args.n)
    print(f"対象: {len(codes)}銘柄")
    update_cache(codes, start=args.start, workers=args.workers)
    print(f"\nキャッシュ場所: {CACHE_DIR}")
    total_mb = sum(p.stat().st_size for p in CACHE_DIR.glob("*.parquet")) / 1e6
    print(f"合計サイズ: {total_mb:.1f} MB")
