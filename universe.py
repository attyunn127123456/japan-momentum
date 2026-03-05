"""
売買代金上位N銘柄を動的に選定する。
J-Quants の直近20営業日の平均売買代金でランキング。
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from jquants import get_daily_quotes_date, get_master

CACHE_PATH = Path("data/universe_cache.json")
CACHE_TTL_HOURS = 24  # 1日キャッシュ


def get_top_liquid_tickers(n: int = 500) -> list[str]:
    """売買代金上位N銘柄のコードリスト（4桁）を返す"""
    import time

    # キャッシュ確認
    if CACHE_PATH.exists():
        age = time.time() - CACHE_PATH.stat().st_mtime
        if age < CACHE_TTL_HOURS * 3600:
            data = json.loads(CACHE_PATH.read_text())
            print(f"  ユニバース: キャッシュから{len(data['tickers'])}銘柄読み込み")
            return data["tickers"][:n]

    print("  ユニバース: 直近20日の売買代金でランキング中...")

    # 直近20営業日の日付を生成
    end = datetime.now()
    dates = []
    d = end
    while len(dates) < 20:
        d -= timedelta(days=1)
        if d.weekday() < 5:  # 平日のみ
            dates.append(d.strftime("%Y-%m-%d"))

    # 各日の全銘柄売買代金を取得して合算
    turnover_sum: dict[str, float] = {}
    fetched = 0
    for dt in dates:
        try:
            df = get_daily_quotes_date(dt)
            if df.empty:
                continue
            for _, row in df.iterrows():
                code = str(row["Code"])
                va = row.get("Va", 0) or 0  # 売買代金
                turnover_sum[code] = turnover_sum.get(code, 0) + float(va)
            fetched += 1
            if fetched % 5 == 0:
                print(f"    {fetched}/20日 完了")
        except Exception as e:
            print(f"    [WARN] {dt}: {e}")

    # ランキング
    ranked = sorted(turnover_sum.items(), key=lambda x: x[1], reverse=True)
    top_codes = [code for code, _ in ranked[:n]]

    # キャッシュ保存
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps({
        "updated": datetime.now().isoformat(),
        "tickers": top_codes,
        "turnover": dict(ranked[:n]),
    }, ensure_ascii=False))

    print(f"  ユニバース: 売買代金上位{len(top_codes)}銘柄を選定")
    return top_codes[:n]


if __name__ == "__main__":
    tickers = get_top_liquid_tickers(500)
    print(f"\nTop 20: {tickers[:20]}")
