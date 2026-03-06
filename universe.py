"""
売買代金上位N銘柄を動的に選定する。
J-Quants の直近20営業日の平均売買代金でランキング。
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from jquants import get_daily_quotes_date, get_master

CACHE_TTL_HOURS = 24  # 1日キャッシュ


def get_top_liquid_tickers(n: int = 4000) -> list[str]:
    """売買代金上位N銘柄のコードリスト（プライム+スタンダード+グロース全銘柄対象）"""
    import time

    cache_path = Path(f"data/universe_cache_{n}.json")

    # キャッシュ確認
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < CACHE_TTL_HOURS * 3600:
            data = json.loads(cache_path.read_text())
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

    # 投資信託・ETF除外（MktNm == 'その他'）
    master = get_master()
    exclude_codes = set(master[master['MktNm'].isin(['その他', 'TOKYO PRO MARKET'])]['Code'].astype(str).tolist())
    print(f"    除外: 投資信託/ETF {len(exclude_codes)}銘柄")

    # ランキング
    ranked = sorted(turnover_sum.items(), key=lambda x: x[1], reverse=True)
    top_codes = [code for code, _ in ranked if code not in exclude_codes][:n]

    # キャッシュ保存
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "updated": datetime.now().isoformat(),
        "tickers": top_codes,
        "turnover": dict(ranked[:n]),
    }, ensure_ascii=False))

    print(f"  ユニバース: 売買代金上位{len(top_codes)}銘柄を選定")
    return top_codes[:n]


if __name__ == "__main__":
    tickers = get_top_liquid_tickers(2000)
    print(f"\nTop 20: {tickers[:20]}")
