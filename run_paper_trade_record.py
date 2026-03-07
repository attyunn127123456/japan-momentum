#!/usr/bin/env python3
"""
毎週のシグナル（TOP銘柄）をペーパートレードログに記録する。
cron例: 毎週月曜 09:00 JST に実行
  0 0 * * 1 cd /Users/panda/Projects/japan-momentum && python3 run_paper_trade_record.py
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).parent

def main():
    # ranking_cache.json からTOP銘柄を取得
    ranking_path = BASE / "backtest/ranking_cache.json"
    if not ranking_path.exists():
        print("ERROR: ranking_cache.json が見つかりません。generate_dashboard_cache.py を先に実行してください。", file=sys.stderr)
        sys.exit(1)

    ranking = json.loads(ranking_path.read_text())
    top_n = ranking.get("params", {}).get("top_n", 2)
    top_codes = ranking.get("top_codes", [])[:top_n]
    as_of = ranking.get("as_of", "")

    if not top_codes:
        print("ERROR: TOP銘柄が見つかりません", file=sys.stderr)
        sys.exit(1)

    # 週の月曜日を week キーにする
    try:
        dt = datetime.strptime(as_of, "%Y-%m-%d")
    except Exception:
        dt = datetime.today()
    monday = dt - timedelta(days=dt.weekday())
    week_key = monday.strftime("%Y-%m-%d")

    print(f"as_of: {as_of}, week: {week_key}, holdings: {top_codes}")

    # daily_signal_output.json からエントリー価格（price）を取得
    signal_path = BASE / "backtest/daily_signal_output.json"
    entry_prices = {}
    if signal_path.exists():
        sig = json.loads(signal_path.read_text())
        all_scores = sig.get("all_scores", sig.get("top20", sig.get("recommended", [])))
        price_map = {s["code"]: s.get("price") for s in all_scores if s.get("price") is not None}
        for code in top_codes:
            if code in price_map:
                entry_prices[code] = price_map[code]

    # 既存ログを読み込み
    log_path = BASE / "backtest/paper_trade_log.json"
    if log_path.exists():
        log = json.loads(log_path.read_text())
    else:
        log = {"entries": []}

    # 同じ週がすでに存在する場合はスキップ
    existing_weeks = [e["week"] for e in log.get("entries", [])]
    if week_key in existing_weeks:
        print(f"SKIP: week {week_key} はすでに記録済みです")
        sys.exit(0)

    # 新エントリーを追加
    new_entry = {
        "week": week_key,
        "holdings": top_codes,
        "entry_prices": entry_prices,
        "exit_prices": {},
        "status": "open",
        "return_pct": None,
    }
    log.setdefault("entries", []).append(new_entry)
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2))

    print(f"OK: 記録しました → {new_entry}")

if __name__ == "__main__":
    main()
