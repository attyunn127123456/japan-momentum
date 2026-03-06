#!/usr/bin/env python3
"""
ダッシュボード用キャッシュを一括生成。
- ranking_cache.json / weekly_picks_cache.json: daily_signal_output.json から生成
- timeseries_cache.json: baseline.equity_curve から生成
- データフロー: daily_signal_output.py → generate_dashboard_cache.py → 全キャッシュ
"""
import sys, json
sys.path.insert(0, '.')
from pathlib import Path
from datetime import datetime
import pandas as pd


def main():
    print("ダッシュボードキャッシュ生成開始...", flush=True)

    # 1. ベストパラメータ取得（単一真実のソース）
    q = json.loads(Path("backtest/hypothesis_queue.json").read_text())
    baseline = q["baseline"]
    params = baseline.get("params", {})

    # 2. equity_curve（timeseries用）は baseline から
    equity_curve = baseline.get("equity_curve", [])
    total_return_pct = baseline.get("total_pct", 0)
    sharpe = baseline.get("sharpe", 0)
    max_dd_pct = baseline.get("max_dd_pct", 0)

    print(f"ベスト: total={total_return_pct}%, sharpe={sharpe}, max_dd={max_dd_pct}%", flush=True)
    print(f"params lb={params.get('lookback')}, top_n={params.get('top_n')}", flush=True)

    # 3. daily_signal_output.json の読み込み
    signal_path = Path("backtest/daily_signal_output.json")
    today = datetime.now().strftime('%Y-%m-%d')

    if not signal_path.exists():
        print("daily_signal未生成、ranking/picksキャッシュをスキップ", flush=True)
        sig = None
    else:
        sig = json.loads(signal_path.read_text())
        as_of = sig.get('as_of', today)
        if as_of == today:
            print(f"daily_signalを使用: as_of={today}", flush=True)
        else:
            print(f"古いsignalを使用: as_of={as_of}", flush=True)

    # 4. 銘柄名マッピング
    name_map = {}
    try:
        master = pd.read_parquet("data/fundamentals/equities_master.parquet")
        name_col = next((c for c in ["CoName", "CompanyName", "Name"] if c in master.columns), None)
        if name_col:
            name_map = dict(zip(master["Code"].astype(str), master[name_col].astype(str)))
    except Exception as e:
        print(f"銘柄名マッピング取得失敗 (無視): {e}", flush=True)

    # 5. ranking_cache.json / weekly_picks_cache.json — daily_signalから生成
    if sig is not None:
        top_codes = [s['code'] for s in sig.get('recommended', [])]
        top20 = sig.get('top20', [])
        as_of = sig.get('as_of', today)

        rankings = []
        for rank, s in enumerate(top20, 1):
            rankings.append({
                "rank": rank,
                "code": s['code'],
                "name": s['name'],
                "score": s['score'],
                "price": None,
                "ret5d_pct": None,
                "ret20d_pct": None,
                "is_top": s['code'] in top_codes,
            })

        ranking_cache = {
            "cached_at": datetime.now().strftime("%Y-%m-%d %H:%M JST"),
            "as_of": as_of,
            "params": {
                "lookback": params.get("lookback"),
                "top_n": params.get("top_n", 2),
                "rebalance": params.get("rebalance", "weekly"),
                "total_return_pct": total_return_pct,
            },
            "top_codes": top_codes,
            "rankings": rankings,
        }
        Path("backtest/ranking_cache.json").write_text(
            json.dumps(ranking_cache, ensure_ascii=False, indent=2, default=str)
        )
        print(f"ranking_cache.json 更新: {as_of}, top={top_codes}, rankings={len(rankings)}件", flush=True)

        # changes は daily_signal の changes をそのまま使う（前日比較済み）
        buy = sig.get('changes', {}).get('buy', [])
        sell = sig.get('changes', {}).get('sell', [])
        hold = sig.get('changes', {}).get('hold', [])
        recommended = sig.get('recommended', [])

        picks_cache = {
            "cached_at": datetime.now().strftime("%Y-%m-%d %H:%M JST"),
            "as_of": as_of,
            "params": {
                "lookback": params.get("lookback"),
                "top_n": params.get("top_n", 2),
                "rebalance": params.get("rebalance", "weekly"),
                "total_return_pct": total_return_pct,
            },
            "recommended": recommended,
            "changes": {"buy": buy, "sell": sell, "hold": hold},
            "ranking": rankings[:20],
        }
        Path("backtest/weekly_picks_cache.json").write_text(
            json.dumps(picks_cache, ensure_ascii=False, indent=2, default=str)
        )
        print("weekly_picks_cache.json 更新完了", flush=True)
    else:
        print("daily_signal未生成のためranking/picksキャッシュはスキップ", flush=True)

    # 6. timeseries_cache.json — equity_curveから生成（データがある時のみ更新）
    if not equity_curve:
        print("equity_curve未保存。timeseries_cacheはスキップ。evolution完了後に再実行してください。", flush=True)
        print("全キャッシュ更新完了！", flush=True)
        return

    # weekly: 100スタートの累積指数
    weekly = [{"date": e["date"], "value": round(e["value"] + 100, 2)} for e in equity_curve]

    # monthly: 月末値から月次リターン計算
    df_eq = pd.DataFrame(equity_curve).set_index("date")
    df_eq.index = pd.to_datetime(df_eq.index)
    monthly_end = df_eq["value"].resample("ME").last()
    monthly_start = monthly_end.shift(1).fillna(0)
    monthly_return = ((monthly_end - monthly_start) / (100 + monthly_start) * 100).round(2)
    monthly = [{"month": str(d)[:7], "return_pct": float(r)} for d, r in monthly_return.items()]

    # holdings: 各週の保有銘柄（最新50件、新しい順）
    holdings = []
    for e in reversed(equity_curve[-50:]):
        stocks = [{"code": c, "name": name_map.get(str(c), str(c))} for c in e.get("holdings", [])]
        holdings.append({"date": e["date"], "stocks": stocks})

    timeseries = {
        "params": params,
        "cached_at": datetime.now().strftime("%Y-%m-%d %H:%M JST"),
        "total_return_pct": total_return_pct,
        "sharpe": sharpe,
        "max_dd_pct": max_dd_pct,
        "weekly": weekly,
        "monthly": monthly,
        "holdings": holdings,
        "nikkei_weekly": [],
    }
    Path("backtest/timeseries_cache.json").write_text(
        json.dumps(timeseries, ensure_ascii=False, indent=2, default=str)
    )
    print(f"timeseries_cache.json 更新完了 (weekly:{len(weekly)}件, monthly:{len(monthly)}件, holdings:{len(holdings)}件)", flush=True)
    print("全キャッシュ更新完了！", flush=True)


if __name__ == "__main__":
    main()
