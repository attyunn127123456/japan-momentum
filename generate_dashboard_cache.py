#!/usr/bin/env python3
"""
ダッシュボード用キャッシュを一括生成。
evolution_engine/run_hypothesis完了後に呼ぶ。
"""
import sys, json
sys.path.insert(0, '.')
from pathlib import Path
from datetime import datetime
from collections import Counter
import pandas as pd


def main():
    print("ダッシュボードキャッシュ生成開始...", flush=True)

    # 1. ベストパラメータ取得
    q = json.loads(Path("backtest/hypothesis_queue.json").read_text())
    baseline = q["baseline"]
    params = baseline.get("params", {})

    # 2. equity_curve 取得
    # まず oos_result から取得（最も信頼できるソース）
    oos = baseline.get("oos_result", {})
    equity_curve = oos.get("equity_curve", [])

    if not equity_curve:
        # フォールバック: evolution_log.json の all エントリから equity_curve 付きを探す
        log_path = Path("backtest/evolution_log.json")
        if log_path.exists():
            log = json.loads(log_path.read_text())
            entries = []
            if isinstance(log, list):
                entries = [e for e in log if e.get("equity_curve")]
            elif isinstance(log, dict):
                entries = [e for e in log.get("all", []) if e.get("equity_curve")]
            if entries:
                best = max(entries, key=lambda e: e.get("total_return_pct", 0))
                equity_curve = best["equity_curve"]
                print(f"evolution_log からequity_curve取得: total={best.get('total_return_pct')}%", flush=True)

    if not equity_curve:
        print("equity_curveなし、スキップ", flush=True)
        return

    total_return_pct = oos.get("total_return_pct") or baseline.get("total_pct")
    sharpe = oos.get("sharpe") or baseline.get("sharpe")
    max_dd_pct = oos.get("max_dd_pct") or baseline.get("max_dd_pct")

    print(f"ベスト: total={total_return_pct}%, sharpe={sharpe}", flush=True)

    # 3. 銘柄名マッピング
    name_map = {}
    try:
        master = pd.read_parquet("data/fundamentals/equities_master.parquet")
        name_col = next((c for c in ["CoName", "CompanyName", "Name"] if c in master.columns), None)
        if name_col:
            name_map = dict(zip(master["Code"].astype(str), master[name_col].astype(str)))
    except Exception as e:
        print(f"銘柄名マッピング取得失敗 (無視): {e}", flush=True)

    # 4. timeseries_cache.json 生成
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
    print(f"timeseries_cache.json 生成完了 (weekly:{len(weekly)}件, monthly:{len(monthly)}件, holdings:{len(holdings)}件)", flush=True)

    # 5. ranking_cache.json 生成
    last_entry = equity_curve[-1]
    top_codes = [str(c) for c in last_entry.get("holdings", [])]

    # 直近20週のholdings出現頻度でスコア付け
    code_counts = Counter()
    for e in equity_curve[-20:]:
        for c in e.get("holdings", []):
            code_counts[str(c)] += 1

    rankings = []
    for rank, (code, count) in enumerate(code_counts.most_common(50), 1):
        rankings.append({
            "rank": rank,
            "code": code,
            "name": name_map.get(code, code),
            "score": round(count / 20, 3),
            "price": None,
            "ret5d_pct": None,
            "ret20d_pct": None,
            "is_top": code in top_codes,
        })

    ranking_cache = {
        "cached_at": datetime.now().strftime("%Y-%m-%d %H:%M JST"),
        "as_of": last_entry["date"],
        "params": {
            "lookback": params.get("lookback"),
            "top_n": params.get("top_n", 2),
            "total_return_pct": total_return_pct,
        },
        "top_codes": top_codes,
        "rankings": rankings,
    }
    Path("backtest/ranking_cache.json").write_text(
        json.dumps(ranking_cache, ensure_ascii=False, indent=2, default=str)
    )
    print(f"ranking_cache.json 生成完了 (top:{top_codes}, rankings:{len(rankings)}件)", flush=True)

    # 6. weekly_picks_cache.json 生成
    prev_holdings = [str(c) for c in equity_curve[-2].get("holdings", [])] if len(equity_curve) >= 2 else []
    buy = [c for c in top_codes if c not in prev_holdings]
    sell = [c for c in prev_holdings if c not in top_codes]
    hold = [c for c in top_codes if c in prev_holdings]

    def stock_info(code):
        return {
            "code": code,
            "name": name_map.get(code, code),
            "price": None,
            "score": next((r["score"] for r in rankings if r["code"] == code), 0),
        }

    picks_cache = {
        "cached_at": datetime.now().strftime("%Y-%m-%d %H:%M JST"),
        "as_of": last_entry["date"],
        "params": {
            "lookback": params.get("lookback"),
            "top_n": params.get("top_n", 2),
            "rebalance": params.get("rebalance", "weekly"),
            "total_return_pct": total_return_pct,
        },
        "recommended": [stock_info(c) for c in top_codes],
        "changes": {
            "buy": [stock_info(c) for c in buy],
            "sell": [stock_info(c) for c in sell],
            "hold": [stock_info(c) for c in hold],
        },
        "ranking": rankings[:20],
    }
    Path("backtest/weekly_picks_cache.json").write_text(
        json.dumps(picks_cache, ensure_ascii=False, indent=2, default=str)
    )
    print("weekly_picks_cache.json 生成完了", flush=True)
    print("全キャッシュ生成完了！", flush=True)


if __name__ == "__main__":
    main()
