"""
ベースラインバックテストを実行し、timeseries_cache.jsonを正しい形式で生成するスクリプト。
"""
import sys
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
os.chdir(str(BASE))

print(f"[{datetime.now().isoformat()}] ベースラインバックテスト開始...")

# hypothesis_queue.json のベースラインパラメータを使用
_q_path = BASE / "backtest/hypothesis_queue.json"
if _q_path.exists():
    _q = json.loads(_q_path.read_text())
    _baseline = _q.get("baseline", {})
    PARAMS = _baseline.get("params", {})
    if PARAMS:
        print(f"  ベストパラメータ使用 (total={_baseline.get('total_pct', '?')}%): {PARAMS}")
    else:
        PARAMS = {
            "lookback": 40,
            "top_n": 2,
            "rebalance": "weekly",
            "ret_w": 0.3,
            "rs_w": 0.3,
            "green_w": 0.2,
            "smooth_w": 0.2,
        }
        print("  フォールバックパラメータ使用")
else:
    PARAMS = {
        "lookback": 40,
        "top_n": 2,
        "rebalance": "weekly",
        "ret_w": 0.3,
        "rs_w": 0.3,
        "green_w": 0.2,
        "smooth_w": 0.2,
    }
    print("  hypothesis_queue.jsonなし、デフォルトパラメータ使用")

end = datetime.now().strftime("%Y-%m-%d")
start = "2023-01-01"

print(f"期間: {start} → {end}, top_n={PARAMS['top_n']}, rebalance={PARAMS['rebalance']}")

from backtest import run_backtest
result = run_backtest(
    start=start,
    end=end,
    top_n=PARAMS["top_n"],
    rebalance=PARAMS["rebalance"],
)

print(f"[{datetime.now().isoformat()}] バックテスト完了。timeseries_cache.jsonを更新中...")

# 現在の timeseries_cache.json を読み込む
cache_path = BASE / "backtest/timeseries_cache.json"
raw = json.loads(cache_path.read_text())

# weekly_holdings（スコア付き）を構築
# result["weekly_holdings"] は最後52週のみ
# all_trades は全週（スコアなし）
weekly_holdings_from_result = {w["date"]: w for w in result.get("weekly_holdings", [])}

# 名前マッピング: equities_master.parquet（全銘柄）から取得
name_map = {}
try:
    import pandas as pd
    master_path = BASE / "data/fundamentals/equities_master.parquet"
    if master_path.exists():
        master_df = pd.read_parquet(master_path)
        for _, row in master_df.iterrows():
            code = str(row["Code"]).strip()
            name = str(row["CoName"]).strip()
            if code and name:
                name_map[code] = name
        print(f"  equities_masterから銘柄名マップ: {len(name_map)}件")
    else:
        print("  equities_master.parquetなし、latest.jsonにフォールバック")
except Exception as e:
    print(f"  equities_master読み込み失敗: {e}")

# フォールバック: results/latest.json からも補完
try:
    latest = json.loads((BASE / "results/latest.json").read_text())
    results_list = latest.get("results", [])
    added = 0
    for r in results_list:
        code = (r.get("ticker", "") or r.get("code", "")).replace(".T", "")
        if code and r.get("name") and code not in name_map:
            name_map[code] = r["name"]
            added += 1
    if added:
        print(f"  latest.jsonから追加: {added}件")
except Exception as e:
    print(f"  latest.json読み込み失敗: {e}")

print(f"  銘柄名マップ合計: {len(name_map)}件")

# all_trades から holdings を構築
all_trades = raw.get("all_trades", [])
holdings_list = []
for trade in all_trades:
    date = trade["date"]
    tickers = trade.get("top_n", [])
    
    # スコアをweekly_holdingsから取得（あれば）
    wh = weekly_holdings_from_result.get(date)
    scores_map = {}
    if wh:
        scores_map = {t.replace(".T", ""): s for t, s in wh.get("scores", {}).items()}
    
    stocks = []
    for ticker in tickers:
        code = ticker.replace(".T", "")
        stocks.append({
            "code": code,
            "name": name_map.get(code, code),
            "score": scores_map.get(code),
        })
    
    holdings_list.append({
        "date": date,
        "stocks": stocks,
    })

# ポートフォリオ週次（100スタートに正規化）
initial_capital = raw.get("initial_capital", 1_000_000)
weekly = [
    {"date": t["date"], "value": round(t["portfolio_value"] / initial_capital * 100, 2)}
    for t in all_trades
]

# 日経週次（100スタートに正規化）
nikkei_start = raw.get("nikkei_start", 1)
equity_curve = raw.get("equity_curve", [])
nikkei_weekly = [
    {"date": e["date"], "value": round(e["nikkei"] / initial_capital * 100, 2)}
    for e in equity_curve
    if e.get("nikkei") is not None
]

# 月次リターン計算
monthly = []
try:
    import pandas as pd
    portfolio_values = {t["date"]: t["portfolio_value"] for t in all_trades}
    series = pd.Series(portfolio_values)
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    monthly_end = series.resample("ME").last().dropna()
    if len(monthly_end) >= 2:
        monthly_returns = monthly_end.pct_change().dropna() * 100
        monthly = [
            {"month": str(d)[:7], "return_pct": round(float(r), 2)}
            for d, r in zip(monthly_returns.index, monthly_returns.values)
        ]
    print(f"  月次リターン: {len(monthly)}件")
except Exception as e:
    print(f"  月次リターン計算失敗: {e}")

# 正しい形式でtimeseries_cache.jsonを上書き
new_cache = {
    "params": PARAMS,
    "cached_at": datetime.now().isoformat(),
    "weekly": weekly,
    "monthly": monthly,
    "nikkei_weekly": nikkei_weekly,
    "holdings": holdings_list,
    # 後続処理のために元データも保持
    "summary": raw.get("summary", {}),
    "all_trades": all_trades,
    "equity_curve": equity_curve,
    "nikkei_start": nikkei_start,
    "initial_capital": initial_capital,
}

cache_path.write_text(json.dumps(new_cache, ensure_ascii=False, indent=2))
print(f"[{datetime.now().isoformat()}] timeseries_cache.json 保存完了!")
print(f"  weekly: {len(weekly)}件")
print(f"  monthly: {len(monthly)}件")
print(f"  nikkei_weekly: {len(nikkei_weekly)}件")
print(f"  holdings: {len(holdings_list)}件")
print("  最初の保有銘柄:", holdings_list[0] if holdings_list else "なし")
print("DONE")
