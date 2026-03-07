"""
レジーム適応型戦略の検証スクリプト。
IS期間（2016-2020）と OOS期間（2021-2022）で
固定パラメータ vs レジーム切り替えを比較。
"""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_cache import read_ohlcv
from universe import get_top_liquid_tickers
from backtest import get_rebalance_dates, get_nikkei_history, detect_regime
from optimize import precompute, eval_params


# ─────────────────────────────────────────────
# パラメータ定義
# ─────────────────────────────────────────────

best_params = {
    'lookback': 100, 'top_n': 2, 'rebalance': 'weekly',
    'trailing_stop': -0.0271, 'exit_threshold': 0.160, 'exit_lookback': 24,
    'exit_rsi_w': 0.432, 'exit_vol_w': 0.230, 'exit_dd_w': 0.150,
    'ret_w': 0.100, 'upside_capture_w': 0.098, 'green_w': 0.095,
    'win_streak_w': 0.089, 'rs_w': 0.0,
}

defensive_params = {
    **best_params,
    'top_n': 5,
    'lookback': 40,
    'trailing_stop': -0.015,
    'exit_rsi_w': 0.5,
}

regime_params = {
    'bull': best_params,
    'bear': None,      # None = 全キャッシュ
    'choppy': defensive_params,
}

PERIODS = {
    "IS":  ("2016-01-01", "2020-12-31"),
    "OOS": ("2021-01-01", "2022-12-31"),
}

N_CODES = 500  # テスト用に銘柄数を絞る


def load_data(start, end, codes, warmup_days=200):
    warmup = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    prices_dict = {}
    for c in codes:
        df = read_ohlcv(c, warmup, end)
        if df is not None and not df.empty and "AdjC" in df.columns:
            prices_dict[c] = df
    nikkei = get_nikkei_history(warmup, end)
    return prices_dict, nikkei, warmup


def run_period(label, start, end, codes):
    print(f"\n{'='*60}")
    print(f"期間: {label} ({start} → {end})")
    print(f"{'='*60}")

    prices_dict, nikkei, warmup = load_data(start, end, codes)
    print(f"  銘柄数: {len(prices_dict)}")

    # 必要な lookback をすべて事前計算
    lb_needed = list({best_params['lookback'], defensive_params['lookback']})
    factor_dfs = precompute(prices_dict, nikkei, lb_needed)

    all_prices = pd.DataFrame({c: prices_dict[c]["AdjC"] for c in prices_dict}).astype(float)
    return_df = all_prices.pct_change()

    rebal_dates = get_rebalance_dates(warmup, end, "weekly")

    # 1) 固定パラメータ
    print("\n[固定パラメータ（best_params）]")
    t0 = time.time()
    r_fixed = eval_params(best_params, factor_dfs, prices_dict, rebal_dates, nikkei, start, return_df)
    print(f"  実行時間: {time.time()-t0:.1f}秒")
    if r_fixed:
        print(f"  total={r_fixed['total_return_pct']:+.2f}%  sharpe={r_fixed['sharpe']:.3f}  "
              f"max_dd={r_fixed['max_dd_pct']:.2f}%  alpha={r_fixed['alpha_pct']:+.2f}%")
    else:
        print("  結果なし（データ不足）")

    # 2) レジーム適応型
    print("\n[レジーム適応型]")
    t0 = time.time()
    r_regime = eval_params(best_params, factor_dfs, prices_dict, rebal_dates, nikkei, start,
                           return_df, regime_params=regime_params)
    print(f"  実行時間: {time.time()-t0:.1f}秒")
    if r_regime:
        print(f"  total={r_regime['total_return_pct']:+.2f}%  sharpe={r_regime['sharpe']:.3f}  "
              f"max_dd={r_regime['max_dd_pct']:.2f}%  alpha={r_regime['alpha_pct']:+.2f}%")
        # レジーム分布を表示
        regime_counts = {}
        for e in r_regime.get("equity_curve", []):
            reg = e.get("regime", "?")
            regime_counts[reg] = regime_counts.get(reg, 0) + 1
        print(f"  レジーム分布: {regime_counts}")
    else:
        print("  結果なし（データ不足）")

    # レジーム別分布を集計（固定パラメータで期間中のレジームを確認）
    print("\n[レジーム判定サンプル]")
    sample_dates = rebal_dates[::4][:12]
    for d in sample_dates:
        if str(d.date()) >= start:
            r = detect_regime(nikkei, d)
            print(f"  {d.date()} → {r}")

    return r_fixed, r_regime


def safe_metrics(r):
    if r is None:
        return {"total_return_pct": None, "sharpe": None, "max_dd_pct": None, "alpha_pct": None}
    return {
        "total_return_pct": r.get("total_return_pct"),
        "sharpe": r.get("sharpe"),
        "max_dd_pct": r.get("max_dd_pct"),
        "alpha_pct": r.get("alpha_pct"),
        "nikkei_pct": r.get("nikkei_pct"),
        "n_trades": r.get("n_trades"),
    }


def main():
    print("レジーム適応型戦略 検証開始")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    codes = get_top_liquid_tickers(N_CODES)
    print(f"銘柄数: {len(codes)}")

    results = {
        "fixed_params": {},
        "regime_adaptive": {},
    }

    for label, (start, end) in PERIODS.items():
        r_fixed, r_regime = run_period(label, start, end, codes)
        results["fixed_params"][label] = safe_metrics(r_fixed)
        results["regime_adaptive"][label] = safe_metrics(r_regime)

    # 比較サマリー
    print("\n" + "="*60)
    print("=== 最終比較 ===")
    print("="*60)
    for label in PERIODS:
        f = results["fixed_params"][label]
        r = results["regime_adaptive"][label]
        print(f"\n{label}期間:")
        print(f"  固定パラメータ  : total={f.get('total_return_pct') or 'N/A':>8}  sharpe={f.get('sharpe') or 'N/A'}")
        print(f"  レジーム適応型  : total={r.get('total_return_pct') or 'N/A':>8}  sharpe={r.get('sharpe') or 'N/A'}")

    # 比較コメント
    comparison_lines = []
    for label in PERIODS:
        f = results["fixed_params"][label]
        r = results["regime_adaptive"][label]
        ft = f.get("total_return_pct")
        rt = r.get("total_return_pct")
        if ft is not None and rt is not None:
            diff = rt - ft
            comparison_lines.append(
                f"{label}: fixed={ft:+.2f}% vs regime={rt:+.2f}% (差={diff:+.2f}%)"
            )
        else:
            comparison_lines.append(f"{label}: データ不足")
    comparison_str = " | ".join(comparison_lines)

    results["comparison"] = comparison_str
    results["run_at"] = datetime.now().isoformat()
    results["params"] = {
        "best_params": best_params,
        "defensive_params": defensive_params,
        "regime_map": {k: ("None=cash" if v is None else "params") for k, v in regime_params.items()},
    }

    print(f"\n比較: {comparison_str}")

    # 結果をファイルに保存
    out_path = Path("backtest/regime_test_result.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果保存: {out_path}")

    # workspace への結果サマリー書き出し
    ws_path = Path("/Users/panda/.openclaw/workspace/backtest/regime_done.json")
    ws_path.parent.mkdir(exist_ok=True)
    summary = {
        "status": "done",
        "run_at": datetime.now().isoformat(),
        "comparison": comparison_str,
        "IS_fixed": results["fixed_params"].get("IS"),
        "IS_regime": results["regime_adaptive"].get("IS"),
        "OOS_fixed": results["fixed_params"].get("OOS"),
        "OOS_regime": results["regime_adaptive"].get("OOS"),
        "result_path": str(out_path.resolve()),
    }
    with open(ws_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"サマリー保存: {ws_path}")

    return results


if __name__ == "__main__":
    main()
