"""
レジームスコア連動型バックテスト

毎リバランス日に:
1. detect_regime() で今日の相場モードを取得
2. get_regime_params() でパラメータとposition_ratioを計算
3. そのパラメータで銘柄選択 + position_ratioでポジションサイズ調整

比較:
- 固定パラメータ (ベースライン): lookback=100, top_n=2, position_ratio=1.0
- レジーム連動型: スコア+確信度両方で調整

検証期間:
- IS:      2016-01-01 〜 2020-12-31
- OOS fold1: 2021-01-01 〜 2022-12-31
- OOS fold2: 2023-01-01 〜 2024-12-31
"""
from __future__ import annotations

import json
import time
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ─── プロジェクト内モジュール ───────────────────────────────────────────────
from fetch_cache import read_ohlcv
from universe import get_top_liquid_tickers
from backtest import get_rebalance_dates, get_nikkei_history
from optimize import precompute, build_score_df, compute_exit_signal
from market_regime import detect_regime, get_regime_params

# ─── ベースラインパラメータ ─────────────────────────────────────────────────
BASE_PARAMS = {
    'lookback':          100,
    'top_n':             2,
    'rebalance':         'weekly',
    'trailing_stop':     -0.0271,
    'exit_threshold':    0.160,
    'exit_lookback':     24,
    'exit_rsi_w':        0.432,
    'exit_vol_w':        0.230,
    'exit_dd_w':         0.150,
    'ret_w':             0.100,
    'upside_capture_w':  0.098,
    'green_w':           0.095,
    'win_streak_w':      0.089,
    'rs_w':              0.0,
}

# ─── 検証期間定義 ──────────────────────────────────────────────────────────
PERIODS = [
    {"name": "IS",      "start": "2016-01-01", "end": "2020-12-31"},
    {"name": "OOS2021", "start": "2021-01-01", "end": "2022-12-31"},
    {"name": "OOS2023", "start": "2023-01-01", "end": "2024-12-31"},
]

N_CODES    = 500   # 銘柄数（速度とカバレッジのバランス）
WARMUP_DAYS = 200  # ウォームアップ日数


# ─── ユーティリティ ────────────────────────────────────────────────────────

def _fb(o):
    """JSON safe変換"""
    if isinstance(o, bool):
        return int(o)
    if isinstance(o, dict):
        return {k: _fb(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_fb(i) for i in o]
    import math
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def calc_stats(portfolio_history: list, nikkei: pd.Series, start: str) -> dict:
    """ポートフォリオ履歴からパフォーマンス統計を計算する"""
    if len(portfolio_history) < 5:
        return {"total_return_pct": 0.0, "sharpe": 0.0, "max_dd_pct": 0.0}

    returns = np.array(portfolio_history)
    final_portfolio = np.prod(1 + returns)
    tr = final_portfolio - 1

    sharpe = float(returns.mean() / returns.std() * np.sqrt(52)) if returns.std() > 0 else 0.0  # weekly

    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = float(abs(((cum - peak) / peak).min()))

    nk = nikkei.loc[start:]
    nk_ret = float(nk.iloc[-1] / nk.iloc[0] - 1) if len(nk) > 1 else 0.0

    return {
        "total_return_pct": round(tr * 100, 2),
        "sharpe":           round(sharpe, 3),
        "max_dd_pct":       round(dd * 100, 2),
        "nikkei_pct":       round(nk_ret * 100, 2),
        "n_trades":         len(returns),
    }


def calc_period_return_raw(code, date, next_date, prices_dict, return_df, params):
    """1銘柄の1期間リターンを計算（trailing_stop + exit_signal対応）"""
    trailing_stop  = params.get("trailing_stop", None)
    exit_threshold = params.get("exit_threshold", None)

    use_trailing    = trailing_stop  is not None and code in prices_dict
    use_exit_signal = exit_threshold is not None and code in prices_dict

    if use_trailing or use_exit_signal:
        try:
            price_df = prices_dict[code]
            mask = (price_df.index > date) & (price_df.index <= next_date)
            period_prices = price_df.loc[mask, 'AdjC']
            if len(period_prices) > 0:
                prev = price_df.loc[price_df.index <= date, 'AdjC']
                if len(prev) == 0:
                    return None
                entry_price = prev.iloc[-1]
                if entry_price <= 0 or np.isnan(entry_price):
                    return None
                peak = entry_price
                exit_price = period_prices.iloc[-1]
                for idx, daily_price in period_prices.items():
                    if np.isnan(daily_price):
                        continue
                    peak = max(peak, daily_price)
                    if use_trailing and (daily_price - peak) / peak <= trailing_stop:
                        exit_price = daily_price
                        break
                    if use_exit_signal:
                        should_exit, _ = compute_exit_signal(price_df, idx, date, params)
                        if should_exit:
                            exit_price = daily_price
                            break
                ret = (exit_price - entry_price) / entry_price
                return ret if not np.isnan(ret) else None
        except Exception:
            pass

    if code in return_df.columns:
        r = return_df.at[next_date, code] if next_date in return_df.index else np.nan
        if not np.isnan(r):
            return r
    return None


# ─── 固定パラメータ バックテスト ───────────────────────────────────────────

def run_fixed_backtest(
    params: dict,
    factor_dfs: dict,
    prices_dict: dict,
    rebal_dates: list,
    nikkei: pd.Series,
    start: str,
    return_df: pd.DataFrame,
) -> dict:
    """
    固定パラメータでバックテストを実行する。
    position_ratio = 1.0（常にフル投資）。
    """
    lb = params["lookback"]
    tn = params["top_n"]

    weights = {k: params.get(k + "_w", 0.0) for k in [
        "ret", "rs", "green", "smooth", "resilience",
        "upside_capture", "win_streak",
    ]}
    score_df = build_score_df(factor_dfs, lb, weights)
    if score_df is None:
        return None

    dates = [d for d in rebal_dates if str(d.date()) >= start]
    portfolio = 1_000_000.0
    returns_list = []

    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]

        if date not in score_df.index:
            continue
        row = score_df.loc[date].dropna()
        if row.empty:
            continue
        top = row.nlargest(tn).index.tolist()

        if date not in return_df.index or next_date not in return_df.index:
            continue

        tot, cnt = 0.0, 0
        for code in top:
            r = calc_period_return_raw(code, date, next_date, prices_dict, return_df, params)
            if r is not None:
                tot += r
                cnt += 1

        if cnt > 0:
            r = tot / cnt
            portfolio *= (1 + r)
            returns_list.append(r)

    return calc_stats(returns_list, nikkei, start)


# ─── レジーム連動型 バックテスト ──────────────────────────────────────────

def run_regime_backtest(
    base_params: dict,
    factor_dfs: dict,
    prices_dict: dict,
    rebal_dates: list,
    nikkei: pd.Series,
    investor_types_df: pd.DataFrame,
    fins_summary_df: pd.DataFrame,
    start: str,
    return_df: pd.DataFrame,
) -> dict:
    """
    レジームスコア連動型バックテストを実行する。
    毎リバランス日に detect_regime() → get_regime_params() を呼んでパラメータを変更。
    """
    # 全lookbackのscore_dfをキャッシュ
    weights = {k: base_params.get(k + "_w", 0.0) for k in [
        "ret", "rs", "green", "smooth", "resilience",
        "upside_capture", "win_streak",
    ]}
    score_df_cache: dict[int, pd.DataFrame | None] = {}
    for lb in factor_dfs:
        score_df_cache[lb] = build_score_df(factor_dfs, lb, weights)

    dates = [d for d in rebal_dates if str(d.date()) >= start]
    portfolio = 1_000_000.0
    returns_list = []
    regime_log: list[str] = []

    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]

        # ── レジーム判定 ──
        try:
            regime_result = detect_regime(
                prices_dict, nikkei, investor_types_df, fins_summary_df, date
            )
        except Exception as e:
            regime_result = {"regime": "choppy", "score": 0.5}

        # ── パラメータ調整 ──
        adjusted = get_regime_params(regime_result, base_params)
        position_ratio = adjusted["position_ratio"]
        regime_log.append(regime_result.get("regime", "choppy"))

        # クラッシュ時はキャッシュ保持（完全パス）
        if position_ratio == 0.0:
            continue

        lb = adjusted["lookback"]
        tn = adjusted["top_n"]

        # lookbackに対応するscore_dfを取得（最近傍へのフォールバック）
        if lb not in score_df_cache:
            available_lbs = sorted(score_df_cache.keys())
            lb = min(available_lbs, key=lambda x: abs(x - lb))
        score_df = score_df_cache[lb]

        if score_df is None or date not in score_df.index:
            continue
        row = score_df.loc[date].dropna()
        if row.empty:
            continue
        top = row.nlargest(tn).index.tolist()

        if date not in return_df.index or next_date not in return_df.index:
            continue

        tot, cnt = 0.0, 0
        for code in top:
            r = calc_period_return_raw(code, date, next_date, prices_dict, return_df, adjusted)
            if r is not None:
                tot += r
                cnt += 1

        if cnt > 0:
            r = (tot / cnt) * position_ratio
            portfolio *= (1 + r)
            returns_list.append(r)

    stats = calc_stats(returns_list, nikkei, start)
    stats["regime_distribution"] = _calc_regime_dist(regime_log)
    return stats


def _calc_regime_dist(regime_log: list) -> dict:
    """レジーム分布を計算する"""
    if not regime_log:
        return {}
    counts = defaultdict(int)
    for r in regime_log:
        counts[r] += 1
    total = len(regime_log)
    return {k: round(v / total * 100, 1) for k, v in counts.items()}


# ─── メイン ───────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=== レジーム連動型バックテスト ===", flush=True)
    print(f"銘柄数: {N_CODES}", flush=True)

    # ── データ読み込み ────────────────────────────────────────────────────
    overall_start = PERIODS[0]["start"]
    overall_end   = PERIODS[-1]["end"]
    warmup_start  = (datetime.strptime(overall_start, "%Y-%m-%d")
                     - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")

    print(f"銘柄リスト取得...", flush=True)
    codes = get_top_liquid_tickers(N_CODES)
    print(f"{len(codes)}銘柄をロード中...", flush=True)

    prices_dict = {}
    for c in codes:
        df = read_ohlcv(c, warmup_start, overall_end)
        if df is not None and not df.empty and "AdjC" in df.columns:
            prices_dict[c] = df

    print(f"{len(prices_dict)}銘柄ロード完了 ({time.time()-t0:.1f}秒)", flush=True)

    nikkei = get_nikkei_history(warmup_start, overall_end)

    # ── 投資家別売買データ・決算データ読み込み ─────────────────────────
    print("投資家別データ・決算データ読み込み...", flush=True)
    try:
        investor_types_df = pd.read_parquet("data/fundamentals/investor_types.parquet")
    except Exception as e:
        print(f"  investor_types.parquet 読み込みエラー: {e}", flush=True)
        investor_types_df = pd.DataFrame(columns=["PubDate", "FrgnBal"])

    try:
        fins_summary_df = pd.read_parquet("data/fundamentals/fins_summary.parquet")
    except Exception as e:
        print(f"  fins_summary.parquet 読み込みエラー: {e}", flush=True)
        fins_summary_df = pd.DataFrame(columns=["DiscDate", "Code", "EPS", "FEPS"])

    # ── ファクター事前計算 ────────────────────────────────────────────────
    # レジーム連動では lookback が 40〜100 の範囲で変化するので複数lookbookを計算
    lookbacks_needed = [40, 60, 80, 100]
    print(f"ファクター計算 (lookback={lookbacks_needed})...", flush=True)
    factor_dfs = precompute(prices_dict, nikkei, lookbacks_needed)

    # ── リターンDF構築 ────────────────────────────────────────────────────
    print("リターンDF構築...", flush=True)
    all_prices = pd.DataFrame({c: prices_dict[c]["AdjC"] for c in prices_dict}).astype(float)
    return_df  = all_prices.pct_change()

    # ── リバランス日（全期間分を一括取得）─────────────────────────────────
    rebal_dates_all = get_rebalance_dates(warmup_start, overall_end, BASE_PARAMS["rebalance"])

    # ── 各期間でバックテスト実行 ──────────────────────────────────────────
    results_fixed   = {}
    results_regime  = {}

    for period in PERIODS:
        name  = period["name"]
        start = period["start"]
        end   = period["end"]
        rebal_dates = [d for d in rebal_dates_all if start <= str(d.date()) <= end]

        print(f"\n--- {name} ({start} 〜 {end}) ---", flush=True)

        # 固定パラメータ
        print(f"  固定パラメータ実行中...", flush=True)
        r_fixed = run_fixed_backtest(
            BASE_PARAMS, factor_dfs, prices_dict,
            rebal_dates, nikkei, start, return_df
        )
        results_fixed[name] = r_fixed or {}
        if r_fixed:
            print(f"  固定: total={r_fixed['total_return_pct']:+.1f}% "
                  f"S={r_fixed['sharpe']:.2f} DD={r_fixed['max_dd_pct']:.1f}%", flush=True)

        # レジーム連動型
        print(f"  レジーム連動型実行中...", flush=True)
        r_regime = run_regime_backtest(
            BASE_PARAMS, factor_dfs, prices_dict,
            rebal_dates, nikkei,
            investor_types_df, fins_summary_df,
            start, return_df
        )
        results_regime[name] = r_regime or {}
        if r_regime:
            print(f"  レジーム: total={r_regime['total_return_pct']:+.1f}% "
                  f"S={r_regime['sharpe']:.2f} DD={r_regime['max_dd_pct']:.1f}%", flush=True)
            dist = r_regime.get("regime_distribution", {})
            if dist:
                dist_str = "  ".join(f"{k}: {v}%" for k, v in sorted(dist.items()))
                print(f"  レジーム分布: {dist_str}", flush=True)

    # ── 結果出力 ─────────────────────────────────────────────────────────
    print("\n\n=== レジーム連動型バックテスト結果 ===\n")
    header = f"{'':12}{'固定パラメータ':>20}{'レジーム連動型':>20}"
    print(header)
    print("-" * len(header))

    for period in PERIODS:
        name = period["name"]
        f = results_fixed.get(name, {})
        r = results_regime.get(name, {})

        f_str = (f"{f['total_return_pct']:+.1f}% S={f['sharpe']:.2f}"
                 if f else "N/A")
        r_str = (f"{r['total_return_pct']:+.1f}% S={r['sharpe']:.2f}"
                 if r else "N/A")
        print(f"{name:<12}{f_str:>20}{r_str:>20}")

    for period in PERIODS:
        name = period["name"]
        r = results_regime.get(name, {})
        dist = r.get("regime_distribution", {})
        if dist:
            print(f"\n[レジーム分布 {name}]")
            for regime_name, pct in sorted(dist.items()):
                print(f"  {regime_name}: {pct}%")

    # ── JSON保存 ─────────────────────────────────────────────────────────
    Path("backtest").mkdir(exist_ok=True)

    result_data = {
        "run_at":        datetime.now().isoformat(),
        "n_codes":       len(prices_dict),
        "base_params":   BASE_PARAMS,
        "periods":       PERIODS,
        "fixed":         results_fixed,
        "regime":        results_regime,
        "elapsed_sec":   round(time.time() - t0, 1),
    }

    out_path = Path("backtest/regime_backtest_result.json")
    out_path.write_text(
        json.dumps(_fb(result_data), ensure_ascii=False, indent=2)
    )
    print(f"\n結果を {out_path} に保存しました", flush=True)

    # ── ワークスペースに完了サマリーを書き出し ──────────────────────────
    workspace_dir = Path("/Users/panda/.openclaw/workspace/backtest")
    workspace_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "completed_at": datetime.now().isoformat(),
        "elapsed_sec":  round(time.time() - t0, 1),
        "n_codes":      len(prices_dict),
        "periods": {
            name: {
                "fixed":  results_fixed.get(name, {}),
                "regime": {k: v for k, v in results_regime.get(name, {}).items()
                           if k != "regime_distribution"},
                "regime_distribution": results_regime.get(name, {}).get(
                    "regime_distribution", {}),
            }
            for name in [p["name"] for p in PERIODS]
        },
        "comparison_note": (
            "OOS2021 fixed vs regime: "
            f"fixed={results_fixed.get('OOS2021', {}).get('total_return_pct', 'N/A')}%, "
            f"regime={results_regime.get('OOS2021', {}).get('total_return_pct', 'N/A')}%"
        ),
    }

    (workspace_dir / "regime_backtest_done.json").write_text(
        json.dumps(_fb(summary), ensure_ascii=False, indent=2)
    )
    print(f"完了サマリーを {workspace_dir}/regime_backtest_done.json に保存しました",
          flush=True)
    print(f"\n総実行時間: {time.time()-t0:.1f}秒", flush=True)


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    main()
