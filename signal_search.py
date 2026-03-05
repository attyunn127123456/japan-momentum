"""
signal_search.py — シグナル積み上げ式の最適化エンジン

Step1: Ablation test（各シグナルを0にした時のsharpe低下を測定）
Step2: 新シグナル追加テスト（delta_sharpe > 0.01 なら candidate に追加）
Step3: 全シグナル同時重み最適化（グリッドサーチ）
Step4: 結果をベースラインとして hypothesis_queue.json を更新
"""
import json, itertools, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from fetch_cache import read_ohlcv
from universe import get_top_liquid_tickers
from backtest import get_rebalance_dates, get_nikkei_history
from optimize import precompute, eval_params

QUEUE_FILE   = Path("backtest/hypothesis_queue.json")
LIBRARY_FILE = Path("backtest/signal_library.json")
START        = "2023-01-01"
END          = datetime.now().strftime("%Y-%m-%d")
N_CODES      = 200
DELTA_THRESHOLD = 0.01  # 緩い基準


def load_library():
    if LIBRARY_FILE.exists():
        return json.loads(LIBRARY_FILE.read_text())
    return {"signals": [], "current_weights": {}, "baseline_sharpe": 0.0, "last_updated": ""}


def save_library(lib):
    lib["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    LIBRARY_FILE.write_text(json.dumps(lib, ensure_ascii=False, indent=2))


def load_queue():
    return json.loads(QUEUE_FILE.read_text())


def save_queue(q):
    QUEUE_FILE.write_text(json.dumps(q, ensure_ascii=False, indent=2))


def load_prices_and_factors():
    codes = get_top_liquid_tickers(N_CODES)
    warmup = (datetime.strptime(START, "%Y-%m-%d") - timedelta(days=200)).strftime("%Y-%m-%d")
    prices_dict = {}
    for c in codes:
        df = read_ohlcv(c, warmup, END)
        if df is None or df.empty or "AdjC" not in df.columns:
            continue
        avg_price = df["AdjC"].tail(20).mean()
        if 200 <= avg_price <= 10000:
            prices_dict[c] = df
    print(f"  {len(prices_dict)}銘柄ロード完了", flush=True)
    nikkei = get_nikkei_history(warmup, END)
    factor_dfs = precompute(prices_dict, nikkei, [80])
    return_df = pd.DataFrame({c: prices_dict[c]["AdjC"] for c in prices_dict}).pct_change()
    rebal_dates = get_rebalance_dates(warmup, END, "weekly")
    return prices_dict, nikkei, factor_dfs, return_df, rebal_dates


def eval_with_weights(weights, factor_dfs, prices_dict, rebal_dates, nikkei, return_df):
    """weightsを受け取ってeval_paramsを呼ぶ。正規化済みを想定。"""
    p = {
        "lookback": 80, "top_n": 10, "rebalance": "weekly",
        **weights,
    }
    # デフォルト補完
    for k in ["ret_w", "rs_w", "green_w", "smooth_w", "resilience_w"]:
        p.setdefault(k, 0.0)
    return eval_params(p, factor_dfs, prices_dict, rebal_dates, nikkei, START, return_df)


# ========== Step1: Ablation Test ==========
def ablation_test(lib, factor_dfs, prices_dict, rebal_dates, nikkei, return_df):
    """各シグナルを0にした時のsharpe低下を測定"""
    print("\n[Step1] Ablation Test", flush=True)
    base_weights = lib["current_weights"].copy()

    # ベースラインsharpe
    r_base = eval_with_weights(base_weights, factor_dfs, prices_dict, rebal_dates, nikkei, return_df)
    if not r_base:
        print("  ベースライン計算失敗", flush=True)
        return lib
    base_sharpe = r_base["sharpe"]
    print(f"  ベースライン sharpe={base_sharpe:.3f}", flush=True)

    for sig in lib["signals"]:
        if sig["status"] not in ("active", "candidate"):
            continue
        key = sig.get("param_key") or (sig["id"] + "_w")
        # そのシグナルを0にする
        ablated = {**base_weights, key: 0.0}
        r = eval_with_weights(ablated, factor_dfs, prices_dict, rebal_dates, nikkei, return_df)
        if r:
            contribution = base_sharpe - r["sharpe"]
            sig["sharpe_contribution"] = round(contribution, 4)
            if contribution > 0:
                sig["status"] = "active"
                print(f"  ✅ {sig['id']}: sharpe低下 {contribution:+.4f} → active", flush=True)
            else:
                sig["status"] = "candidate"
                print(f"  ⚠️  {sig['id']}: sharpe低下 {contribution:+.4f} → candidate（他との相性あり）", flush=True)

    lib["baseline_sharpe"] = base_sharpe
    save_library(lib)
    return lib, base_sharpe


# ========== Step2: 新シグナル追加テスト ==========
NEW_SIGNAL_CANDIDATES = [
    {"id": "rsi14",         "desc": "RSI(14) — 過熱/過冷検出",       "param_key": "rsi14_w"},
    {"id": "ma_deviation",  "desc": "移動平均乖離率",                  "param_key": "ma_deviation_w"},
    {"id": "momentum_12_1", "desc": "12ヶ月-1ヶ月モメンタム（古典）", "param_key": "momentum_12_1_w"},
    {"id": "resilience",    "desc": "下落耐性スコア",                  "param_key": "resilience_w"},
    {"id": "eps_growth",    "desc": "EPS成長率",                       "param_key": "eps_growth_w"},
    {"id": "rev_growth",    "desc": "売上成長率",                      "param_key": "rev_growth_w"},
    {"id": "credit_ratio",  "desc": "信用倍率（買い残/売り残）",       "param_key": "credit_ratio_w"},
]

def test_new_signals(lib, base_sharpe, factor_dfs, prices_dict, rebal_dates, nikkei, return_df):
    """新シグナルを既存シグナルに追加して効果測定"""
    print("\n[Step2] 新シグナル追加テスト", flush=True)
    existing_ids = {s["id"] for s in lib["signals"]}
    base_weights = lib["current_weights"].copy()

    for cand in NEW_SIGNAL_CANDIDATES:
        if cand["id"] in existing_ids:
            continue
        key = cand["param_key"]
        # 既存重みはそのままに新シグナルを0.1で追加、全体を正規化
        new_weights = {**base_weights, key: 0.1}
        total = sum(v for v in new_weights.values() if v > 0)
        if total > 0:
            new_weights = {k: round(v / total, 4) for k, v in new_weights.items()}

        # factor_dfsにそのシグナルがあるか確認
        # （rsi14, ma_deviation, momentum_12_1, resilience_wはprecomputeで計算済み）
        # なければスキップ
        sample_key = next(iter(factor_dfs.keys()), None)
        if sample_key:
            cols = factor_dfs[sample_key].columns.tolist()
            factor_col = cand["id"].replace("_w", "")
            # resilience_wはresilienceとして保存されている場合もある
            has_factor = factor_col in cols or cand["id"].rstrip("_w") in cols
            if key not in ["resilience_w"] and not has_factor:
                print(f"  ⚠️  {cand['id']}: ファクターデータなし → スキップ", flush=True)
                continue

        r = eval_with_weights(new_weights, factor_dfs, prices_dict, rebal_dates, nikkei, return_df)
        if not r:
            continue
        delta = r["sharpe"] - base_sharpe
        print(f"  {cand['id']}: delta_sharpe={delta:+.4f}", flush=True)

        if delta > DELTA_THRESHOLD:
            lib["signals"].append({
                "id": cand["id"],
                "desc": cand["desc"],
                "type": "ohlcv",
                "param_key": key,
                "status": "candidate",
                "best_weight": 0.1,
                "sharpe_contribution": round(delta, 4),
                "added_at": datetime.now().strftime("%Y-%m-%d"),
            })
            print(f"  ✅ {cand['id']}: candidate追加", flush=True)

    save_library(lib)
    return lib


# ========== Step3: 全シグナル同時重み最適化 ==========
def optimize_all_signals(lib, factor_dfs, prices_dict, rebal_dates, nikkei, return_df):
    """active + candidate シグナル全部で重みグリッドサーチ"""
    print("\n[Step3] 全シグナル同時重み最適化", flush=True)

    active_signals = [s for s in lib["signals"] if s["status"] in ("active", "candidate")]
    if not active_signals:
        print("  シグナルなし、スキップ", flush=True)
        return lib

    keys = [s.get("param_key") or (s["id"] + "_w") for s in active_signals]
    print(f"  最適化対象: {keys}", flush=True)

    # グリッドサーチ（各重みを0.05刻み）
    weight_options = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    combos = list(itertools.product(weight_options, repeat=len(keys)))
    # 正規化してユニークな組み合わせに絞る
    print(f"  グリッド組み合わせ: {len(combos)}件（正規化後に重複除去）", flush=True)

    best_sharpe = lib.get("baseline_sharpe", 0.0)
    best_weights = lib["current_weights"].copy()
    tested = 0

    for vals in combos:
        total = sum(vals)
        if total == 0:
            continue
        norm = tuple(round(v / total, 4) for v in vals)
        w = dict(zip(keys, norm))
        # 不足分は0埋め
        for k in ["ret_w", "rs_w", "green_w", "smooth_w", "resilience_w"]:
            w.setdefault(k, 0.0)

        r = eval_with_weights(w, factor_dfs, prices_dict, rebal_dates, nikkei, return_df)
        if r and r["sharpe"] > best_sharpe:
            best_sharpe = r["sharpe"]
            best_weights = w.copy()
        tested += 1
        if tested % 500 == 0:
            print(f"  ... {tested}/{len(combos)} 完了, best_sharpe={best_sharpe:.3f}", flush=True)

    print(f"  最適化完了: best_sharpe={best_sharpe:.3f}", flush=True)
    print(f"  最適重み: {best_weights}", flush=True)

    # best_weightを各シグナルに保存
    for sig in lib["signals"]:
        k = sig.get("param_key") or (sig["id"] + "_w")
        if k in best_weights:
            sig["best_weight"] = best_weights[k]

    lib["current_weights"] = best_weights
    lib["baseline_sharpe"] = best_sharpe
    save_library(lib)
    return lib, best_sharpe, best_weights


# ========== Step4: hypothesis_queue.json を更新 ==========
def update_baseline(best_sharpe, best_weights):
    """最適化結果でベースラインを更新"""
    print("\n[Step4] ベースライン更新", flush=True)
    queue = load_queue()
    old_sharpe = queue["baseline"]["sharpe"]

    if best_sharpe > old_sharpe:
        queue["baseline"]["sharpe"] = best_sharpe
        queue["baseline"]["params"].update(best_weights)
        queue["baseline"]["date"] = datetime.now().strftime("%Y-%m-%d")
        queue["baseline"]["hypothesis"] = "signal_search"
        save_queue(queue)
        print(f"  ✅ ベースライン更新: {old_sharpe:.3f} → {best_sharpe:.3f}", flush=True)
    else:
        print(f"  ベースライン変化なし: {old_sharpe:.3f} >= {best_sharpe:.3f}", flush=True)


# ========== メイン ==========
def run_signal_search():
    print("\n=== Signal Search 開始 ===", flush=True)
    t0 = time.time()

    lib = load_library()
    prices_dict, nikkei, factor_dfs, return_df, rebal_dates = load_prices_and_factors()

    # Step1: Ablation test
    result = ablation_test(lib, factor_dfs, prices_dict, rebal_dates, nikkei, return_df)
    if isinstance(result, tuple):
        lib, base_sharpe = result
    else:
        lib = result
        base_sharpe = lib.get("baseline_sharpe", 0.0)

    # Step2: 新シグナル追加テスト
    lib = test_new_signals(lib, base_sharpe, factor_dfs, prices_dict, rebal_dates, nikkei, return_df)

    # Step3: 全シグナル最適化
    result3 = optimize_all_signals(lib, factor_dfs, prices_dict, rebal_dates, nikkei, return_df)
    if isinstance(result3, tuple):
        lib, best_sharpe, best_weights = result3
    else:
        lib = result3
        best_sharpe = lib.get("baseline_sharpe", base_sharpe)
        best_weights = lib["current_weights"]

    # Step4: ベースライン更新
    update_baseline(best_sharpe, best_weights)

    elapsed = time.time() - t0
    print(f"\n=== Signal Search 完了 ({elapsed:.0f}秒) ===", flush=True)
    print(f"  最終 baseline_sharpe={best_sharpe:.3f}", flush=True)


if __name__ == "__main__":
    run_signal_search()
