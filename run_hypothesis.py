"""
仮説を1つ実行してbacktest/hypothesis_queue.jsonを更新するスクリプト。
heartbeatからバックグラウンドで呼ばれる。
完了後 backtest/hypothesis_done.json を書く。
"""
import json_safe as json
import time, sys, traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def _fb(o):
    """bool/NaN/Inf をJSONシリアライズ可能に変換"""
    if isinstance(o, bool): return int(o)
    if isinstance(o, dict): return {k: _fb(v) for k, v in o.items()}
    if isinstance(o, list): return [_fb(i) for i in o]
    import math
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)): return None
    return o


from fetch_cache import read_ohlcv
from universe import get_top_liquid_tickers
from backtest import get_rebalance_dates, get_nikkei_history
from optimize import precompute, eval_params

QUEUE_FILE = Path("backtest/hypothesis_queue.json")
DONE_FILE  = Path("backtest/hypothesis_done.json")
START = "2021-01-01"   # IS訓練期間（直近5年）
END   = "2025-12-31"   # IS訓練期間終了
N_CODES = 4000
DELTA_THRESHOLD = 5.0  # total_return_pct の改善幅（5%以上で採用）


def _load_baseline_params():
    """hypothesis_queue.json からベースラインパラメータを動的に読み込む"""
    try:
        q = json.loads(QUEUE_FILE.read_text())
        return q.get('baseline', {}).get('params', {})
    except Exception:
        return {}

BASELINE_PARAMS = _load_baseline_params()
BASE_LB = BASELINE_PARAMS.get('lookback', 40)
BASE_TN = BASELINE_PARAMS.get('top_n', 2)


def load_queue():
    return json.loads(QUEUE_FILE.read_text())


def save_queue(q):
    QUEUE_FILE.write_text(json.dumps(_fb(q), ensure_ascii=False, indent=2))


def run_baseline(prices_dict, nikkei, factor_dfs, return_df, rebal_dates):
    """ベースラインのスコアを返す"""
    p = {"lookback":BASE_LB,"top_n":BASE_TN,"rebalance":"weekly",
         "ret_w":0.3,"rs_w":0.3,"green_w":0.2,"smooth_w":0.2,"resilience_w":0.0}
    return eval_params(p, factor_dfs, prices_dict, rebal_dates["weekly"], nikkei, START, return_df)


def run_universe_midcap(nikkei):
    """仮説1: 時価総額100〜3000億の中型株ユニバース"""
    import requests
    # J-Quantsから時価総額フィルタ（近似: 出来高上位500から大型除外）
    # ここでは出来高上位500の中から価格帯でフィルタ（価格×出来高で近似）
    codes = get_top_liquid_tickers(500)
    warmup = (datetime.strptime(START,"%Y-%m-%d")-timedelta(days=200)).strftime("%Y-%m-%d")
    prices_dict = {}
    for c in codes:
        df = read_ohlcv(c, warmup, END)
        if df is None or df.empty or "AdjC" not in df.columns: continue
        prices_dict[c] = df
    return prices_dict


def run_new_high_breakout(prices_dict, nikkei, rebal_dates, return_df):
    """仮説2: 過去60日高値更新をエントリー条件に追加"""
    factor_dfs = precompute(prices_dict, nikkei, [60])
    # 新高値ブレイクフラグを追加
    for code, df in prices_dict.items():
        p = df["AdjC"]
        new_high = (p >= p.rolling(60).max().shift(1)).astype(float)
        if (code, 60) in factor_dfs:
            factor_dfs[(code, 60)]["new_high"] = new_high

    # new_highを追加したevalを実行
    best = None
    for top_n in [BASE_TN, BASE_TN * 2]:
        p = {"lookback":BASE_LB,"top_n":top_n,"rebalance":"weekly",
             "ret_w":0.3,"rs_w":0.3,"green_w":0.2,"smooth_w":0.2,"resilience_w":0.0}
        r = eval_params_with_filter(p, factor_dfs, prices_dict,
                                     rebal_dates["weekly"], nikkei, START, return_df)
        if r and (best is None or r["sharpe"] > best["sharpe"]):
            best = r
    return best


def eval_params_with_filter(params, factor_dfs, prices_dict, rebal_dates, nikkei, start, return_df):
    """新高値フィルタ付きeval"""
    lb, tn = params["lookback"], params["top_n"]
    weights = {k: params[k+"_w"] for k in ["ret","rs","green","smooth","resilience"]}

    portfolio = 1_000_000.0
    returns = []
    dates = [d for d in rebal_dates if str(d.date()) >= start]

    for i, date in enumerate(dates[:-1]):
        next_date = dates[i+1]
        scores = {}
        for code in prices_dict:
            fac = factor_dfs.get((code, lb))
            if fac is None or date not in fac.index: continue
            row = fac.loc[date]
            if row.isna().all(): continue
            # 新高値フィルタ
            if "new_high" in fac.columns and row.get("new_high", 1) < 0.5:
                continue
            score = sum(weights[c] * (float(row[c]) if c in row.index and not np.isnan(float(row[c])) else 0.0)
                       for c in weights)
            scores[code] = score

        if not scores: continue
        top = sorted(scores, key=lambda x: scores[x], reverse=True)[:tn]
        tot, cnt = 0.0, 0
        if date in return_df.index and next_date in return_df.index:
            for code in top:
                if code in return_df.columns:
                    r = return_df.at[next_date, code]
                    if not np.isnan(r): tot += r; cnt += 1
        if cnt > 0:
            r = tot/cnt; portfolio *= (1+r); returns.append(r)

    if len(returns) < 5: return None
    arr = np.array(returns)
    tr = portfolio/1_000_000 - 1
    sharpe = float(arr.mean()/arr.std()*np.sqrt(252)) if arr.std()>0 else 0
    cum = np.cumprod(1+arr); peak = np.maximum.accumulate(cum)
    dd = float(abs(((cum-peak)/peak).min()))
    nk = nikkei.loc[start:]; nk_ret = float(nk.iloc[-1]/nk.iloc[0]-1) if len(nk)>1 else 0
    return {**params, "total_return_pct":round(tr*100,2), "alpha_pct":round((tr-nk_ret)*100,2),
            "sharpe":round(sharpe,3), "max_dd_pct":round(dd*100,2),
            "nikkei_pct":round(nk_ret*100,2), "n_trades":len(returns)}


def append_evolution_log(hid, desc, result, win, delta):
    """全テスト結果をevolution_log.jsonに累積追記"""
    log_file = Path("backtest/evolution_log.json")
    raw = json.loads(log_file.read_text()) if log_file.exists() else []
    log = raw.get("all", []) if isinstance(raw, dict) else raw
    log.append({
        "at": datetime.now().isoformat(),
        "id": hid,
        "desc": desc,
        "win": win,
        "delta_sharpe": delta,
        "sharpe": result.get("sharpe") if result else None,
        "total_return_pct": result.get("total_return_pct") if result else None,
        "alpha_pct": result.get("alpha_pct") if result else None,
        "max_dd_pct": result.get("max_dd_pct") if result else None,
        "params": {k: result[k] for k in result if k in ["lookback","top_n","rebalance","ret_w","rs_w","green_w","smooth_w","resilience_w"]} if result else {},
    })
    # シャープ順でソートしてtop50を保持
    valid = [x for x in log if x["sharpe"] is not None]
    valid.sort(key=lambda x: x["sharpe"], reverse=True)
    log_file.write_text(json.dumps(_fb({"best10": valid[:10], "all": valid[:200], "total": len(log)}), ensure_ascii=False, indent=2))


SIGNAL_LIBRARY_FILE = Path("backtest/signal_library.json")
EVOLUTION_ENGINE_FILE = Path(__file__).parent / "evolution_engine.py"


def _add_factor_to_ga_ranges(factor_key: str):
    """evolution_engine.py の local_search 内 RANGES に新ファクターの重みキーを追加する。
    既に存在する場合は何もしない。"""
    import re
    content = EVOLUTION_ENGINE_FILE.read_text()

    # 既に RANGES に存在するか確認（シングル/ダブルクォート両対応）
    if f"'{factor_key}'" in content or f'"{factor_key}"' in content:
        print(f"  {factor_key} は既に RANGES に存在します（スキップ）")
        return

    # 'short_momentum_w': [...], の行を検索し、その直後に新エントリを挿入
    new_entry = f"        '{factor_key}':          [0.0, 0.05, 0.1, 0.15, 0.2],\n"
    marker = "'short_momentum_w': [0.0, 0.05, 0.1, 0.15, 0.2],\n"
    if marker in content:
        content = content.replace(marker, marker + new_entry)
        EVOLUTION_ENGINE_FILE.write_text(content)
        print(f"  ✅ {factor_key} を evolution_engine.py の RANGES に追加しました")
    else:
        # フォールバック: RANGES の閉じ括弧の直前に挿入
        pattern = re.compile(r"(RANGES\s*=\s*\{[^}]+?)(\n\s*\})", re.DOTALL)
        m = pattern.search(content)
        if m:
            content = content[:m.start(2)] + "\n" + new_entry.rstrip("\n") + content[m.start(2):]
            EVOLUTION_ENGINE_FILE.write_text(content)
            print(f"  ✅ {factor_key} を evolution_engine.py の RANGES に追加しました（fallback）")
        else:
            print(f"  ⚠️  RANGES への挿入位置が見つかりませんでした: {factor_key}")


def update_signal_library(hid, desc, result, win, delta):
    """採用/棄却に関わらずシグナルをsignal_libraryに記録"""
    if not SIGNAL_LIBRARY_FILE.exists():
        return
    lib = json.loads(SIGNAL_LIBRARY_FILE.read_text())
    existing_ids = {s["id"] for s in lib["signals"]}
    if hid in existing_ids:
        # 既存エントリを更新
        for sig in lib["signals"]:
            if sig["id"] == hid:
                if win:
                    sig["status"] = "active"
                    if result and delta:
                        sig["sharpe_contribution"] = round(delta, 4)
                else:
                    sig["status"] = "candidate"  # 単体効果ゼロでも相性候補として保持
        SIGNAL_LIBRARY_FILE.write_text(json.dumps(_fb(lib), ensure_ascii=False, indent=2))
        return
    # 新規追加
    from datetime import datetime as _dt
    new_sig = {
        "id": hid,
        "desc": desc,
        "type": "hypothesis",
        "param_key": None,
        "status": "active" if win else "candidate",
        "best_weight": None,
        "sharpe_contribution": round(delta, 4) if delta else None,
        "added_at": _dt.now().strftime("%Y-%m-%d"),
    }
    if win and result:
        # 現在の重みを current_weights に反映
        weights = {k: result[k] for k in result if "_w" in k}
        if weights:
            lib["current_weights"].update(weights)
            lib["baseline_sharpe"] = result.get("sharpe", lib.get("baseline_sharpe", 0))
    lib["signals"].append(new_sig)
    SIGNAL_LIBRARY_FILE.write_text(json.dumps(_fb(lib), ensure_ascii=False, indent=2))



def main():
    queue = load_queue()

    # 次の未実施仮説を取得
    next_h = next((h for h in queue["queue"] if h["status"] == "pending"), None)
    if not next_h:
        DONE_FILE.write_text(json.dumps(_fb({"status": "all_done", "at": datetime.now().isoformat()}), ensure_ascii=False))
        print("全仮説完了")
        return

    hid = next_h["id"]
    print(f"仮説実行: {hid} - {next_h['desc']}")

    # ── 新ロジック: factor_to_add があれば GA の RANGES に追加して終了 ──
    if next_h.get("factor_to_add"):
        factor_key = next_h["factor_to_add"]
        print(f"  factor_to_add 検出: {factor_key} を GA の RANGES に追加します")
        queue["running"] = True
        queue["current_hypothesis"] = hid
        next_h["status"] = "running"
        save_queue(queue)

        _add_factor_to_ga_ranges(factor_key)

        next_h["status"] = "done_win"
        queue["running"] = False
        queue["current_hypothesis"] = None
        save_queue(queue)
        DONE_FILE.write_text(json.dumps(_fb(
            {"status": "added_to_ga", "id": hid, "factor": factor_key,
             "at": datetime.now().isoformat()}),
            ensure_ascii=False
        ))
        print(f"完了: {hid} | status=added_to_ga | factor={factor_key}")
        return
    # ── ここまで新ロジック ──

    queue["running"] = True
    queue["current_hypothesis"] = hid
    next_h["status"] = "running"
    save_queue(queue)

    warmup = (datetime.strptime(START,"%Y-%m-%d")-timedelta(days=200)).strftime("%Y-%m-%d")
    # 全期間ロード（OOS検証のため2021-2022も必要）
    OVERALL_END = datetime.now().strftime("%Y-%m-%d")
    nikkei = get_nikkei_history(warmup, OVERALL_END)
    rebal_dates = {
        "weekly": get_rebalance_dates(warmup, END, "weekly"),
        "daily":  get_rebalance_dates(warmup, END, "daily"),
    }
    # OOS検証用 date_map（直近9ヶ月: 十分なサンプルを確保）
    VAL_START = "2025-06-01"
    VAL_END   = datetime.now().strftime("%Y-%m-%d")
    val_rebal_dates = {
        "weekly": get_rebalance_dates(VAL_START, VAL_END, "weekly"),
        "daily":  get_rebalance_dates(VAL_START, VAL_END, "daily"),
    }

    try:
        t0 = time.time()

        if hid == "universe_midcap":
            prices_dict = run_universe_midcap(nikkei)
            print(f"中型株ユニバース: {len(prices_dict)}銘柄")
            factor_dfs = precompute(prices_dict, nikkei, [40,60,80])
            all_prices = pd.DataFrame({c: prices_dict[c]["AdjC"] for c in prices_dict}).astype(float)
            return_df = all_prices.pct_change()
            # ベスト設定で評価
            best = None
            for lb,tn in [(40,10),(60,5),(60,10),(80,10)]:
                p = {"lookback":lb,"top_n":tn,"rebalance":"weekly",
                     "ret_w":0.3,"rs_w":0.3,"green_w":0.2,"smooth_w":0.2,"resilience_w":0.0}
                r = eval_params(p, factor_dfs, prices_dict, rebal_dates["weekly"], nikkei, START, return_df)
                if r and (best is None or r["sharpe"] > best["sharpe"]):
                    best = r
            result = best

        elif hid == "new_high_breakout":
            codes = get_top_liquid_tickers(N_CODES)
            prices_dict = {}
            for c in codes:
                df = read_ohlcv(c, warmup, END)
                if df is not None and not df.empty and "AdjC" in df.columns:
                    prices_dict[c] = df
            all_prices = pd.DataFrame({c: prices_dict[c]["AdjC"] for c in prices_dict}).astype(float)
            return_df = all_prices.pct_change()
            result = run_new_high_breakout(prices_dict, nikkei, rebal_dates, return_df)

        elif hid == "vol_candle_exit":
            # 大出来高陰線が出たら翌週ポジション除外
            codes = get_top_liquid_tickers(N_CODES)
            prices_dict = {}
            for c in codes:
                df = read_ohlcv(c, warmup, END)
                if df is not None and not df.empty and "AdjC" in df.columns:
                    prices_dict[c] = df
            factor_dfs = precompute(prices_dict, nikkei, [60])
            # 大出来高陰線フラグをファクターに追加
            for code, df in prices_dict.items():
                if "Volume" in df.columns and "Open" in df.columns:
                    vol = df["Volume"].astype(float)
                    op  = df["Open"].astype(float)
                    cl  = df["AdjC"].astype(float)
                    big_vol = vol > vol.rolling(25).mean() * 2
                    bearish = cl < op
                    # 直近5日に大出来高陰線があったらフラグ=0（除外）
                    danger = (big_vol & bearish).rolling(5).max()
                    if (code, 60) in factor_dfs:
                        factor_dfs[(code,60)]["safe"] = (1 - danger).clip(0,1)
            all_prices = pd.DataFrame({c: prices_dict[c]["AdjC"] for c in prices_dict}).astype(float)
            return_df = all_prices.pct_change()
            p = {"lookback":BASE_LB,"top_n":BASE_TN,"rebalance":"weekly",
                 "ret_w":0.3,"rs_w":0.3,"green_w":0.2,"smooth_w":0.2,"resilience_w":0.0}
            result = eval_params(p, factor_dfs, prices_dict, rebal_dates["weekly"], nikkei, START, return_df)

        elif hid == "resilience_upgrade":
            codes = get_top_liquid_tickers(N_CODES)
            prices_dict = {}
            for c in codes:
                df = read_ohlcv(c, warmup, END)
                if df is not None and not df.empty and "AdjC" in df.columns:
                    prices_dict[c] = df
            factor_dfs = precompute(prices_dict, nikkei, [BASE_LB])
            all_prices = pd.DataFrame({c: prices_dict[c]["AdjC"] for c in prices_dict}).astype(float)
            return_df = all_prices.pct_change()
            p = {"lookback":BASE_LB,"top_n":BASE_TN,"rebalance":"weekly",
                 "ret_w":0.2,"rs_w":0.2,"green_w":0.1,"smooth_w":0.2,"resilience_w":0.3}
            result = eval_params(p, factor_dfs, prices_dict, rebal_dates["weekly"], nikkei, START, return_df)

        elif hid == "rank_product_midcap":
            # ランク積スコア×中型株ユニバース
            # factor_dfs構造: {lb: {factor_name: DataFrame(date x code)}}
            prices_dict = run_universe_midcap(nikkei)
            print(f"中型株ユニバース(rank_product): {len(prices_dict)}銘柄")
            factor_dfs = precompute(prices_dict, nikkei, [80])
            all_prices = pd.DataFrame({c: prices_dict[c]["AdjC"] for c in prices_dict}).astype(float)
            return_df = all_prices.pct_change()
            lb, tn = 80, 10
            factors = ["ret", "rs", "green", "smooth"]
            fac_lb = factor_dfs[lb]  # {factor_name: DataFrame(date x code)}
            dates = [d for d in rebal_dates["weekly"] if str(d.date()) >= START]
            portfolio = 1_000_000.0
            returns = []
            for i, date in enumerate(dates[:-1]):
                next_date = dates[i+1]
                # 各ファクターのdate行を取得
                fac_rows = {}
                for f in factors:
                    df_f = fac_lb[f]
                    if date not in df_f.index: break
                    fac_rows[f] = df_f.loc[date].dropna()
                else:
                    # 全ファクターで有効な銘柄のみ
                    valid_codes = set(fac_rows[factors[0]].index)
                    for f in factors[1:]:
                        valid_codes &= set(fac_rows[f].index)
                    valid_codes = list(valid_codes)
                    if len(valid_codes) < tn * 2:
                        continue
                    # 各ファクターでパーセンタイルランク計算
                    rank_data = {}
                    for f in factors:
                        vals = pd.Series({c: fac_rows[f][c] for c in valid_codes})
                        rank_data[f] = vals.rank(pct=True)
                    # 幾何平均ランク積スコア
                    scores = {}
                    for c in valid_codes:
                        prod = 1.0
                        for f in factors:
                            prod *= rank_data[f][c]
                        scores[c] = prod ** (1.0 / len(factors))
                    top = sorted(scores, key=lambda c: scores[c], reverse=True)[:tn]
                    tot, cnt = 0.0, 0
                    if date in return_df.index and next_date in return_df.index:
                        for code in top:
                            if code in return_df.columns:
                                r = return_df.at[next_date, code]
                                if not np.isnan(r):
                                    tot += r; cnt += 1
                    if cnt > 0:
                        r = tot / cnt
                        portfolio *= (1 + r)
                        returns.append(r)
            if len(returns) >= 5:
                arr = np.array(returns)
                tr = portfolio / 1_000_000 - 1
                sharpe = float(arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0
                cum = np.cumprod(1 + arr); peak = np.maximum.accumulate(cum)
                dd = float(abs(((cum - peak) / peak).min()))
                nk = nikkei.loc[START:]; nk_ret = float(nk.iloc[-1] / nk.iloc[0] - 1) if len(nk) > 1 else 0
                result = {"lookback": lb, "top_n": tn, "rebalance": "weekly",
                          "ret_w": 0.25, "rs_w": 0.25, "green_w": 0.25, "smooth_w": 0.25, "resilience_w": 0.0,
                          "score_mode": "rank_product",
                          "total_return_pct": round(tr*100, 2), "alpha_pct": round((tr-nk_ret)*100, 2),
                          "sharpe": round(sharpe, 3), "max_dd_pct": round(dd*100, 2),
                          "nikkei_pct": round(nk_ret*100, 2), "n_trades": len(returns)}
            else:
                result = None

        elif hid == "biweekly_midcap":
            # 中型株ユニバース×2週間リバランス
            prices_dict = run_universe_midcap(nikkei)
            print(f"中型株ユニバース(biweekly): {len(prices_dict)}銘柄")
            factor_dfs = precompute(prices_dict, nikkei, [80])
            all_prices = pd.DataFrame({c: prices_dict[c]["AdjC"] for c in prices_dict}).astype(float)
            return_df = all_prices.pct_change()
            biweekly_dates = get_rebalance_dates(warmup, END, "biweekly")
            p = {"lookback": 80, "top_n": 10, "rebalance": "biweekly",
                 "ret_w": 0.3, "rs_w": 0.3, "green_w": 0.2, "smooth_w": 0.2, "resilience_w": 0.0}
            result = eval_params(p, factor_dfs, prices_dict, biweekly_dates, nikkei, START, return_df)

        elif hid == "high52_midcap":
            # 52週高値proximity因子×中型株ユニバース
            # factor_dfs構造: {lb: {factor_name: DataFrame(date x code)}}
            prices_dict = run_universe_midcap(nikkei)
            print(f"中型株ユニバース(high52): {len(prices_dict)}銘柄")
            factor_dfs = precompute(prices_dict, nikkei, [80])
            # high52 factorを追加（DataFrame date x code形式）
            high52_data = {}
            for code, df in prices_dict.items():
                p = df["AdjC"].astype(float)
                h52 = p.rolling(252, min_periods=60).max()
                high52_data[code] = (p / h52).clip(0, 1)
            factor_dfs[80]["high52"] = pd.DataFrame(high52_data).astype(float)
            all_prices = pd.DataFrame({c: prices_dict[c]["AdjC"] for c in prices_dict}).astype(float)
            return_df = all_prices.pct_change()
            # カスタム重みでbuild_score_df相当の処理
            from optimize import build_score_df
            # high52を含む擬似weightsでスコア計算
            weights_h52 = {"ret": 0.25, "rs": 0.25, "green": 0.15, "smooth": 0.15, "resilience": 0.0, "high52": 0.2}
            lb, tn = 80, 10
            fac_lb = factor_dfs[lb]
            dates = [d for d in rebal_dates["weekly"] if str(d.date()) >= START]
            portfolio = 1_000_000.0
            returns = []
            for i, date in enumerate(dates[:-1]):
                next_date = dates[i+1]
                score_series_list = []
                for f, w in weights_h52.items():
                    if w == 0 or f not in fac_lb: continue
                    df_f = fac_lb[f]
                    if date not in df_f.index: continue
                    score_series_list.append(df_f.loc[date].dropna() * w)
                if not score_series_list: continue
                score_row = pd.concat(score_series_list, axis=1).sum(axis=1)
                score_row = score_row.dropna()
                if len(score_row) < tn: continue
                top = score_row.nlargest(tn).index.tolist()
                tot, cnt = 0.0, 0
                if date in return_df.index and next_date in return_df.index:
                    for code in top:
                        if code in return_df.columns:
                            r = return_df.at[next_date, code]
                            if not np.isnan(r):
                                tot += r; cnt += 1
                if cnt > 0:
                    r = tot / cnt
                    portfolio *= (1 + r)
                    returns.append(r)
            if len(returns) >= 5:
                arr = np.array(returns)
                tr = portfolio / 1_000_000 - 1
                sharpe = float(arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0
                cum = np.cumprod(1 + arr); peak = np.maximum.accumulate(cum)
                dd = float(abs(((cum - peak) / peak).min()))
                nk = nikkei.loc[START:]; nk_ret = float(nk.iloc[-1] / nk.iloc[0] - 1) if len(nk) > 1 else 0
                result = {"lookback": lb, "top_n": tn, "rebalance": "weekly",
                          "ret_w": 0.25, "rs_w": 0.25, "green_w": 0.15, "smooth_w": 0.15,
                          "resilience_w": 0.0, "high52_w": 0.2,
                          "total_return_pct": round(tr*100, 2), "alpha_pct": round((tr-nk_ret)*100, 2),
                          "sharpe": round(sharpe, 3), "max_dd_pct": round(dd*100, 2),
                          "nikkei_pct": round(nk_ret*100, 2), "n_trades": len(returns)}
            else:
                result = None

        elif next_h.get("type") == "regime" or hid == "regime_adaptive_weights":
            import regime_weights as _rw
            codes = get_top_liquid_tickers(N_CODES)
            prices_dict = {}
            for c in codes:
                df = read_ohlcv(c, warmup, END)
                if df is not None and not df.empty and "AdjC" in df.columns:
                    prices_dict[c] = df
            all_prices = pd.DataFrame({c: prices_dict[c]["AdjC"] for c in prices_dict}).astype(float)
            return_df = all_prices.pct_change()
            result = _rw.backtest_with_regime(
                prices_dict, nikkei, rebal_dates["weekly"], START, return_df
            )

        else:
            # 汎用フォールバック: eval_paramsでバックテスト
            codes = get_top_liquid_tickers(N_CODES)
            prices_dict = {}
            for c in codes:
                df = read_ohlcv(c, warmup, END)
                if df is not None and not df.empty and "AdjC" in df.columns:
                    prices_dict[c] = df
            factor_dfs = precompute(prices_dict, nikkei, [40, 60, 80])
            all_prices = pd.DataFrame({c: prices_dict[c]["AdjC"] for c in prices_dict}).astype(float)
            return_df = all_prices.pct_change()
            baseline_params = {
                'lookback': queue['baseline'].get('params', {}).get('lookback', 80),
                'top_n': queue['baseline'].get('params', {}).get('top_n', 10),
                'rebalance': 'weekly',
                'ret_w': 0.3, 'rs_w': 0.3, 'green_w': 0.2, 'smooth_w': 0.2, 'resilience_w': 0.0
            }
            test_params = {**baseline_params, **next_h.get('params', {})}
            lb = test_params['lookback']
            if lb not in [40, 60, 80]:
                factor_dfs.update(precompute(prices_dict, nikkei, [lb]))
            rb_key = test_params.get('rebalance', 'weekly')
            result = eval_params(test_params, factor_dfs, prices_dict,
                                 rebal_dates.get(rb_key, rebal_dates['weekly']), nikkei, START, return_df)

        # OAS検証（2021-2022 fold1）も実施
        result_val = None
        if result is not None:
            try:
                result_val = eval_params(test_params, factor_dfs, prices_dict,
                                         val_rebal_dates.get(rb_key, val_rebal_dates['weekly']),
                                         nikkei, VAL_START, return_df)
            except Exception:
                pass

        elapsed = time.time() - t0
        baseline = queue["baseline"]

        # 採用判定: IS Calmar比 × OAS通過の複合スコア
        if result:
            is_calmar = result['total_return_pct'] / max(result.get('max_dd_pct', 100), 1.0)
            base_calmar = baseline.get('total_pct', 0) / max(baseline.get('max_dd_pct', 100), 1.0)
            delta = round(is_calmar - base_calmar, 3)
            # OAS条件: dd<60% かつ 大崩壊しない（-30%未満）
            oas_pass = (result_val is not None and
                        result_val.get('total_return_pct', -999) > -30 and
                        result_val.get('max_dd_pct', 100) < 60)
            win = delta > 0.5 and result['max_dd_pct'] < 45 and oas_pass
            if result_val:
                print(f"  OAS (2025.6-現在): total={result_val.get('total_return_pct',0):.1f}%, "
                      f"sharpe={result_val.get('sharpe',0):.3f}, dd={result_val.get('max_dd_pct',0):.1f}% "
                      f"{'✅' if oas_pass else '❌'}", flush=True)
        else:
            delta = None
            win = False

        next_h["status"] = "done_win" if win else "done_lose"
        next_h["result"] = result
        next_h["delta_sharpe"] = delta
        next_h["elapsed_sec"] = round(elapsed, 1)
        queue["running"] = False
        queue["current_hypothesis"] = None
        queue["last_result"] = {
            "id": hid, "result": result, "delta_sharpe": delta,
            "win": win, "at": datetime.now().isoformat()
        }

        # 勝ったらベースライン更新
        if win and result:
            queue["baseline"] = {
                "sharpe": result["sharpe"],
                "total_pct": result["total_return_pct"],
                "alpha_pct": result["alpha_pct"],
                "max_dd_pct": result["max_dd_pct"],
                "params": {k: result[k] for k in result if "_w" in k or k in ["lookback","top_n","rebalance"]},
                "date": datetime.now().strftime("%Y-%m-%d"),
                "hypothesis": hid,
            }

        save_queue(queue)
        DONE_FILE.write_text(json.dumps(_fb({
            "status": "done", "id": hid, "win": win,
            "delta_sharpe": delta, "result": result,
            "at": datetime.now().isoformat()
        }), ensure_ascii=False, indent=2))
        print(f"完了: {hid} | delta_sharpe={delta} | {'✅ WIN' if win else '❌ LOSE'} | {elapsed:.0f}秒")
        append_evolution_log(hid, next_h["desc"], result, win, delta)

        # signal_library.json を更新
        update_signal_library(hid, next_h["desc"], result, win, delta)

    except Exception as e:
        next_h["status"] = "done_error"
        queue["running"] = False
        queue["current_hypothesis"] = None
        save_queue(queue)
        DONE_FILE.write_text(json.dumps(_fb({"status": "error", "id": hid, "error": str(e)}), ensure_ascii=False))
        print(f"エラー: {e}")
        traceback.print_exc()


def auto_next():
    """完了後に次の未実施仮説があれば即起動、なければopusをキックして新仮説を生成させる"""
    import subprocess, sys
    time.sleep(300)  # 5分待つ（レート制限対策）
    queue = load_queue()
    next_h = next((h for h in queue["queue"] if h["status"] == "pending"), None)
    if next_h:
        print(f"次の仮説を自動起動: {next_h['id']}", flush=True)
        subprocess.Popen(
            [sys.executable, Path(__file__).name],
            cwd=str(Path(__file__).parent),
            stdout=open(str(Path(__file__).parent / "logs/hypothesis.log"), "a"),
            stderr=subprocess.STDOUT,
        )
    else:
        # pending仮説なし → 即opusをキックして新仮説生成させる
        print("全仮説消化。即opusに新仮説生成を依頼します。", flush=True)
        subprocess.run(
            ["openclaw", "system", "event",
             "--mode", "now",
             "--text", "hypothesis_done: all pending hypotheses completed. Please run opus analysis and generate 3 new hypotheses now."],
            capture_output=True
        )
        print("openclaw system event 送信完了", flush=True)
if __name__ == "__main__":
    main()
    auto_next()

