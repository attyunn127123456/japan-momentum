"""
仮説を1つ実行してbacktest/hypothesis_queue.jsonを更新するスクリプト。
heartbeatからバックグラウンドで呼ばれる。
完了後 backtest/hypothesis_done.json を書く。
"""
import json, time, sys, traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from fetch_cache import read_ohlcv
from universe import get_top_liquid_tickers
from backtest import get_rebalance_dates, get_nikkei_history
from optimize import precompute, eval_params

QUEUE_FILE = Path("backtest/hypothesis_queue.json")
DONE_FILE  = Path("backtest/hypothesis_done.json")
START = "2023-01-01"
END   = datetime.now().strftime("%Y-%m-%d")
N_CODES = 200


def load_queue():
    return json.loads(QUEUE_FILE.read_text())


def save_queue(q):
    QUEUE_FILE.write_text(json.dumps(q, ensure_ascii=False, indent=2))


def run_baseline(prices_dict, nikkei, factor_dfs, return_df, rebal_dates):
    """ベースラインのスコアを返す"""
    p = {"lookback":60,"top_n":5,"rebalance":"weekly",
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
        # 価格フィルタ: 中型株近似（500〜5000円帯 = 中型が多い）
        avg_price = df["AdjC"].tail(20).mean()
        if 200 <= avg_price <= 10000:
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
    for top_n in [5, 10]:
        p = {"lookback":60,"top_n":top_n,"rebalance":"weekly",
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


def main():
    queue = load_queue()

    # 次の未実施仮説を取得
    next_h = next((h for h in queue["queue"] if h["status"] == "pending"), None)
    if not next_h:
        DONE_FILE.write_text(json.dumps({"status":"all_done","at":datetime.now().isoformat()}, ensure_ascii=False))
        print("全仮説完了")
        return

    hid = next_h["id"]
    print(f"仮説実行: {hid} - {next_h['desc']}")
    queue["running"] = True
    queue["current_hypothesis"] = hid
    next_h["status"] = "running"
    save_queue(queue)

    warmup = (datetime.strptime(START,"%Y-%m-%d")-timedelta(days=200)).strftime("%Y-%m-%d")
    nikkei = get_nikkei_history(warmup, END)
    rebal_dates = {
        "weekly": get_rebalance_dates(warmup, END, "weekly"),
        "daily":  get_rebalance_dates(warmup, END, "daily"),
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
            p = {"lookback":60,"top_n":5,"rebalance":"weekly",
                 "ret_w":0.3,"rs_w":0.3,"green_w":0.2,"smooth_w":0.2,"resilience_w":0.0}
            result = eval_params(p, factor_dfs, prices_dict, rebal_dates["weekly"], nikkei, START, return_df)

        elif hid == "resilience_upgrade":
            codes = get_top_liquid_tickers(N_CODES)
            prices_dict = {}
            for c in codes:
                df = read_ohlcv(c, warmup, END)
                if df is not None and not df.empty and "AdjC" in df.columns:
                    prices_dict[c] = df
            factor_dfs = precompute(prices_dict, nikkei, [60])
            all_prices = pd.DataFrame({c: prices_dict[c]["AdjC"] for c in prices_dict}).astype(float)
            return_df = all_prices.pct_change()
            p = {"lookback":60,"top_n":5,"rebalance":"weekly",
                 "ret_w":0.2,"rs_w":0.2,"green_w":0.1,"smooth_w":0.2,"resilience_w":0.3}
            result = eval_params(p, factor_dfs, prices_dict, rebal_dates["weekly"], nikkei, START, return_df)

        else:
            result = None

        elapsed = time.time() - t0
        baseline = queue["baseline"]
        delta = round((result["sharpe"] - baseline["sharpe"]), 3) if result else None
        win = delta is not None and delta > 0.05

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
        DONE_FILE.write_text(json.dumps({
            "status": "done", "id": hid, "win": win,
            "delta_sharpe": delta, "result": result,
            "at": datetime.now().isoformat()
        }, ensure_ascii=False, indent=2))
        print(f"完了: {hid} | delta_sharpe={delta} | {'✅ WIN' if win else '❌ LOSE'} | {elapsed:.0f}秒")

    except Exception as e:
        next_h["status"] = "pending"  # リトライ可能に
        queue["running"] = False
        save_queue(queue)
        DONE_FILE.write_text(json.dumps({"status":"error","id":hid,"error":str(e)}, ensure_ascii=False))
        print(f"エラー: {e}")
        traceback.print_exc()


def auto_next():
    """完了後に次の未実施仮説があれば自分自身を再起動"""
    import subprocess, sys
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
        print("全仮説完了！heartbeatで分析・新仮説生成を行います。", flush=True)
if __name__ == "__main__":
    main()
    auto_next()

