"""高速グリッドサーチ: 全銘柄×全日付スコアをDF化して一括評価"""
import itertools, json, time
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

GRID = {
    "lookback":     [40, 60, 80],
    "top_n":        [5, 10, 20],
    "rebalance":    ["weekly", "daily"],
    "ret_w":        [0.3, 0.5, 0.7],
    "rs_w":         [0.1, 0.2, 0.3],
    "green_w":      [0.1, 0.2],
    "smooth_w":     [0.0, 0.1, 0.2],
    "resilience_w": [0.0, 0.1],
}


def precompute(prices_dict, nikkei, lookbacks):
    """全銘柄×全lookbackのファクターをrollingで事前計算。
    返り値: {lb: {factor: DataFrame(date x code)}}
    """
    t0 = time.time()
    nk_rets = nikkei.pct_change()
    down_mask = (nk_rets < -0.01).astype(float)
    
    # lb -> factor -> {code: Series}
    data = {lb: {"ret":{},"rs":{},"green":{},"smooth":{},"resilience":{},"ret5":{},"ret10":{},"omega":{},"high52":{}} for lb in lookbacks}
    
    for code, df in prices_dict.items():
        p = df["AdjC"].dropna()
        dr = p.pct_change()
        # 短期リターン（ルックバック非依存、最初に計算）
        ret5  = (p / p.shift(5) - 1).astype(float)
        ret10 = (p / p.shift(10) - 1).astype(float)
        # 52週高値proximity（ルックバック非依存）
        h52   = p.rolling(252, min_periods=60).max()
        high52 = (p / h52.replace(0, np.nan)).clip(0, 1).astype(float)
        for lb in lookbacks:
            ret    = (p / p.shift(lb) - 1).astype(float)
            nk_ret = (nikkei / nikkei.shift(lb) - 1).astype(float)
            rs     = (ret - nk_ret).astype(float)
            green  = dr.rolling(lb).apply(lambda x: float((x>0).mean()), raw=True).astype(float)
            smooth = (1.0 - dr.rolling(lb).std() * 20).clip(0, 1).astype(float)
            dm     = down_mask.reindex(dr.index, fill_value=0)
            nk_dm  = down_mask.reindex(nk_rets.index, fill_value=0)
            n_d    = dm.rolling(lb).sum().replace(0, np.nan)
            res    = ((dr*dm).rolling(lb).sum() - (nk_rets*nk_dm).rolling(lb).sum().reindex(dr.index)) / n_d
            # Omega比（上昇/下落の非対称性）
            dr_lb    = dr.rolling(lb)
            up_mean  = dr_lb.apply(lambda x: float(x[x>0].mean())  if (x>0).any() else 0.0, raw=True)
            dn_mean  = dr_lb.apply(lambda x: float(-x[x<0].mean()) if (x<0).any() else 1e-8, raw=True)
            omega    = (up_mean / dn_mean.replace(0, 1e-8)).astype(float)
            
            data[lb]["ret"][code]        = ret
            data[lb]["rs"][code]         = rs
            data[lb]["green"][code]      = green
            data[lb]["smooth"][code]     = smooth
            data[lb]["resilience"][code] = res.astype(float)
            data[lb]["ret5"][code]       = ret5
            data[lb]["ret10"][code]      = ret10
            data[lb]["omega"][code]      = omega
            data[lb]["high52"][code]     = high52
    
    # DataFrameに変換
    factor_dfs = {}
    for lb in lookbacks:
        factor_dfs[lb] = {fac: pd.DataFrame(data[lb][fac]).astype(float)
                          for fac in data[lb]}
    
    print(f"事前計算完了: {len(prices_dict)}銘柄×{len(lookbacks)}lb / {time.time()-t0:.1f}秒", flush=True)
    return factor_dfs


def build_score_df(factor_dfs, lb, weights):
    """重みを使ってスコアDF(date x code)を構築"""
    facs = factor_dfs[lb]
    score = None
    for fac, w in weights.items():
        if w == 0:
            continue
        df = facs.get(fac)
        if df is None:
            continue
        score = df * w if score is None else score + df * w
    return score.astype(float) if score is not None else None


def eval_params(params, factor_dfs, prices_dict, rebal_dates, nikkei, start, return_df):
    lb, tn = params["lookback"], params["top_n"]
    weights = {k: params[k+"_w"] for k in ["ret","rs","green","smooth","resilience"]}
    
    score_df = build_score_df(factor_dfs, lb, weights)

    # 短期モメンタムファクター（5日・10日リターンの百分位ランク平均）
    short_momentum_w = params.get('short_momentum_w', 0.0)
    if short_momentum_w > 0 and lb in factor_dfs and \
       'ret5' in factor_dfs[lb] and 'ret10' in factor_dfs[lb]:
        short_mom_df = (factor_dfs[lb]['ret5'] + factor_dfs[lb]['ret10']) / 2
        short_mom_ranked = short_mom_df.rank(axis=1, pct=True)
        if score_df is None:
            score_df = short_mom_ranked * short_momentum_w
        else:
            score_df = score_df + short_mom_ranked * short_momentum_w

    # 52週高値proximityファクター
    high52_w = params.get('high52_w', 0.0)
    if high52_w > 0 and lb in factor_dfs and 'high52' in factor_dfs[lb]:
        high52_ranked = factor_dfs[lb]['high52'].rank(axis=1, pct=True)
        if score_df is None:
            score_df = high52_ranked * high52_w
        else:
            score_df = score_df + high52_ranked * high52_w

    # Omega比ファクター（上昇/下落の非対称性）
    omega_w = params.get('omega_w', 0.0)
    if omega_w > 0 and lb in factor_dfs and 'omega' in factor_dfs[lb]:
        omega_ranked = factor_dfs[lb]['omega'].rank(axis=1, pct=True)
        if score_df is None:
            score_df = omega_ranked * omega_w
        else:
            score_df = score_df + omega_ranked * omega_w

    if score_df is None:
        return None
    
    dates = [d for d in rebal_dates if str(d.date()) >= start]
    
    portfolio = 1_000_000.0
    returns = []
    
    for i, date in enumerate(dates[:-1]):
        next_date = dates[i+1]
        if date not in score_df.index:
            continue
        
        row = score_df.loc[date].dropna()
        if row.empty:
            continue
        top = row.nlargest(tn).index.tolist()
        
        # リターン計算
        tot, cnt = 0.0, 0
        if date in return_df.index and next_date in return_df.index:
            for code in top:
                if code in return_df.columns:
                    r = return_df.at[next_date, code]
                    if not np.isnan(r):
                        tot += r; cnt += 1
        if cnt > 0:
            r = tot / cnt
            portfolio *= (1+r)
            returns.append(r)
    
    if len(returns) < 5:
        return None
    
    arr = np.array(returns)
    tr = portfolio/1_000_000 - 1
    sharpe = float(arr.mean()/arr.std()*np.sqrt(252)) if arr.std()>0 else 0
    cum = np.cumprod(1+arr); peak = np.maximum.accumulate(cum)
    dd = float(abs(((cum-peak)/peak).min()))
    nk = nikkei.loc[start:]; nk_ret = float(nk.iloc[-1]/nk.iloc[0]-1) if len(nk)>1 else 0
    
    return {**params, "total_return_pct": round(tr*100,2),
            "alpha_pct": round((tr-nk_ret)*100,2), "sharpe": round(sharpe,3),
            "max_dd_pct": round(dd*100,2), "nikkei_pct": round(nk_ret*100,2),
            "n_trades": len(returns)}


def run_grid(start="2023-01-01", end="2026-03-05", n_codes=2000):
    t0 = time.time()
    print(f"データ読み込み ({n_codes}銘柄)...", flush=True)
    codes = get_top_liquid_tickers(n_codes)
    warmup = (datetime.strptime(start,"%Y-%m-%d")-timedelta(days=200)).strftime("%Y-%m-%d")
    prices_dict = {}
    for c in codes:
        df = read_ohlcv(c, warmup, end)
        if df is not None and not df.empty and "AdjC" in df.columns:
            prices_dict[c] = df
    print(f"  {len(prices_dict)}銘柄ロード完了", flush=True)
    
    nikkei = get_nikkei_history(warmup, end)
    factor_dfs = precompute(prices_dict, nikkei, GRID["lookback"])
    
    # 日次リターンDF(date x code)を事前構築
    print("リターンDF構築...", flush=True)
    all_prices = pd.DataFrame({c: prices_dict[c]["AdjC"] for c in prices_dict}).astype(float)
    return_df = all_prices.pct_change()
    
    daily_d  = get_rebalance_dates(warmup, end, "daily")
    weekly_d = get_rebalance_dates(warmup, end, "weekly")
    date_map = {"daily": daily_d, "weekly": weekly_d}
    
    all_params = list(itertools.product(
        GRID["lookback"],GRID["top_n"],GRID["rebalance"],
        GRID["ret_w"],GRID["rs_w"],GRID["green_w"],GRID["smooth_w"],GRID["resilience_w"]))
    print(f"\n{len(all_params)}パターン評価中...", flush=True)
    
    results = []
    t1 = time.time()
    for i,(lb,tn,rb,ret_w,rs_w,gr_w,sm_w,res_w) in enumerate(all_params):
        p = {"lookback":lb,"top_n":tn,"rebalance":rb,"ret_w":ret_w,"rs_w":rs_w,
             "green_w":gr_w,"smooth_w":sm_w,"resilience_w":res_w}
        r = eval_params(p, factor_dfs, prices_dict, date_map[rb], nikkei, start, return_df)
        if r: results.append(r)
        if (i+1)%200==0:
            print(f"  {i+1}/{len(all_params)} ({time.time()-t1:.0f}秒)", flush=True)
    
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    nk_pct = results[0]["nikkei_pct"] if results else 0
    
    print(f"\n=== TOP 10（シャープ順）| 日経: {nk_pct:+.1f}% ===")
    for i,r in enumerate(results[:10],1):
        print(f"{i:2}. sharpe={r['sharpe']:+.3f} total={r['total_return_pct']:+.1f}% "
              f"alpha={r['alpha_pct']:+.1f}% dd={r['max_dd_pct']:.1f}% "
              f"lb={r['lookback']} tn={r['top_n']} {r['rebalance'][:1]} "
              f"ret={r['ret_w']} rs={r['rs_w']} gr={r['green_w']} sm={r['smooth_w']} res={r['resilience_w']}")
    
    out = {"run_at":datetime.now().isoformat(),"start":start,"end":end,
           "total_tested":len(results),"nikkei_pct":nk_pct,"top10":results[:10],"all":results[:300]}
    Path("backtest").mkdir(exist_ok=True)
    Path("backtest/optimize_latest.json").write_text(json.dumps(_fb(out),ensure_ascii=False,indent=2))
    print(f"\n完了 ({time.time()-t0:.0f}秒) → backtest/optimize_latest.json")
    return results


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    run_grid(start=args.start, end=args.end, n_codes=args.n)
