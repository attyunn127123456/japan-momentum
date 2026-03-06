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
    "lookback":     [20, 30, 45, 60, 80],
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
    data = {lb: {"ret":{},"rs":{},"green":{},"smooth":{},"resilience":{},"ret5":{},"ret10":{},"omega":{},"high52":{},"close_location":{},"range_expand":{},"win_streak":{},"sector_momentum":{},"overnight_return":{},"volume_acceleration":{},"higher_lows":{},"body_strength":{},"vol_return_corr":{},"accumulation":{},"momentum_consistency":{},"upside_capture":{},"gap_momentum":{},"volume_confirm":{},"ret_skip":{},"cluster_boost":{},"return_autocorr":{},"volume_slope":{},"clean_momentum":{}} for lb in lookbacks}

    # 業種コードマップを読み込む
    try:
        master = pd.read_parquet('data/fundamentals/equities_master.parquet')
        sector_map = master.set_index('Code')['Sector33Code'].to_dict()
    except Exception:
        sector_map = {}

    def max_streak(x):
        streak = max_s = 0
        for v in x:
            if v > 0:
                streak += 1
                max_s = max(max_s, streak)
            else:
                streak = 0
        return float(max_s)

    for code, df in prices_dict.items():
        p = df["AdjC"].dropna()
        dr = p.pct_change()
        # 短期リターン（ルックバック非依存、最初に計算）
        ret5  = (p / p.shift(5) - 1).astype(float)
        ret10 = (p / p.shift(10) - 1).astype(float)
        # 52週高値proximity（ルックバック非依存）
        h52   = p.rolling(252, min_periods=60).max()
        high52 = (p / h52.replace(0, np.nan)).clip(0, 1).astype(float)
        # 新ファクター（ルックバック非依存部分）
        high = df["High"].reindex(p.index).ffill() if "High" in df.columns else pd.Series(np.nan, index=p.index)
        low  = df["Low"].reindex(p.index).ffill()  if "Low"  in df.columns else pd.Series(np.nan, index=p.index)
        rng  = (high - low).replace(0, np.nan)
        atr5  = (high - low).rolling(5).mean()
        atr40 = (high - low).rolling(40).mean().replace(0, np.nan)
        range_expand = (atr5 / atr40).astype(float)
        win_streak = dr.rolling(20).apply(max_streak, raw=True).astype(float)
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
            # 終値位置ファクター（lb依存: rolling(lb)）
            close_loc = ((p - low) / rng).rolling(lb).mean().astype(float)
            data[lb]["close_location"][code] = close_loc
            # ATRブレイクアウト・連続陽線（lb非依存だが全lbに保存）
            data[lb]["range_expand"][code] = range_expand
            data[lb]["win_streak"][code]   = win_streak
            # 業種モメンタム（後でまとめて計算）
            data[lb]["sector_momentum"][code] = pd.Series(np.nan, index=p.index)
            # 夜間リターン（close→翌open）
            if "Open" in df.columns:
                op = df["Open"].reindex(p.index).ffill()
                overnight = (op / p.shift(1) - 1)
                overnight_cum = overnight.rolling(lb).sum().astype(float)
                data[lb]["overnight_return"][code] = overnight_cum
            else:
                data[lb]["overnight_return"][code] = pd.Series(np.nan, index=p.index)
            # 出来高加速度（4週間トレンドの傾き）
            if "Volume" in df.columns:
                vol = df["Volume"].reindex(p.index).replace(0, np.nan)
                vol_norm = vol / vol.rolling(60, min_periods=20).mean()
                vol_accel = vol_norm.rolling(20).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan, raw=True
                ).astype(float)
                data[lb]["volume_acceleration"][code] = vol_accel
            else:
                data[lb]["volume_acceleration"][code] = pd.Series(np.nan, index=p.index)

            # --- phantom weight解消: 未実装ファクター追加 ---

            # higher_lows: 下値切り上げ（lb依存）
            low_series = df["Low"].reindex(p.index).ffill() if "Low" in df.columns else p
            rolling_low_min = low_series.rolling(lb).min()
            higher_lows = rolling_low_min.diff(5).apply(lambda x: 1.0 if x > 0 else 0.0).rolling(10).mean().astype(float)
            data[lb]["higher_lows"][code] = higher_lows

            # body_strength: 実体強度（lb依存）
            if "Open" in df.columns and "High" in df.columns and "Low" in df.columns:
                op = df["Open"].reindex(p.index).ffill()
                rng_bs = (df["High"].reindex(p.index).ffill() - df["Low"].reindex(p.index).ffill()).replace(0, np.nan)
                body = (p - op) / rng_bs
                data[lb]["body_strength"][code] = body.rolling(lb).mean().astype(float)
            else:
                data[lb]["body_strength"][code] = pd.Series(np.nan, index=p.index)

            # vol_return_corr: 出来高リターン相関（lb依存）
            if "Volume" in df.columns:
                vol_vrc = df["Volume"].reindex(p.index).replace(0, np.nan)
                vol_chg = vol_vrc.pct_change()
                corr_series = dr.rolling(lb).corr(vol_chg).astype(float)
                data[lb]["vol_return_corr"][code] = corr_series
            else:
                data[lb]["vol_return_corr"][code] = pd.Series(np.nan, index=p.index)

            # accumulation: 方向性出来高（lb依存）
            if "Volume" in df.columns:
                vol_acc = df["Volume"].reindex(p.index).replace(0, np.nan)
                up_vol = vol_acc.where(dr > 0, 0)
                down_vol = vol_acc.where(dr < 0, 0)
                accum = (up_vol.rolling(lb).sum() / (down_vol.rolling(lb).sum().replace(0, np.nan))).astype(float)
                data[lb]["accumulation"][code] = accum
            else:
                data[lb]["accumulation"][code] = pd.Series(np.nan, index=p.index)

            # momentum_consistency: 週次勝率（lb依存）
            weekly_win = dr.rolling(5).sum().apply(lambda x: 1.0 if x > 0 else 0.0)
            data[lb]["momentum_consistency"][code] = weekly_win.rolling(max(lb // 5, 1)).mean().astype(float)

            # upside_capture: 上方キャプチャ（lb依存）
            up_days = nk_rets > 0
            stock_ret_on_up = dr.where(up_days)
            nk_ret_on_up = nk_rets.reindex(dr.index).where(up_days.reindex(dr.index))
            capture = (stock_ret_on_up.rolling(lb).mean() / nk_ret_on_up.rolling(lb).mean().replace(0, np.nan)).astype(float)
            data[lb]["upside_capture"][code] = capture

            # gap_momentum: 寄り付きギャップ累積（lb依存）
            if "Open" in df.columns:
                op_gm = df["Open"].reindex(p.index).ffill()
                gap = (op_gm / p.shift(1) - 1)
                data[lb]["gap_momentum"][code] = gap.rolling(lb).mean().astype(float)
            else:
                data[lb]["gap_momentum"][code] = pd.Series(np.nan, index=p.index)

            # volume_confirm: 出来高確認モメンタム（lb依存）
            if "Volume" in df.columns:
                vol_vc = df["Volume"].reindex(p.index).replace(0, np.nan)
                vol_avg_vc = vol_vc.rolling(lb).mean()
                vol_confirm = (dr * (vol_vc / vol_avg_vc.replace(0, np.nan))).rolling(lb).mean().astype(float)
                data[lb]["volume_confirm"][code] = vol_confirm
            else:
                data[lb]["volume_confirm"][code] = pd.Series(np.nan, index=p.index)

            # ret_skip: スキップ期間モメンタム (lb+21日前から21日前)
            ret_skip = (p.shift(21) / p.shift(lb + 21) - 1).astype(float)
            data[lb]["ret_skip"][code] = ret_skip

            # cluster_boost: マルチタイムフレームモメンタム一致度（lb依存）
            mom5  = dr.rolling(5).mean()
            mom10 = dr.rolling(10).mean()
            mom20 = dr.rolling(20).mean()
            cluster = ((mom5.gt(0).astype(float) + mom10.gt(0).astype(float) + mom20.gt(0).astype(float)) / 3.0)
            data[lb]["cluster_boost"][code] = cluster.rolling(lb).mean().astype(float)

            # return_autocorr: リターン自己相関（lag-1）
            autocorr = dr.rolling(lb).apply(
                lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 10 else np.nan,
                raw=True
            ).astype(float)
            data[lb]["return_autocorr"][code] = autocorr

            # volume_slope: 出来高トレンド（対数出来高のrolling回帰傾き）
            if "Volume" in df.columns:
                vol_vs = df["Volume"].reindex(p.index).replace(0, np.nan)
                log_vol = np.log(vol_vs.clip(lower=1))
                vol_slope = log_vol.rolling(lb).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 10 else np.nan,
                    raw=True
                ).astype(float)
                data[lb]["volume_slope"][code] = vol_slope
            else:
                data[lb]["volume_slope"][code] = pd.Series(np.nan, index=p.index)

            # clean_momentum: ピークからの乖離が10%以内の期間のみのリターン累積
            running_max = p.expanding().max()
            dd_from_peak = (p / running_max - 1)
            in_uptrend = (dd_from_peak >= -0.10).astype(float)
            clean_ret = (dr * in_uptrend).rolling(lb).sum().astype(float)
            data[lb]["clean_momentum"][code] = clean_ret

    # 業種モメンタムを一括計算（全銘柄のretが揃ってから）
    for lb in lookbacks:
        ret_df = pd.DataFrame(data[lb]["ret"])  # date x code
        sector_ret = {}
        for code in ret_df.columns:
            sc = sector_map.get(str(code), sector_map.get(int(code) if str(code).isdigit() else code, None))
            if sc is not None:
                sector_ret.setdefault(sc, []).append(ret_df[code])
        sector_avg = {sc: pd.concat(sers, axis=1).mean(axis=1) for sc, sers in sector_ret.items()}
        for code in ret_df.columns:
            sc = sector_map.get(str(code), sector_map.get(int(code) if str(code).isdigit() else code, None))
            if sc in sector_avg:
                data[lb]["sector_momentum"][code] = sector_avg[sc].astype(float)

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
    weights = {k: params.get(k+"_w", 0.0) for k in ["ret","rs","green","smooth","resilience"]}
    
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

    # 終値位置ファクター（引け買い圧力）
    close_location_w = params.get('close_location_w', 0.0)
    if close_location_w > 0 and lb in factor_dfs and 'close_location' in factor_dfs[lb]:
        cl_ranked = factor_dfs[lb]['close_location'].rank(axis=1, pct=True)
        if score_df is None:
            score_df = cl_ranked * close_location_w
        else:
            score_df = score_df + cl_ranked * close_location_w

    # ATRブレイクアウト検出ファクター
    range_expand_w = params.get('range_expand_w', 0.0)
    if range_expand_w > 0 and lb in factor_dfs and 'range_expand' in factor_dfs[lb]:
        re_ranked = factor_dfs[lb]['range_expand'].rank(axis=1, pct=True)
        if score_df is None:
            score_df = re_ranked * range_expand_w
        else:
            score_df = score_df + re_ranked * range_expand_w

    # 最大連続陽線日数ファクター
    win_streak_w = params.get('win_streak_w', 0.0)
    if win_streak_w > 0 and lb in factor_dfs and 'win_streak' in factor_dfs[lb]:
        ws_ranked = factor_dfs[lb]['win_streak'].rank(axis=1, pct=True)
        if score_df is None:
            score_df = ws_ranked * win_streak_w
        else:
            score_df = score_df + ws_ranked * win_streak_w

    # 業種モメンタムファクター
    sector_momentum_w = params.get('sector_momentum_w', 0.0)
    if sector_momentum_w > 0 and lb in factor_dfs and 'sector_momentum' in factor_dfs[lb]:
        sm_ranked = factor_dfs[lb]['sector_momentum'].rank(axis=1, pct=True)
        if score_df is None:
            score_df = sm_ranked * sector_momentum_w
        else:
            score_df = score_df + sm_ranked * sector_momentum_w

    # 夜間リターンファクター
    overnight_return_w = params.get('overnight_return_w', 0.0)
    if overnight_return_w > 0 and lb in factor_dfs and 'overnight_return' in factor_dfs[lb]:
        or_ranked = factor_dfs[lb]['overnight_return'].rank(axis=1, pct=True)
        if score_df is None:
            score_df = or_ranked * overnight_return_w
        else:
            score_df = score_df + or_ranked * overnight_return_w

    # 出来高加速度ファクター
    volume_acceleration_w = params.get('volume_acceleration_w', 0.0)
    if volume_acceleration_w > 0 and lb in factor_dfs and 'volume_acceleration' in factor_dfs[lb]:
        va_ranked = factor_dfs[lb]['volume_acceleration'].rank(axis=1, pct=True)
        if score_df is None:
            score_df = va_ranked * volume_acceleration_w
        else:
            score_df = score_df + va_ranked * volume_acceleration_w

    # 下値切り上げファクター
    higher_lows_w = params.get('higher_lows_w', 0.0)
    if higher_lows_w > 0 and lb in factor_dfs and 'higher_lows' in factor_dfs[lb]:
        hl_ranked = factor_dfs[lb]['higher_lows'].rank(axis=1, pct=True)
        score_df = hl_ranked * higher_lows_w if score_df is None else score_df + hl_ranked * higher_lows_w

    # 実体強度ファクター
    body_strength_w = params.get('body_strength_w', 0.0)
    if body_strength_w > 0 and lb in factor_dfs and 'body_strength' in factor_dfs[lb]:
        bs_ranked = factor_dfs[lb]['body_strength'].rank(axis=1, pct=True)
        score_df = bs_ranked * body_strength_w if score_df is None else score_df + bs_ranked * body_strength_w

    # 出来高リターン相関ファクター
    vol_return_corr_w = params.get('vol_return_corr_w', 0.0)
    if vol_return_corr_w > 0 and lb in factor_dfs and 'vol_return_corr' in factor_dfs[lb]:
        vrc_ranked = factor_dfs[lb]['vol_return_corr'].rank(axis=1, pct=True)
        score_df = vrc_ranked * vol_return_corr_w if score_df is None else score_df + vrc_ranked * vol_return_corr_w

    # 方向性出来高ファクター
    accumulation_w = params.get('accumulation_w', 0.0)
    if accumulation_w > 0 and lb in factor_dfs and 'accumulation' in factor_dfs[lb]:
        acc_ranked = factor_dfs[lb]['accumulation'].rank(axis=1, pct=True)
        score_df = acc_ranked * accumulation_w if score_df is None else score_df + acc_ranked * accumulation_w

    # 週次勝率ファクター
    momentum_consistency_w = params.get('momentum_consistency_w', 0.0)
    if momentum_consistency_w > 0 and lb in factor_dfs and 'momentum_consistency' in factor_dfs[lb]:
        mc_ranked = factor_dfs[lb]['momentum_consistency'].rank(axis=1, pct=True)
        score_df = mc_ranked * momentum_consistency_w if score_df is None else score_df + mc_ranked * momentum_consistency_w

    # 上方キャプチャファクター
    upside_capture_w = params.get('upside_capture_w', 0.0)
    if upside_capture_w > 0 and lb in factor_dfs and 'upside_capture' in factor_dfs[lb]:
        uc_ranked = factor_dfs[lb]['upside_capture'].rank(axis=1, pct=True)
        score_df = uc_ranked * upside_capture_w if score_df is None else score_df + uc_ranked * upside_capture_w

    # 寄り付きギャップモメンタムファクター
    gap_momentum_w = params.get('gap_momentum_w', 0.0)
    if gap_momentum_w > 0 and lb in factor_dfs and 'gap_momentum' in factor_dfs[lb]:
        gm_ranked = factor_dfs[lb]['gap_momentum'].rank(axis=1, pct=True)
        score_df = gm_ranked * gap_momentum_w if score_df is None else score_df + gm_ranked * gap_momentum_w

    # 出来高確認モメンタムファクター
    volume_confirm_w = params.get('volume_confirm_w', 0.0)
    if volume_confirm_w > 0 and lb in factor_dfs and 'volume_confirm' in factor_dfs[lb]:
        vc_ranked = factor_dfs[lb]['volume_confirm'].rank(axis=1, pct=True)
        score_df = vc_ranked * volume_confirm_w if score_df is None else score_df + vc_ranked * volume_confirm_w

    # スキップ期間モメンタムファクター
    ret_skip_w = params.get('ret_skip_w', 0.0)
    if ret_skip_w > 0 and lb in factor_dfs and 'ret_skip' in factor_dfs[lb]:
        rs_ranked = factor_dfs[lb]['ret_skip'].rank(axis=1, pct=True)
        score_df = rs_ranked * ret_skip_w if score_df is None else score_df + rs_ranked * ret_skip_w

    # マルチTFモメンタム一致度ファクター
    cluster_boost_w = params.get('cluster_boost_w', 0.0)
    if cluster_boost_w > 0 and lb in factor_dfs and 'cluster_boost' in factor_dfs[lb]:
        cb_ranked = factor_dfs[lb]['cluster_boost'].rank(axis=1, pct=True)
        score_df = cb_ranked * cluster_boost_w if score_df is None else score_df + cb_ranked * cluster_boost_w

    # リターン自己相関ファクター
    return_autocorr_w = params.get('return_autocorr_w', 0.0)
    if return_autocorr_w > 0 and lb in factor_dfs and 'return_autocorr' in factor_dfs[lb]:
        ra_ranked = factor_dfs[lb]['return_autocorr'].rank(axis=1, pct=True)
        score_df = ra_ranked * return_autocorr_w if score_df is None else score_df + ra_ranked * return_autocorr_w

    # 出来高トレンドファクター
    volume_slope_w = params.get('volume_slope_w', 0.0)
    if volume_slope_w > 0 and lb in factor_dfs and 'volume_slope' in factor_dfs[lb]:
        vs_ranked = factor_dfs[lb]['volume_slope'].rank(axis=1, pct=True)
        score_df = vs_ranked * volume_slope_w if score_df is None else score_df + vs_ranked * volume_slope_w

    # クリーンモメンタムファクター
    clean_momentum_w = params.get('clean_momentum_w', 0.0)
    if clean_momentum_w > 0 and lb in factor_dfs and 'clean_momentum' in factor_dfs[lb]:
        cm_ranked = factor_dfs[lb]['clean_momentum'].rank(axis=1, pct=True)
        score_df = cm_ranked * clean_momentum_w if score_df is None else score_df + cm_ranked * clean_momentum_w

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

    # 週次時系列データを収集（dashboard用）
    equity_curve = []
    portfolio2 = 1_000_000.0
    for i, date in enumerate(dates[:-1]):
        next_date = dates[i+1]
        if date not in score_df.index:
            continue
        row = score_df.loc[date].dropna()
        if row.empty:
            continue
        top = row.nlargest(tn).index.tolist()
        tot, cnt = 0.0, 0
        if date in return_df.index and next_date in return_df.index:
            for code in top:
                if code in return_df.columns:
                    r = return_df.at[next_date, code]
                    if not np.isnan(r):
                        tot += r; cnt += 1
        if cnt > 0:
            portfolio2 *= (1 + tot/cnt)
        equity_curve.append({
            "date": str(date.date()),
            "value": round((portfolio2/1_000_000 - 1) * 100, 2),
            "holdings": [str(c) for c in top]
        })

    return {**params, "total_return_pct": round(tr*100,2),
            "alpha_pct": round((tr-nk_ret)*100,2), "sharpe": round(sharpe,3),
            "max_dd_pct": round(dd*100,2), "nikkei_pct": round(nk_ret*100,2),
            "n_trades": len(returns), "equity_curve": equity_curve}


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


def select_independent_factors(factor_dfs, lb, all_factors, corr_threshold=0.7):
    """相関0.7以上のファクターペアから弱い方を除外。
    all_factors は '_w' サフィックス付きのパラメータ名でも素の名前でも可。
    factor_dfs のキーは '_w' なし（例: 'ret', 'rs', 'high52'）なので自動マッピングする。
    """
    # パラメータ名 → factor_dfs キー のマッピング
    PARAM_TO_KEY = {
        'ret_w': 'ret',
        'rs_w': 'rs',
        'green_w': 'green',
        'smooth_w': 'smooth',
        'resilience_w': 'resilience',
        'high52_w': 'high52',
        'omega_w': 'omega',
        'short_momentum_w': 'ret5',   # ret5 + ret10 の代表として ret5
        'close_location_w': 'close_location',
        'range_expand_w': 'range_expand',
        'win_streak_w': 'win_streak',
        'sector_momentum_w': 'sector_momentum',
        'overnight_return_w': 'overnight_return',
        'volume_acceleration_w': 'volume_acceleration',
    }

    fac_lb = factor_dfs.get(lb, {})
    factor_series = {}
    for fname in all_factors:
        # '_w' 付きパラメータ名からキーを解決
        fkey = PARAM_TO_KEY.get(fname, fname.rstrip('_w') if fname.endswith('_w') else fname)
        if fkey in fac_lb:
            s = fac_lb[fkey].stack().dropna()
            if len(s) > 100:
                factor_series[fname] = s   # キーはパラメータ名のまま保持

    if len(factor_series) < 2:
        return list(factor_series.keys())

    df = pd.DataFrame(factor_series).dropna()
    if len(df) > 1000:
        df = df.sample(1000, random_state=42)
    corr = df.corr().abs()

    # 除外ロジック：相関>threshold のペアで後に来る方を除外
    selected = list(factor_series.keys())
    to_remove = set()
    for i, f1 in enumerate(selected):
        for f2 in selected[i+1:]:
            if f1 in corr.index and f2 in corr.index:
                if corr.loc[f1, f2] > corr_threshold:
                    to_remove.add(f2)  # f2を除外（f1優先）

    result = [f for f in selected if f not in to_remove]
    print(f"Factor Selection: {len(selected)}→{len(result)}個 (除外: {to_remove})")
    return result


def run_optuna_optimization(baseline_params, factor_dfs, prices_dict, nikkei,
                            date_map, start, return_df, n_trials=300):
    """Optuna TPEでパラメータ最適化。GAの local_search を置き換える。
    Returns: [(params, result), ...] 形式（local_search と同じ形式）
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    all_weight_factors = [
        'ret_w', 'rs_w', 'green_w', 'smooth_w', 'resilience_w',
        'high52_w', 'omega_w', 'short_momentum_w', 'cluster_boost_w',
        'ret_skip_w', 'volume_confirm_w', 'gap_momentum_w',
        'close_location_w', 'range_expand_w', 'win_streak_w',
        'accumulation_w', 'momentum_consistency_w', 'upside_capture_w',
        'return_autocorr_w', 'volume_slope_w', 'clean_momentum_w',
    ]

    base_lb = baseline_params.get('lookback', 60)
    available_lbs = [lb for lb in [20, 40, 60, 80, 100, 120] if lb in factor_dfs]
    if not available_lbs:
        available_lbs = [base_lb]

    # Factor Selection: factor_dfsに実際に存在するもの + eval_paramsが知ってるもの
    factor_lb = base_lb if base_lb in factor_dfs else available_lbs[0]
    # factor_dfsに存在するキー + alias経由で使えるキー
    existing_factors = list(factor_dfs.get(factor_lb, {}).keys())
    # eval_paramsがaliasで対応する重みキーも含める
    alias_supported = ['short_momentum_w','cluster_boost_w','ret_skip_w',
                       'volume_confirm_w','gap_momentum_w','omega_w',
                       'accumulation_w','momentum_consistency_w','upside_capture_w',
                       'return_autocorr_w','volume_slope_w','clean_momentum_w']
    # factor_dfsにあるものはcorr計算でフィルタ、aliasはそのまま通す
    fdf_factors = [f[:-2] if f.endswith("_w") else f for f in all_weight_factors
                   if f[:-2] in existing_factors]
    fdf_active = select_independent_factors(factor_dfs, factor_lb,
                                            [f for f in all_weight_factors if f[:-2] in existing_factors])
    # aliasは常に含める
    alias_active = [f for f in alias_supported if f in all_weight_factors]
    active_factors = list(dict.fromkeys(fdf_active + alias_active))  # 重複除去
    if not active_factors:
        active_factors = all_weight_factors  # フォールバック
    print(f"Optuna探索対象ファクター: {active_factors}", flush=True)
    print(f"Optuna TPE: {n_trials}trials 開始...", flush=True)

    all_results = []
    best_score = -999.0

    def objective(trial):
        nonlocal best_score

        # 構造パラメータ
        trial_lb = trial.suggest_categorical('lookback', available_lbs)
        top_n = trial.suggest_int('top_n', 1, 7)

        # ファクター重み（0〜0.5の連続値）
        weights = {}
        for f in active_factors:
            weights[f] = trial.suggest_float(f, 0.0, 0.5)

        # 重みを正規化（合計1.0）
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}
        else:
            return -999.0

        # eval_params が必ず参照するコア5ファクターが欠けている場合は 0.0 で補完
        for core_k in ['ret_w', 'rs_w', 'green_w', 'smooth_w', 'resilience_w']:
            if core_k not in weights:
                weights[core_k] = 0.0

        trial_rb = trial.suggest_categorical('rebalance', ['weekly', 'monthly', 'daily'])
        params = {
            'lookback': trial_lb,
            'top_n': top_n,
            'rebalance': trial_rb,
            **weights,
        }

        try:
            r = eval_params(params, factor_dfs, prices_dict,
                            date_map[trial_rb], nikkei, start, return_df)
            if r is None:
                return -999.0
            if r.get('max_dd_pct', 100) > 40:
                return -999.0  # 制約違反ペナルティ
            score = r.get('total_return_pct', -999.0)
            all_results.append((params, r))

            # ベスト更新時にログ
            if score > best_score:
                best_score = score
                print(f"  新ベスト: total={score:.1f}%, sharpe={r.get('sharpe', 0):.3f}, "
                      f"lb={trial_lb}, tn={top_n}", flush=True)

            return float(score)
        except Exception:
            return -999.0

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=50),
    )

    # 現在のベストパラメータを初期試行として追加
    try:
        init_params = {
            'lookback': baseline_params.get('lookback', available_lbs[0]),
            'top_n': baseline_params.get('top_n', 3),
        }
        for f in active_factors:
            init_params[f] = baseline_params.get(f, 0.0)
        study.enqueue_trial(init_params)
    except Exception:
        pass

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_val = study.best_value if study.trials else -999.0
    print(f"Optuna完了: {n_trials}trials, best_total={best_val:.1f}%", flush=True)

    # total_return_pct 降順でソート、上位20件を返す（local_search と同形式）
    all_results.sort(key=lambda x: x[1].get('total_return_pct', -999), reverse=True)
    return all_results[:20]


def run_oos_validation(best_params, oos_start="2020-01-01", oos_end=None, n_codes=2000, codes=None):
    """
    Optunaで見つけたベストパラメータを全期間データで検証（過学習チェック用）。
    OOS期間のデータが不足していれば自動バックフィルを試みる。

    Args:
        codes: 既にロード済みの銘柄リスト（省略時はAPIで取得）
    Returns: dict (eval_params 結果) or None
    """
    if oos_end is None:
        oos_end = datetime.now().strftime("%Y-%m-%d")

    print(f"\n=== OOS検証: {oos_start}〜{oos_end} ===", flush=True)

    if codes is None:
        codes = get_top_liquid_tickers(n_codes)

    # バックフィルが必要か確認
    from fetch_cache import get_first_date, backfill_cache
    needs_backfill = []
    for c in codes[:200]:  # サンプルチェック（全銘柄は重いので代表）
        first = get_first_date(c)
        if first is None or first > oos_start:
            needs_backfill.append(c)
    
    if needs_backfill:
        print(f"バックフィル対象: {len(needs_backfill)}件（レート制限回避のためスキップ）", flush=True)
    else:
        print(f"キャッシュに十分な履歴データあり", flush=True)

    warmup = (datetime.strptime(oos_start, "%Y-%m-%d") - timedelta(days=200)).strftime("%Y-%m-%d")
    prices_dict = {}
    for c in codes:
        df = read_ohlcv(c, warmup, oos_end)
        if df is not None and not df.empty and "AdjC" in df.columns:
            prices_dict[c] = df

    if not prices_dict:
        print("OOS検証: データなし", flush=True)
        return None

    print(f"OOS検証: {len(prices_dict)}銘柄ロード完了", flush=True)

    nikkei = get_nikkei_history(warmup, oos_end)
    lb = best_params.get("lookback", 60)
    factor_dfs = precompute(prices_dict, nikkei, [lb])

    rebalance = best_params.get("rebalance", "weekly")
    # monthly も対応
    rebal_dates = get_rebalance_dates(warmup, oos_end, rebalance)
    all_prices = pd.DataFrame({c: df["AdjC"] for c, df in prices_dict.items() if "AdjC" in df.columns})
    return_df = all_prices.pct_change()

    result = eval_params(best_params, factor_dfs, prices_dict, rebal_dates, nikkei, oos_start, return_df)
    if result:
        print(
            f"OOS結果: total={result['total_return_pct']}%, "
            f"sharpe={result['sharpe']}, max_dd={result['max_dd_pct']}%, "
            f"alpha={result['alpha_pct']}%, nikkei={result['nikkei_pct']}%",
            flush=True,
        )
    else:
        print("OOS検証: 結果なし（データ不足）", flush=True)

    return result


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    run_grid(start=args.start, end=args.end, n_codes=args.n)
