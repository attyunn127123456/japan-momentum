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
            # min_periods=max(5, lb//10) で上昇日が少なくても計算できるようにする
            up_days = nk_rets > 0
            stock_ret_on_up = dr.where(up_days)
            nk_ret_on_up = nk_rets.reindex(dr.index).where(up_days.reindex(dr.index))
            _uc_min = max(5, lb // 10)
            capture = (stock_ret_on_up.rolling(lb, min_periods=_uc_min).mean() /
                       nk_ret_on_up.rolling(lb, min_periods=_uc_min).mean().replace(0, np.nan)).astype(float)
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
                def _vol_slope_fn(x):
                    mask = ~np.isnan(x)
                    if mask.sum() < 10:
                        return np.nan
                    idx = np.where(mask)[0]
                    return float(np.polyfit(idx, x[mask], 1)[0])
                vol_slope = log_vol.rolling(lb, min_periods=10).apply(
                    _vol_slope_fn,
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


def compute_exit_signal(prices_df, current_date, entry_date, params):
    """
    現在保有中の銘柄の「天井スコア」を計算。
    複数シグナルを合成し、スコアが exit_threshold を超えたら売り。
    上昇中（直近高値から5%以内かつエントリー比プラス）はスコアを半減させる。

    Returns: (should_exit: bool, score: float)
    """
    try:
        lb = params.get("exit_lookback", 20)
        mask = prices_df.index <= current_date
        df = prices_df[mask].tail(lb + 5)
        if len(df) < 10:
            return False, 0.0

        close = df['AdjC'].values
        high = df['H'].values if 'H' in df.columns else close
        low = df['L'].values if 'L' in df.columns else close
        # 出来高カラム名: Vo または AdjVo
        if 'Vo' in df.columns:
            volume = df['Vo'].values
        elif 'AdjVo' in df.columns:
            volume = df['AdjVo'].values
        else:
            volume = None

        score = 0.0

        # 1. RSIシグナル（RSI > 75 = 過熱）
        rsi_w = params.get("exit_rsi_w", 0.3)
        if len(close) >= 14:
            delta = np.diff(close)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gain[-14:])
            avg_loss = np.mean(loss[-14:])
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - 100 / (1 + rs)
                if rsi > 75:
                    score += rsi_w * (rsi - 75) / 25  # 75〜100を0〜1に正規化

        # 2. クライマックス出来高（出来高 > 平均の2.0倍）
        vol_w = params.get("exit_vol_w", 0.25)
        if volume is not None and len(volume) >= 10:
            avg_vol = np.mean(volume[-10:-1])
            current_vol = volume[-1]
            if avg_vol > 0 and current_vol > avg_vol * 2.0:
                score += vol_w * min((current_vol / avg_vol - 2.0) / 2.0, 1.0)

        # 3. 高値からの下落（直近高値から-3%以上落ちた）
        dd_w = params.get("exit_dd_w", 0.3)
        peak = np.max(close[-lb:])
        current = close[-1]
        dd_from_peak = (current - peak) / peak
        if dd_from_peak < -0.03:
            score += dd_w * min(abs(dd_from_peak) / 0.15, 1.0)

        # 4. モメンタム失速（短期モメンタムが鈍化）
        mom_w = params.get("exit_mom_w", 0.15)
        if len(close) >= 11:
            ret5  = (close[-1] / close[-6]  - 1) if close[-6]  > 0 else 0
            ret10 = (close[-1] / close[-11] - 1) if close[-11] > 0 else 0
            if ret5 < ret10 * 0.3:  # 直近5日が直近10日の30%未満 = 失速
                score += mom_w

        # エントリー後上昇中は売らない（高値更新中はスコア半減）
        if entry_date is not None:
            mask_entry = (prices_df.index >= entry_date) & (prices_df.index <= current_date)
            since_entry = prices_df[mask_entry]['AdjC']
            if len(since_entry) >= 2:
                entry_price = since_entry.iloc[0]
                recent_high = since_entry.max()
                current_price = since_entry.iloc[-1]
                if current_price >= entry_price * 1.02 and current_price >= recent_high * 0.95:
                    score *= 0.5

        threshold = params.get("exit_threshold", 0.5)
        return score >= threshold, round(score, 3)

    except Exception:
        return False, 0.0


def _eval_params_regime(default_params, factor_dfs, prices_dict, rebal_dates, nikkei, start,
                        return_df, regime_params, long_short=False, use_open_prices=False):
    """
    レジーム適応型 eval_params。
    リバランス日ごとに detect_regime() を呼び、対応する params に切り替えて評価。
    regime_params[regime] = None の場合はそのリバランス期間をスキップ（キャッシュ保持）。
    """
    from backtest import detect_regime

    dates = [d for d in rebal_dates if str(d.date()) >= start]

    # 各 regime ごとに score_df をキャッシュ（lookback が異なる可能性があるため）
    _score_cache = {}

    def get_score_df_for_params(p):
        lb = p["lookback"]
        weights = {k: p.get(k + "_w", 0.0) for k in ["ret", "rs", "green", "smooth", "resilience"]}
        cache_key = (lb, tuple(sorted(weights.items())))
        if cache_key not in _score_cache:
            _score_cache[cache_key] = build_score_df(factor_dfs, lb, weights)
        return _score_cache[cache_key]

    portfolio = 1_000_000.0
    returns = []
    equity_curve = []

    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        next_next_date = dates[i + 2] if i + 2 < len(dates) else None

        # レジーム判定
        regime = detect_regime(nikkei, date)
        # regime_params に対応するエントリーがあれば上書き、なければ default_params を使用
        if regime in regime_params:
            active_params = regime_params[regime]
        else:
            active_params = default_params

        # None = 全キャッシュ（ポジション取らない）
        if active_params is None:
            equity_curve.append({
                "date": str(date.date()),
                "value": round((portfolio / 1_000_000 - 1) * 100, 2),
                "holdings": [],
                "regime": regime,
            })
            continue

        lb = active_params["lookback"]
        tn = active_params["top_n"]
        score_df = get_score_df_for_params(active_params)

        if score_df is None or date not in score_df.index:
            continue

        row = score_df.loc[date].dropna()
        if row.empty:
            continue
        top = row.nlargest(tn).index.tolist()

        trailing_stop = active_params.get("trailing_stop", None)
        exit_threshold = active_params.get("exit_threshold", None)

        def calc_period_return(code, date, next_date, next_next_date=None):
            use_trailing = trailing_stop is not None and code in prices_dict
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
                                should_exit, _ = compute_exit_signal(price_df, idx, date, active_params)
                                if should_exit:
                                    exit_price = daily_price
                                    break
                        ret = (exit_price - entry_price) / entry_price
                        return ret if not np.isnan(ret) else None
                except Exception:
                    pass
            if code in return_df.columns:
                r = return_df.at[next_date, code]
                if not np.isnan(r):
                    return r
            return None

        tot, cnt = 0.0, 0
        if date in return_df.index and next_date in return_df.index:
            for code in top:
                r = calc_period_return(code, date, next_date, next_next_date)
                if r is not None:
                    tot += r
                    cnt += 1

        if cnt > 0:
            r = tot / cnt
            portfolio *= (1 + r)
            returns.append(r)

        equity_curve.append({
            "date": str(date.date()),
            "value": round((portfolio / 1_000_000 - 1) * 100, 2),
            "holdings": [str(c) for c in top],
            "regime": regime,
        })

    if len(returns) < 5:
        return None

    arr = np.array(returns)
    tr = portfolio / 1_000_000 - 1
    sharpe = float(arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0
    cum = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(cum)
    dd = float(abs(((cum - peak) / peak).min()))
    nk = nikkei.loc[start:]
    nk_ret = float(nk.iloc[-1] / nk.iloc[0] - 1) if len(nk) > 1 else 0

    return {
        **default_params,
        "total_return_pct": round(tr * 100, 2),
        "alpha_pct": round((tr - nk_ret) * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(dd * 100, 2),
        "nikkei_pct": round(nk_ret * 100, 2),
        "n_trades": len(returns),
        "equity_curve": equity_curve,
        "regime_adaptive": True,
    }


def eval_params(params, factor_dfs, prices_dict, rebal_dates, nikkei, start, return_df,
                long_short=False, use_open_prices=False, regime_params=None):
    # regime_params が渡された場合はレジーム適応型モードで実行
    if regime_params is not None:
        return _eval_params_regime(params, factor_dfs, prices_dict, rebal_dates, nikkei, start,
                                   return_df, regime_params, long_short=long_short,
                                   use_open_prices=use_open_prices)
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
            score_df = score_df.add(short_mom_ranked * short_momentum_w, fill_value=0)

    # 52週高値proximityファクター
    high52_w = params.get('high52_w', 0.0)
    if high52_w > 0 and lb in factor_dfs and 'high52' in factor_dfs[lb]:
        high52_ranked = factor_dfs[lb]['high52'].rank(axis=1, pct=True)
        if score_df is None:
            score_df = high52_ranked * high52_w
        else:
            score_df = score_df.add(high52_ranked * high52_w, fill_value=0)

    # Omega比ファクター（上昇/下落の非対称性）
    omega_w = params.get('omega_w', 0.0)
    if omega_w > 0 and lb in factor_dfs and 'omega' in factor_dfs[lb]:
        omega_ranked = factor_dfs[lb]['omega'].rank(axis=1, pct=True)
        if score_df is None:
            score_df = omega_ranked * omega_w
        else:
            score_df = score_df.add(omega_ranked * omega_w, fill_value=0)

    # 終値位置ファクター（引け買い圧力）
    close_location_w = params.get('close_location_w', 0.0)
    if close_location_w > 0 and lb in factor_dfs and 'close_location' in factor_dfs[lb]:
        cl_ranked = factor_dfs[lb]['close_location'].rank(axis=1, pct=True)
        if score_df is None:
            score_df = cl_ranked * close_location_w
        else:
            score_df = score_df.add(cl_ranked * close_location_w, fill_value=0)

    # ATRブレイクアウト検出ファクター
    range_expand_w = params.get('range_expand_w', 0.0)
    if range_expand_w > 0 and lb in factor_dfs and 'range_expand' in factor_dfs[lb]:
        re_ranked = factor_dfs[lb]['range_expand'].rank(axis=1, pct=True)
        if score_df is None:
            score_df = re_ranked * range_expand_w
        else:
            score_df = score_df.add(re_ranked * range_expand_w, fill_value=0)

    # 最大連続陽線日数ファクター
    win_streak_w = params.get('win_streak_w', 0.0)
    if win_streak_w > 0 and lb in factor_dfs and 'win_streak' in factor_dfs[lb]:
        ws_ranked = factor_dfs[lb]['win_streak'].rank(axis=1, pct=True)
        if score_df is None:
            score_df = ws_ranked * win_streak_w
        else:
            score_df = score_df.add(ws_ranked * win_streak_w, fill_value=0)

    # 業種モメンタムファクター
    sector_momentum_w = params.get('sector_momentum_w', 0.0)
    if sector_momentum_w > 0 and lb in factor_dfs and 'sector_momentum' in factor_dfs[lb]:
        sm_ranked = factor_dfs[lb]['sector_momentum'].rank(axis=1, pct=True)
        if score_df is None:
            score_df = sm_ranked * sector_momentum_w
        else:
            score_df = score_df.add(sm_ranked * sector_momentum_w, fill_value=0)

    # 夜間リターンファクター
    overnight_return_w = params.get('overnight_return_w', 0.0)
    if overnight_return_w > 0 and lb in factor_dfs and 'overnight_return' in factor_dfs[lb]:
        or_ranked = factor_dfs[lb]['overnight_return'].rank(axis=1, pct=True)
        if score_df is None:
            score_df = or_ranked * overnight_return_w
        else:
            score_df = score_df.add(or_ranked * overnight_return_w, fill_value=0)

    # 出来高加速度ファクター
    volume_acceleration_w = params.get('volume_acceleration_w', 0.0)
    if volume_acceleration_w > 0 and lb in factor_dfs and 'volume_acceleration' in factor_dfs[lb]:
        va_ranked = factor_dfs[lb]['volume_acceleration'].rank(axis=1, pct=True)
        if score_df is None:
            score_df = va_ranked * volume_acceleration_w
        else:
            score_df = score_df.add(va_ranked * volume_acceleration_w, fill_value=0)

    # 下値切り上げファクター
    higher_lows_w = params.get('higher_lows_w', 0.0)
    if higher_lows_w > 0 and lb in factor_dfs and 'higher_lows' in factor_dfs[lb]:
        hl_ranked = factor_dfs[lb]['higher_lows'].rank(axis=1, pct=True)
        score_df = hl_ranked * higher_lows_w if score_df is None else score_df.add(hl_ranked * higher_lows_w, fill_value=0)

    # 実体強度ファクター
    body_strength_w = params.get('body_strength_w', 0.0)
    if body_strength_w > 0 and lb in factor_dfs and 'body_strength' in factor_dfs[lb]:
        bs_ranked = factor_dfs[lb]['body_strength'].rank(axis=1, pct=True)
        score_df = bs_ranked * body_strength_w if score_df is None else score_df.add(bs_ranked * body_strength_w, fill_value=0)

    # 出来高リターン相関ファクター
    vol_return_corr_w = params.get('vol_return_corr_w', 0.0)
    if vol_return_corr_w > 0 and lb in factor_dfs and 'vol_return_corr' in factor_dfs[lb]:
        vrc_ranked = factor_dfs[lb]['vol_return_corr'].rank(axis=1, pct=True)
        score_df = vrc_ranked * vol_return_corr_w if score_df is None else score_df.add(vrc_ranked * vol_return_corr_w, fill_value=0)

    # 方向性出来高ファクター
    accumulation_w = params.get('accumulation_w', 0.0)
    if accumulation_w > 0 and lb in factor_dfs and 'accumulation' in factor_dfs[lb]:
        acc_ranked = factor_dfs[lb]['accumulation'].rank(axis=1, pct=True)
        score_df = acc_ranked * accumulation_w if score_df is None else score_df.add(acc_ranked * accumulation_w, fill_value=0)

    # 週次勝率ファクター
    momentum_consistency_w = params.get('momentum_consistency_w', 0.0)
    if momentum_consistency_w > 0 and lb in factor_dfs and 'momentum_consistency' in factor_dfs[lb]:
        mc_ranked = factor_dfs[lb]['momentum_consistency'].rank(axis=1, pct=True)
        score_df = mc_ranked * momentum_consistency_w if score_df is None else score_df.add(mc_ranked * momentum_consistency_w, fill_value=0)

    # 上方キャプチャファクター
    upside_capture_w = params.get('upside_capture_w', 0.0)
    if upside_capture_w > 0 and lb in factor_dfs and 'upside_capture' in factor_dfs[lb]:
        uc_ranked = factor_dfs[lb]['upside_capture'].rank(axis=1, pct=True)
        score_df = uc_ranked * upside_capture_w if score_df is None else score_df.add(uc_ranked * upside_capture_w, fill_value=0)

    # 寄り付きギャップモメンタムファクター
    gap_momentum_w = params.get('gap_momentum_w', 0.0)
    if gap_momentum_w > 0 and lb in factor_dfs and 'gap_momentum' in factor_dfs[lb]:
        gm_ranked = factor_dfs[lb]['gap_momentum'].rank(axis=1, pct=True)
        score_df = gm_ranked * gap_momentum_w if score_df is None else score_df.add(gm_ranked * gap_momentum_w, fill_value=0)

    # 出来高確認モメンタムファクター
    volume_confirm_w = params.get('volume_confirm_w', 0.0)
    if volume_confirm_w > 0 and lb in factor_dfs and 'volume_confirm' in factor_dfs[lb]:
        vc_ranked = factor_dfs[lb]['volume_confirm'].rank(axis=1, pct=True)
        score_df = vc_ranked * volume_confirm_w if score_df is None else score_df.add(vc_ranked * volume_confirm_w, fill_value=0)

    # スキップ期間モメンタムファクター
    ret_skip_w = params.get('ret_skip_w', 0.0)
    if ret_skip_w > 0 and lb in factor_dfs and 'ret_skip' in factor_dfs[lb]:
        rs_ranked = factor_dfs[lb]['ret_skip'].rank(axis=1, pct=True)
        score_df = rs_ranked * ret_skip_w if score_df is None else score_df.add(rs_ranked * ret_skip_w, fill_value=0)

    # マルチTFモメンタム一致度ファクター
    cluster_boost_w = params.get('cluster_boost_w', 0.0)
    if cluster_boost_w > 0 and lb in factor_dfs and 'cluster_boost' in factor_dfs[lb]:
        cb_ranked = factor_dfs[lb]['cluster_boost'].rank(axis=1, pct=True)
        score_df = cb_ranked * cluster_boost_w if score_df is None else score_df.add(cb_ranked * cluster_boost_w, fill_value=0)

    # リターン自己相関ファクター
    return_autocorr_w = params.get('return_autocorr_w', 0.0)
    if return_autocorr_w > 0 and lb in factor_dfs and 'return_autocorr' in factor_dfs[lb]:
        ra_ranked = factor_dfs[lb]['return_autocorr'].rank(axis=1, pct=True)
        score_df = ra_ranked * return_autocorr_w if score_df is None else score_df.add(ra_ranked * return_autocorr_w, fill_value=0)

    # 出来高トレンドファクター
    volume_slope_w = params.get('volume_slope_w', 0.0)
    if volume_slope_w > 0 and lb in factor_dfs and 'volume_slope' in factor_dfs[lb]:
        vs_ranked = factor_dfs[lb]['volume_slope'].rank(axis=1, pct=True)
        score_df = vs_ranked * volume_slope_w if score_df is None else score_df.add(vs_ranked * volume_slope_w, fill_value=0)

    # クリーンモメンタムファクター
    clean_momentum_w = params.get('clean_momentum_w', 0.0)
    if clean_momentum_w > 0 and lb in factor_dfs and 'clean_momentum' in factor_dfs[lb]:
        cm_ranked = factor_dfs[lb]['clean_momentum'].rank(axis=1, pct=True)
        score_df = cm_ranked * clean_momentum_w if score_df is None else score_df.add(cm_ranked * clean_momentum_w, fill_value=0)

    if score_df is None:
        return None
    
    dates = [d for d in rebal_dates if str(d.date()) >= start]

    # トレーリングストップ設定（例: -0.07 = 高値から7%下落で即売り）
    trailing_stop = params.get("trailing_stop", None)
    # 天井検知エグジット設定
    exit_threshold = params.get("exit_threshold", None)

    def calc_period_return(code, date, next_date, next_next_date=None):
        """
        保有期間リターンを計算。
        use_open_prices=True の場合:
          - エントリー: next_date の AdjO（翌朝初値）
          - エグジット: 天井/ストップ発動時は当日 AdjC、それ以外は next_next_date の AdjO
        通常モード:
          - エントリー: date の AdjC（前日終値）
          - エグジット: next_date の AdjC（次リバランス日終値）
        """
        if use_open_prices and code in prices_dict:
            try:
                price_df = prices_dict[code]
                # エントリー: next_date の AdjO（翌朝初値）
                if 'AdjO' not in price_df.columns:
                    pass  # fallthrough to normal mode
                else:
                    entry_rows = price_df.loc[price_df.index == next_date, 'AdjO']
                    if len(entry_rows) == 0:
                        pass  # fallthrough
                    else:
                        entry_price = float(entry_rows.iloc[0])
                        if entry_price <= 0 or np.isnan(entry_price):
                            pass  # fallthrough
                        else:
                            # エグジット候補: next_next_date の AdjO
                            exit_price = None
                            if next_next_date is not None:
                                exit_rows = price_df.loc[price_df.index == next_next_date, 'AdjO']
                                if len(exit_rows) > 0:
                                    exit_price = float(exit_rows.iloc[0])

                            # 当日（next_date）内でのストップ/天井シグナル判定
                            use_trailing = trailing_stop is not None
                            use_exit_signal = exit_threshold is not None
                            if use_trailing or use_exit_signal:
                                mask = price_df.index == next_date
                                day_close = price_df.loc[mask, 'AdjC']
                                if len(day_close) > 0:
                                    close_price = float(day_close.iloc[0])
                                    peak = entry_price
                                    # trailing_stop: 当日内に entry→close で判定
                                    if use_trailing:
                                        peak = max(peak, close_price)
                                        if (close_price - peak) / peak <= trailing_stop:
                                            exit_price = close_price  # 当日引けでストップ
                                    # 天井シグナル: 当日引け時点で判定
                                    if use_exit_signal and exit_price is None:
                                        should_exit, _ = compute_exit_signal(price_df, next_date, date, params)
                                        if should_exit:
                                            exit_price = close_price  # 当日引けでエグジット

                            if exit_price is None:
                                # next_next_dateのAdjOも取れない場合は当日終値にフォールバック
                                day_close = price_df.loc[price_df.index == next_date, 'AdjC']
                                if len(day_close) > 0:
                                    exit_price = float(day_close.iloc[0])

                            if exit_price and exit_price > 0 and not np.isnan(exit_price):
                                ret = (exit_price - entry_price) / entry_price
                                return ret if not np.isnan(ret) else None
            except Exception:
                pass

        # 通常モード（終値ベース）
        use_trailing = trailing_stop is not None and code in prices_dict
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
            r = return_df.at[next_date, code]
            if not np.isnan(r):
                return r
        return None

    portfolio = 1_000_000.0
    returns = []
    equity_curve = []

    for i, date in enumerate(dates[:-1]):
        next_date = dates[i+1]
        next_next_date = dates[i+2] if i+2 < len(dates) else None
        if date not in score_df.index:
            continue

        row = score_df.loc[date].dropna()
        if row.empty:
            continue
        top = row.nlargest(tn).index.tolist()

        # ロングショートモード: 下位tn銘柄をショート
        short_top = []
        if long_short:
            valid_row = row.dropna()
            if len(valid_row) >= tn * 2:
                short_top = valid_row.nsmallest(tn).index.tolist()

        # リターン計算（trailing_stop対応）
        tot, cnt = 0.0, 0
        if date in return_df.index and next_date in return_df.index:
            for code in top:
                r = calc_period_return(code, date, next_date, next_next_date)
                if r is not None:
                    tot += r; cnt += 1
            if long_short and short_top:
                short_tot, short_cnt = 0.0, 0
                for code in short_top:
                    r = calc_period_return(code, date, next_date, next_next_date)
                    if r is not None:
                        # ショート: リターン反転 - 借株コスト0.1%/週
                        short_tot += (-r)  # 手数料0円（信用売りの金利は別途考慮）
                        short_cnt += 1
                if short_cnt > 0:
                    # ロングとショートを平均
                    long_avg = tot / cnt if cnt > 0 else 0.0
                    short_avg = short_tot / short_cnt
                    combined = (long_avg + short_avg) / 2
                    portfolio *= (1 + combined)
                    returns.append(combined)
                    continue  # 下のif cnt > 0をスキップ
        if cnt > 0:
            r = tot / cnt
            portfolio *= (1 + r)
            returns.append(r)

        # equity_curve 記録
        equity_curve.append({
            "date": str(date.date()),
            "value": round((portfolio / 1_000_000 - 1) * 100, 2),
            "holdings": [str(c) for c in top]
        })

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
            "n_trades": len(returns), "equity_curve": equity_curve}


def run_grid(start="2016-01-01", end="2020-12-31", n_codes=2000):
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
                            date_map, start, return_df, n_trials=300,
                            val_date_map=None, val_start=None):
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
        # trailing_stop: 0.0 = なし, 0.03〜0.15 = 3〜15%ストップ
        trial_stop_raw = trial.suggest_float("trailing_stop_raw", 0.0, 0.15)
        trial_trailing = -trial_stop_raw if trial_stop_raw > 0.001 else None
        # 天井検知エグジット: exit_threshold_raw=0.0 → 無効, >0 → 有効
        trial_exit_threshold_raw = trial.suggest_float("exit_threshold_raw", 0.0, 0.8)
        trial_exit_threshold = trial_exit_threshold_raw if trial_exit_threshold_raw > 0.05 else None
        trial_exit_lookback = trial.suggest_int("exit_lookback", 10, 30)
        trial_exit_rsi_w = trial.suggest_float("exit_rsi_w", 0.0, 0.5)
        trial_exit_vol_w = trial.suggest_float("exit_vol_w", 0.0, 0.4)
        trial_exit_dd_w = trial.suggest_float("exit_dd_w", 0.0, 0.5)
        trial_exit_mom_w = trial.suggest_float("exit_mom_w", 0.0, 0.3)
        params = {
            'lookback': trial_lb,
            'top_n': top_n,
            'rebalance': trial_rb,
            'trailing_stop': trial_trailing,
            'exit_threshold': trial_exit_threshold,
            'exit_lookback': trial_exit_lookback,
            'exit_rsi_w': trial_exit_rsi_w,
            'exit_vol_w': trial_exit_vol_w,
            'exit_dd_w': trial_exit_dd_w,
            'exit_mom_w': trial_exit_mom_w,
            **weights,
        }

        try:
            rb_key = trial_rb if trial_rb in date_map else 'weekly'

            # IS期間の評価
            r_is = eval_params(params, factor_dfs, prices_dict,
                               date_map[rb_key], nikkei, start, return_df)
            if r_is is None:
                return -999.0
            if r_is.get('max_dd_pct', 100) > 50:
                return -999.0  # IS期間のDD上限

            # Calmar比（リターン÷MaxDD）: Sharpeよりモメンタム戦略に適している
            is_dd = max(r_is.get('max_dd_pct', 100), 1.0)
            is_calmar = r_is.get('total_return_pct', 0) / is_dd

            # OOS期間の評価（val_date_mapが渡されている場合）
            oos_calmar = None
            if val_date_map is not None and val_start is not None:
                val_rb_key = trial_rb if trial_rb in val_date_map else 'weekly'
                r_val = eval_params(params, factor_dfs, prices_dict,
                                    val_date_map[val_rb_key], nikkei, val_start, return_df)
                if r_val is not None and r_val.get('n_trades', 0) >= 10:
                    val_dd = max(r_val.get('max_dd_pct', 100), 1.0)
                    oos_calmar = r_val.get('total_return_pct', 0) / val_dd

            # 複合スコア: OOS重視（IS 30% + OOS 70%）
            if oos_calmar is not None:
                score = is_calmar * 0.3 + oos_calmar * 0.7
                # OOSがマイナスリターンなら大幅ペナルティ
                if r_val.get('total_return_pct', 0) < 0:
                    score -= 50.0
            else:
                # OOSなし（フォールバック）: IS calmarのみ
                score = is_calmar

            all_results.append((params, r_is))

            # ベスト更新時にログ
            if score > best_score:
                best_score = score
                oos_str = f", oos_total={r_val.get('total_return_pct',0):.1f}%" if oos_calmar is not None else ""
                print(f"  新ベスト: is_total={r_is.get('total_return_pct',0):.1f}%"
                      f"{oos_str}, calmar={score:.2f}, lb={trial_lb}, tn={top_n}", flush=True)

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


def run_walk_forward_validation(params, n_codes=500, codes=None):
    """
    Walk-Forward Cross Validationを実行。
    複数OOS期間での平均スコアを返す。

    Returns: {
        "folds": [{"id":..., "total_return_pct":..., "sharpe":..., "max_dd_pct":..., "passed":...}, ...],
        "avg_total_return": ...,
        "avg_sharpe": ...,
        "avg_max_dd": ...,
        "all_passed": bool,  # 全fold でmax_dd<45%かつtotal>0
        "n_passed": int,     # 合格fold数
    }
    """
    import warnings; warnings.filterwarnings('ignore')

    FOLDS = [
        {"id": "fold1", "test_start": "2021-01-01", "test_end": "2022-12-31"},
        {"id": "fold2", "test_start": "2023-01-01", "test_end": "2024-06-30"},
        {"id": "fold3", "test_start": "2024-07-01", "test_end": datetime.now().strftime("%Y-%m-%d")},
    ]

    # データロード（全期間分を1回だけロード）
    overall_start = "2016-01-01"
    overall_end = datetime.now().strftime("%Y-%m-%d")
    warmup = (datetime.strptime(overall_start, "%Y-%m-%d") - timedelta(days=200)).strftime("%Y-%m-%d")

    if codes is None:
        codes = get_top_liquid_tickers(n_codes)

    print(f"Walk-Forward: データロード ({len(codes)}銘柄)...", flush=True)
    prices_dict = {}
    for c in codes:
        df = read_ohlcv(c, warmup, overall_end)
        if df is not None and not df.empty and 'AdjC' in df.columns:
            prices_dict[c] = df

    if len(prices_dict) < 50:
        print(f"Walk-Forward: データ不足 ({len(prices_dict)}銘柄)", flush=True)
        return None

    print(f"Walk-Forward: {len(prices_dict)}銘柄ロード完了", flush=True)

    nikkei = get_nikkei_history(overall_start, overall_end)
    lb = params.get("lookback", 60)
    lookbacks = [lb] if lb else [20, 40, 60, 80, 100]
    factor_dfs = precompute(prices_dict, nikkei, lookbacks)
    return_df = pd.DataFrame({c: prices_dict[c]['AdjC'] for c in prices_dict}).pct_change()

    fold_results = []
    for fold in FOLDS:
        try:
            test_start = fold["test_start"]
            test_end = fold["test_end"]
            rebal = get_rebalance_dates(test_start, test_end, params.get("rebalance", "weekly"))
            result = eval_params(params, factor_dfs, prices_dict, rebal, nikkei, test_start, return_df)

            if result is None:
                fold_results.append({"id": fold["id"], "error": "計算失敗", "passed": False})
                continue

            passed = (result["max_dd_pct"] < 45.0 and result["total_return_pct"] > 0)
            fold_results.append({
                "id": fold["id"],
                "test_start": test_start,
                "test_end": test_end,
                "total_return_pct": result["total_return_pct"],
                "sharpe": result["sharpe"],
                "max_dd_pct": result["max_dd_pct"],
                "passed": passed,
            })
            print(f"  fold={fold['id']}: total={result['total_return_pct']:.1f}%, sharpe={result['sharpe']:.3f}, max_dd={result['max_dd_pct']:.1f}% {'✅' if passed else '❌'}", flush=True)
        except Exception as e:
            fold_results.append({"id": fold["id"], "error": str(e), "passed": False})
            print(f"  fold={fold['id']}: エラー {e}", flush=True)

    valid = [
        f for f in fold_results
        if "total_return_pct" in f
        and f["total_return_pct"] is not None
        and not (isinstance(f["total_return_pct"], float) and np.isnan(f["total_return_pct"]))
        and "max_dd_pct" in f
        and f["max_dd_pct"] is not None
        and not (isinstance(f["max_dd_pct"], float) and np.isnan(f["max_dd_pct"]))
    ]
    if not valid:
        return None

    return {
        "folds": fold_results,
        "avg_total_return": float(np.mean([f["total_return_pct"] for f in valid])),
        "avg_sharpe": float(np.mean([f["sharpe"] for f in valid])),
        "avg_max_dd": float(np.mean([f["max_dd_pct"] for f in valid])),
        "all_passed": all(f["passed"] for f in fold_results),
        "n_passed": sum(f["passed"] for f in fold_results),
    }


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2020-12-31")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    run_grid(start=args.start, end=args.end, n_codes=args.n)
