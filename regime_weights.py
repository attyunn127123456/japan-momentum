"""
相場レジーム判定と、レジーム別の重みセットを管理するモジュール。
OHLCVのみで計算可能（外部APIなし）。
"""
import numpy as np
import pandas as pd

REGIME_WEIGHTS = {
    "bull":     {"ret_w": 0.40, "rs_w": 0.35, "green_w": 0.15, "smooth_w": 0.10, "lookback": 60, "top_n": 15},
    "bear":     {"ret_w": 0.20, "rs_w": 0.20, "green_w": 0.35, "smooth_w": 0.25, "lookback": 40, "top_n": 5},
    "high_vol": {"ret_w": 0.30, "rs_w": 0.30, "green_w": 0.20, "smooth_w": 0.20, "lookback": 30, "top_n": 8},
    "neutral":  {"ret_w": 0.30, "rs_w": 0.30, "green_w": 0.20, "smooth_w": 0.20, "lookback": 60, "top_n": 10},
}


def detect_regime(nikkei: pd.Series, date) -> str:
    """指定日のレジームを返す。優先度: high_vol > bear > bull > neutral"""
    ts = pd.Timestamp(date)
    hist = nikkei.loc[:ts].dropna()
    if len(hist) < 21:
        return "neutral"

    # 20日ボラ（日次リターンのstd×√252）
    daily_ret = hist.pct_change().dropna()
    vol_20 = float(daily_ret.tail(20).std() * np.sqrt(252)) if len(daily_ret) >= 20 else 0.0

    # high_vol: 年率25%超
    if vol_20 > 0.25:
        return "high_vol"

    current_price = float(hist.iloc[-1])
    ma200 = float(hist.tail(200).mean()) if len(hist) >= 200 else float(hist.mean())
    ret_20 = float(hist.iloc[-1] / hist.iloc[-21] - 1) if len(hist) >= 21 else 0.0
    above_ma200 = current_price > ma200

    # bear: 200MA下 OR 20日リターン < -5%
    if (not above_ma200) or (ret_20 < -0.05):
        return "bear"

    # bull: 200MA上 かつ 20日リターン > 0
    if above_ma200 and ret_20 > 0:
        return "bull"

    return "neutral"


def get_weights_for_date(nikkei: pd.Series, date) -> dict:
    """その日のレジーム重みを返す"""
    regime = detect_regime(nikkei, date)
    return {**REGIME_WEIGHTS[regime], "regime": regime}


def backtest_with_regime(prices_dict: dict, nikkei: pd.Series, rebal_dates: list,
                          start: str, return_df: pd.DataFrame) -> dict:
    """レジーム切り替えありでバックテスト実行。"""
    dates = [d for d in rebal_dates if str(d.date()) >= start]
    if not dates:
        return None

    portfolio = 1_000_000.0
    returns = []
    regime_log = []

    for i, date in enumerate(dates[:-1]):
        next_date = dates[i + 1]

        # レジーム判定 & 重み取得
        w = get_weights_for_date(nikkei, date)
        regime = w["regime"]
        lb = w["lookback"]
        top_n = w["top_n"]

        scores = {}
        for code, df in prices_dict.items():
            if "AdjC" not in df.columns:
                continue
            p = df["AdjC"].loc[:date].dropna()
            if len(p) < lb + 5:
                continue
            try:
                ret = float(p.iloc[-1] / p.iloc[-lb] - 1)
                nk_hist = nikkei.loc[:date].dropna()
                nk_ret = float(nk_hist.iloc[-1] / nk_hist.iloc[-lb] - 1) if len(nk_hist) > lb else 0.0
                rs = ret - nk_ret
                daily = p.pct_change().dropna().tail(20)
                green = float((daily > 0).mean()) if len(daily) >= 10 else 0.5
                window = p.tail(lb).values.astype(float)
                x = np.arange(len(window), dtype=float)
                if len(x) > 1:
                    cov = np.cov(x, window)
                    slope = cov[0, 1] / (np.var(x) + 1e-10)
                    pred = window.mean() + slope * (x - x.mean())
                    ss_tot = np.var(window) * len(window)
                    ss_res = np.sum((window - pred) ** 2)
                    smooth = max(0.0, 1.0 - ss_res / (ss_tot + 1e-10))
                else:
                    smooth = 0.0
                score = (w["ret_w"] * ret + w["rs_w"] * rs +
                         w["green_w"] * green + w["smooth_w"] * smooth)
                scores[code] = score
            except Exception:
                continue

        if not scores:
            continue

        top_codes = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_n]

        if date in return_df.index and next_date in return_df.index:
            tot, cnt = 0.0, 0
            for code in top_codes:
                if code in return_df.columns:
                    r = return_df.at[next_date, code]
                    if not np.isnan(r):
                        tot += r
                        cnt += 1
            if cnt > 0:
                period_ret = tot / cnt
                portfolio *= (1 + period_ret)
                returns.append(period_ret)
                regime_log.append({"date": str(date.date()), "regime": regime, "ret": round(period_ret, 4)})

    if len(returns) < 5:
        return None

    arr = np.array(returns)
    tr = portfolio / 1_000_000 - 1
    sharpe = float(arr.mean() / arr.std() * np.sqrt(52)) if arr.std() > 0 else 0.0
    cum = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(cum)
    dd = float(abs(((cum - peak) / peak).min()))
    nk = nikkei.loc[start:]
    nk_ret = float(nk.iloc[-1] / nk.iloc[0] - 1) if len(nk) > 1 else 0.0

    regime_counts = {}
    for entry in regime_log:
        regime_counts[entry["regime"]] = regime_counts.get(entry["regime"], 0) + 1

    return {
        "type": "regime",
        "total_return_pct": round(tr * 100, 2),
        "alpha_pct": round((tr - nk_ret) * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(dd * 100, 2),
        "nikkei_pct": round(nk_ret * 100, 2),
        "n_trades": len(returns),
        "regime_counts": regime_counts,
        "regime_log_tail": regime_log[-10:],
    }
