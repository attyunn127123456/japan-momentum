"""
Layer 1: 相場の空気シグナル
毎日の市場全体データから「今日の相場モード」を計算する。

Signals:
  1. market_breadth   : 当日上昇銘柄数ベースの強弱 (-1 to +1)
  2. nikkei_momentum  : 5/20/60日モメンタム
  3. foreign_flow     : 外国人フローのトレンド
  4. cross_section_vol: 銘柄間リターンのクロスセクション標準偏差
  5. earnings_momentum: 直近EPS上方修正銘柄の比率

detect_regime() でこれらを統合してレジームを判定する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 1. Market Breadth
# ---------------------------------------------------------------------------

def calc_market_breadth(prices_dict: dict, date) -> float:
    """
    当日上昇銘柄数 / 全銘柄数 を -1〜+1 に正規化して返す。
    date 当日の前日比がプラスなら上昇とみなす。
    """
    date = pd.Timestamp(date)
    up = 0
    total = 0
    for code, df in prices_dict.items():
        if df is None or df.empty:
            continue
        col = "AdjC" if "AdjC" in df.columns else ("C" if "C" in df.columns else None)
        if col is None:
            continue
        s = df[col].loc[:date].dropna()
        if len(s) < 2:
            continue
        # dateと直前の値があれば OK
        last = s.iloc[-1]
        prev = s.iloc[-2]
        if prev <= 0:
            continue
        total += 1
        if last > prev:
            up += 1

    if total == 0:
        return 0.0

    ratio = up / total          # 0〜1
    # 正規化: ratio=0.5 → 0, ratio=1 → +1, ratio=0 → -1
    return (ratio - 0.5) * 2.0


# ---------------------------------------------------------------------------
# 2. Nikkei Momentum
# ---------------------------------------------------------------------------

def calc_nikkei_momentum(
    nikkei: pd.Series,
    date,
    windows: List[int] = None,
) -> dict:
    """
    5/20/60日のシンプルリターン（モメンタム）を返す。
    nikkei: Date-indexed pd.Series of closing prices
    """
    if windows is None:
        windows = [5, 20, 60]

    date = pd.Timestamp(date)
    s = nikkei.loc[:date].dropna()
    result = {}
    for w in windows:
        key = f"mom{w}"
        if len(s) <= w:
            result[key] = 0.0
        else:
            cur = float(s.iloc[-1])
            past = float(s.iloc[-w - 1])
            result[key] = (cur / past - 1) if past > 0 else 0.0
    return result


# ---------------------------------------------------------------------------
# 3. Foreign Flow
# ---------------------------------------------------------------------------

def calc_foreign_flow(investor_types_df: pd.DataFrame, date) -> float:
    """
    直近4週間の外国人買越額(FrgnBal)のトレンドを返す。
    正 = 買い越し増加傾向（強気サイン）
    investor_types_df は PubDate 列を持つ weekly データ。
    """
    date = pd.Timestamp(date)
    df = investor_types_df.copy()

    # PubDate を datetime に
    df["_date"] = pd.to_datetime(df["PubDate"], errors="coerce")
    df = df.dropna(subset=["_date", "FrgnBal"])
    df = df[df["_date"] <= date].sort_values("_date")

    # 直近4週間分（最大4行）
    recent = df.tail(4)
    if len(recent) < 2:
        return 0.0

    # 単純な線形トレンド（slope of FrgnBal over time index）
    y = recent["FrgnBal"].values.astype(float)
    x = np.arange(len(y))
    if np.std(x) == 0:
        return 0.0

    slope = np.polyfit(x, y, 1)[0]

    # 正規化: 総買越額の絶対値スケールに対して slope を比率化
    scale = np.abs(y).mean() if np.abs(y).mean() > 0 else 1.0
    normalized = np.clip(slope / scale, -1.0, 1.0)
    return float(normalized)


# ---------------------------------------------------------------------------
# 4. Cross-Sectional Volatility
# ---------------------------------------------------------------------------

def calc_cross_section_vol(prices_dict: dict, date, lookback: int = 20) -> float:
    """
    直近 lookback 日の銘柄間リターンのクロスセクション標準偏差を返す。
    高い → 銘柄ごとに動きがバラバラ（選別相場）
    低い → 全部一緒に動く（マクロ主導）
    """
    date = pd.Timestamp(date)
    start = date - pd.Timedelta(days=lookback * 2)  # 余裕を持って取得

    daily_rets = []
    for code, df in prices_dict.items():
        if df is None or df.empty:
            continue
        col = "AdjC" if "AdjC" in df.columns else ("C" if "C" in df.columns else None)
        if col is None:
            continue
        s = df[col].loc[start:date].dropna()
        if len(s) < lookback + 1:
            continue
        r = s.pct_change().dropna().tail(lookback)
        daily_rets.append(r)

    if len(daily_rets) < 10:
        return 0.0

    ret_df = pd.concat(daily_rets, axis=1)
    # 各日のクロスセクション標準偏差を計算し平均
    cs_std = ret_df.std(axis=1).mean()
    return float(cs_std)


# ---------------------------------------------------------------------------
# 5. Earnings Momentum
# ---------------------------------------------------------------------------

def calc_earnings_momentum(
    fins_summary_df: pd.DataFrame,
    date,
    lookback_days: int = 30,
) -> float:
    """
    直近 lookback_days 日以内に発表された決算でEPS上方修正した銘柄の比率を返す。
    fins_summary_df は DiscDate, Code, EPS, FEPS 列を持つ。
    """
    date = pd.Timestamp(date)
    start = date - pd.Timedelta(days=lookback_days)

    df = fins_summary_df.copy()
    df["_disc"] = pd.to_datetime(df["DiscDate"], errors="coerce")
    df = df.dropna(subset=["_disc"])
    recent = df[(df["_disc"] > start) & (df["_disc"] <= date)].copy()

    if recent.empty:
        return 0.0

    # FEPS（会社予想EPS）とEPS実績を比較して上方修正を判定
    # 実績EPSが前回予想を超えていれば上方修正とみなす
    # シンプルに: EPS > 0 かつ FEPS が存在してEPS > FEPS * 0.9 を上方修正とする
    # もしくは同一Codeの前回開示と比較する方法もあるが、ここはシンプルに実装
    has_feps = "FEPS" in recent.columns
    if has_feps:
        recent = recent.dropna(subset=["EPS", "FEPS"])
        if recent.empty:
            return 0.0
        upward = (recent["EPS"] > recent["FEPS"]).sum()
        total = len(recent)
    else:
        recent = recent.dropna(subset=["EPS"])
        if recent.empty:
            return 0.0
        upward = (recent["EPS"] > 0).sum()
        total = len(recent)

    return float(upward / total) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# 6. Regime Detection
# ---------------------------------------------------------------------------

def detect_regime(
    prices_dict: dict,
    nikkei: pd.Series,
    investor_types_df: pd.DataFrame,
    fins_summary_df: pd.DataFrame,
    date,
) -> dict:
    """
    上記5つのシグナルを組み合わせてレジームを判定する。

    Returns:
        {
            'date': '2024-01-15',
            'regime': 'bull_trend',   # bull_trend / bear_trend / choppy / crash
            'score': 0.72,
            'signals': {...},
            'recommended_strategy': {...},
        }
    """
    date = pd.Timestamp(date)
    date_str = date.strftime("%Y-%m-%d")

    # --- シグナル計算 ---
    breadth = calc_market_breadth(prices_dict, date)
    mom_dict = calc_nikkei_momentum(nikkei, date)
    foreign_flow = calc_foreign_flow(investor_types_df, date)
    cross_vol = calc_cross_section_vol(prices_dict, date)
    earnings_mom = calc_earnings_momentum(fins_summary_df, date)

    mom5 = mom_dict.get("mom5", 0.0)
    mom20 = mom_dict.get("mom20", 0.0)
    mom60 = mom_dict.get("mom60", 0.0)

    signals = {
        "breadth": round(float(breadth), 4),
        "mom5": round(float(mom5), 4),
        "mom20": round(float(mom20), 4),
        "mom60": round(float(mom60), 4),
        "foreign_flow": round(float(foreign_flow), 4),
        "cross_vol": round(float(cross_vol), 6),
        "earnings_momentum": round(float(earnings_mom), 4),
    }

    # --- レジーム判定（優先順位: crash > bear > bull > choppy）---
    # breadth は calc_market_breadth の出力 (-1〜+1)
    # 元スペックのブレッドス比率 0.55 → normalized +0.1
    # 元スペックのブレッドス比率 0.40 → normalized -0.2
    # 元スペックのブレッドス比率 0.30 → normalized -0.4

    if mom20 < -0.08 or (mom5 < -0.05 and breadth < -0.4):
        # 暴落: 20日-8%以上の急落 or 5日-5%+全面安
        regime = "crash"
    elif mom20 < -0.04 or (mom60 < -0.05 and breadth < -0.2):
        # 弱気: 20日-4%超の下落 or 60日-5%+広範な下落
        regime = "bear_trend"
    elif mom60 > 0.05 and (breadth > -0.1 or mom20 > 0.03):
        # 強気: 60日+5%超 かつ (直近20日もプラス or 幅広上昇)
        regime = "bull_trend"
    else:
        regime = "choppy"

    # --- 強気度スコア (0〜1) ---
    # 各シグナルを 0〜1 に変換して平均
    score_parts = [
        np.clip((breadth + 1) / 2, 0, 1),           # breadth -1〜+1 → 0〜1
        np.clip((mom60 + 0.10) / 0.20, 0, 1),       # -10%〜+10% → 0〜1
        np.clip((mom20 + 0.05) / 0.10, 0, 1),       # -5%〜+5% → 0〜1
        np.clip((foreign_flow + 1) / 2, 0, 1),      # -1〜+1 → 0〜1
        np.clip(earnings_mom, 0, 1),                 # すでに 0〜1
    ]
    score = float(np.mean(score_parts))

    # --- 推奨パラメータ ---
    strategy_map = {
        "bull_trend": {
            "top_n": 2,
            "lookback": 100,
            "trailing_stop": -0.027,
            "use_cash": False,
        },
        "choppy": {
            "top_n": 4,
            "lookback": 60,
            "trailing_stop": -0.04,
            "use_cash": False,
        },
        "bear_trend": {
            "top_n": 5,
            "lookback": 40,
            "trailing_stop": -0.02,
            "use_cash": False,
        },
        "crash": {
            "top_n": 0,
            "lookback": 20,
            "trailing_stop": -0.01,
            "use_cash": True,
        },
    }

    return {
        "date": date_str,
        "regime": regime,
        "score": round(score, 4),
        "signals": signals,
        "recommended_strategy": strategy_map[regime],
    }


# ---------------------------------------------------------------------------
# Utility: bulk loader for prices_dict
# ---------------------------------------------------------------------------

def load_prices_dict(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """
    ticker リストを渡すと {code: DataFrame} の辞書を返す。
    fetch_cache.read_ohlcv() を利用。
    """
    from fetch_cache import read_ohlcv

    prices = {}
    for code in tickers:
        c = code.replace(".T", "")
        df = read_ohlcv(c, start=start, end=end)
        if df is not None and not df.empty:
            prices[c] = df
    return prices
