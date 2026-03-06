"""
Backtest: hold top-N momentum stocks, rebalance weekly.
Uses J-Quants V2 for data.

Usage:
  python3 backtest.py
  python3 backtest.py --start 2022-01-01 --end 2025-01-01 --top-n 5 --rebalance weekly
"""
import argparse
import json_safe as json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from jquants import fetch_all_history
from momentum import calculate_momentum_score, apply_filters


def get_nikkei_history(start: str, end: str) -> pd.Series:
    import requests, os
    API_KEY = os.environ.get("JQUANTS_API_KEY", "cph3PdiF8zxH9GxClcFfShcJdSUzuNpV9ho_zMPm4a8")
    all_data = []
    params = {"from": start, "to": end}
    while True:
        r = requests.get(
            "https://api.jquants.com/v2/indices/bars/daily/topix",
            headers={"x-api-key": API_KEY}, params=params
        )
        # フォールバック: yfinance
        if not r.ok:
            import yfinance as yf
            df = yf.download("^N225", start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df["Close"].dropna()
        d = r.json()
        all_data.extend(d.get("data", []))
        if not d.get("pagination_key"):
            break
        params = {**params, "pagination_key": d["pagination_key"]}

    df = pd.DataFrame(all_data)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()["C"].dropna()


def get_rebalance_dates(start: str, end: str, freq: str) -> list:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    freq_map = {"daily": "B", "weekly": "W-MON", "biweekly": "2W-MON", "monthly": "MS"}
    return [d for d in pd.date_range(s, e, freq=freq_map.get(freq, "W-MON")) if d <= e]


def score_on_date(prices_dict: dict, nikkei: pd.Series, ticker: str, date: pd.Timestamp) -> float:
    code = ticker.replace(".T", "")
    df = prices_dict.get(code)
    if df is None:
        df = prices_dict.get(ticker)
    if df is None or df.empty:
        return None
    p = df["AdjC"].loc[:date].dropna() if "AdjC" in df.columns else df["C"].loc[:date].dropna()
    n = nikkei.loc[:date].dropna()
    if len(p) < 60:
        return None
    v = pd.Series(np.ones(len(p)))
    passes, _ = apply_filters(ticker, p, v, market_cap=1e11)
    if not passes:
        return None
    scores = calculate_momentum_score(p, v, n)
    return scores["total"] if scores["valid"] else None


def run_backtest(start: str, end: str, top_n: int, rebalance: str, use_regime: bool = False) -> dict:
    from universe import get_top_liquid_tickers
    from fetch_cache import read_ohlcv
    raw_codes = get_top_liquid_tickers(500)
    TICKERS = [f"{c}.T" for c in raw_codes]

    from datetime import datetime as _dt, timedelta as _td
    warmup_start = (_dt.strptime(start, "%Y-%m-%d") - _td(days=180)).strftime("%Y-%m-%d")
    print(f"キャッシュからデータ読み込み中 ({warmup_start} → {end}, {len(TICKERS)}銘柄)...")
    prices_dict = {}
    for c in raw_codes:
        df = read_ohlcv(c, warmup_start, end)
        if df is not None and not df.empty:
            prices_dict[c] = df
    print(f"  {len(prices_dict)}銘柄のデータ読み込み完了")
    nikkei = get_nikkei_history(start, end)

    rebalance_dates = get_rebalance_dates(start, end, rebalance)
    if use_regime:
        import regime_weights as _rw
    portfolio_value = 1_000_000.0
    holdings: dict = {}
    trades_log = []
    weekly_holdings = []

    print(f"\nバックテスト開始: {start}〜{end}, Top-{top_n}, {rebalance}")

    for i, reb_date in enumerate(rebalance_dates):
        # レジーム対応: use_regime=True のとき重みとtop_nを動的切り替え
        if use_regime:
            _w = _rw.get_weights_for_date(nikkei, reb_date)
            top_n = _w["top_n"]
        # スコア計算
        scores = {}
        for t in TICKERS:
            code = t.replace(".T","")
            s = score_on_date(prices_dict, nikkei, t, reb_date)
            if s is not None:
                scores[code] = s
        if not scores:
            continue

        top_codes = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_n]
        top_tickers = [f'{c}.T' for c in top_codes]

        # 現在ポートフォリオ評価
        current_value = 0.0
        for t, shares in holdings.items():
            code = t.replace(".T", "")
            if code in prices_dict:
                p_series = prices_dict[code]["AdjC"].loc[:reb_date].dropna()
                if len(p_series) > 0:
                    current_value += shares * float(p_series.iloc[-1])
        if not holdings:
            current_value = portfolio_value

        # 売り → 買い (0.1% コスト)
        sell_cost = current_value * 0.001 * len(holdings)
        cash = current_value - sell_cost
        per_stock = cash / top_n
        new_holdings = {}
        for t in top_tickers:
            code = t.replace(".T", "")
            if code not in prices_dict:
                continue
            p_series = prices_dict[code]["AdjC"].loc[:reb_date].dropna()
            if len(p_series) == 0:
                continue
            price = float(p_series.iloc[-1])
            shares = (per_stock * 0.999) / price
            new_holdings[t] = shares

        holdings = new_holdings
        portfolio_value = cash * 0.999

        weekly_holdings.append({
            "date": str(reb_date.date()),
            "tickers": top_tickers,
            "scores": {t: round(scores[t.replace(".T","")], 1) for t in top_tickers},
        })
        trades_log.append({
            "date": str(reb_date.date()),
            "portfolio_value": round(portfolio_value, 0),
            "top_n": top_tickers,
        })

        if i % 10 == 0:
            print(f"  {reb_date.date()} | ¥{portfolio_value:,.0f} | {top_tickers[:3]}")

    # 最終価値
    end_ts = pd.Timestamp(end)
    final_value = sum(
        holdings[t] * float(prices_dict[t.replace(".T","")]["AdjC"].loc[:end_ts].iloc[-1])
        for t in holdings
        if t.replace(".T","") in prices_dict and len(prices_dict[t.replace(".T","")]["AdjC"].loc[:end_ts]) > 0
    ) or portfolio_value

    # 日経リターン
    n_start = float(nikkei.loc[pd.Timestamp(start):].iloc[0])
    n_end = float(nikkei.loc[:end_ts].iloc[-1])
    nikkei_return = (n_end / n_start - 1) * 100

    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365
    total_return = (final_value / 1_000_000 - 1) * 100
    annual_return = ((final_value / 1_000_000) ** (1/years) - 1) * 100 if years > 0 else 0

    # イクイティカーブ（日経のみ、ポートフォリオは週次で近似）
    eq_dates = nikkei.iloc[::5].index
    equity_curve = [{"date": str(d.date()), "nikkei": round(float(nikkei.loc[d]) / n_start * 1_000_000, 0)}
                    for d in eq_dates]

    summary = {
        "start": start, "end": end, "top_n": top_n, "rebalance": rebalance,
        "initial_capital": 1_000_000,
        "final_value": round(final_value, 0),
        "total_return_pct": round(total_return, 2),
        "annual_return_pct": round(annual_return, 2),
        "nikkei_return_pct": round(nikkei_return, 2),
        "alpha_pct": round(total_return - nikkei_return, 2),
        "num_rebalances": len(trades_log),
    }

    result = {
        "summary": summary,
        "equity_curve": equity_curve,
        "weekly_holdings": weekly_holdings[-52:],
        "trades": trades_log[-20:],
    }

    out_dir = Path("backtest")
    out_dir.mkdir(exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    for fname in [f"results_{today}.json", "latest.json"]:
        with open(out_dir / fname, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # 名前マッピング: results/latest.json から取得
    name_map = {}
    try:
        latest_path = out_dir.parent / "results/latest.json"
        if latest_path.exists():
            latest_data = json.loads(latest_path.read_text())
            results_list = latest_data.get("results", [])
            for r in results_list:
                code = (r.get("ticker", "") or r.get("code", "")).replace(".T", "")
                if code and r.get("name"):
                    name_map[code] = r["name"]
    except Exception:
        pass

    # weekly_holdingsのスコアマップ（最後52週）
    weekly_holdings_score_map = {}
    for wh in weekly_holdings:
        weekly_holdings_score_map[wh["date"]] = {
            t.replace(".T", ""): s for t, s in wh.get("scores", {}).items()
        }

    # holdings構築
    holdings_list = []
    for trade in trades_log:
        date = trade["date"]
        tickers = trade.get("top_n", [])
        scores_for_date = weekly_holdings_score_map.get(date, {})
        stocks = []
        for ticker in tickers:
            code = ticker.replace(".T", "")
            stocks.append({
                "code": code,
                "name": name_map.get(code, code),
                "score": scores_for_date.get(code),
            })
        holdings_list.append({"date": date, "stocks": stocks})

    # weekly（100スタート正規化）
    weekly_series = [
        {"date": t["date"], "value": round(t["portfolio_value"] / 1_000_000 * 100, 2)}
        for t in trades_log
    ]

    # nikkei_weekly（100スタート正規化）
    nikkei_weekly_series = [
        {"date": e["date"], "value": round(e["nikkei"] / 1_000_000 * 100, 2)}
        for e in equity_curve
        if e.get("nikkei") is not None
    ]

    # 月次リターン計算
    monthly_list = []
    try:
        if len(trades_log) >= 2:
            pv = {t["date"]: t["portfolio_value"] for t in trades_log}
            series_m = pd.Series(pv)
            series_m.index = pd.to_datetime(series_m.index)
            series_m = series_m.sort_index()
            monthly_end = series_m.resample("ME").last().dropna()
            if len(monthly_end) >= 2:
                monthly_returns = monthly_end.pct_change().dropna() * 100
                monthly_list = [
                    {"month": str(d)[:7], "return_pct": round(float(r), 2)}
                    for d, r in zip(monthly_returns.index, monthly_returns.values)
                ]
    except Exception:
        pass

    # タイムシリーズキャッシュ（全トレード保存・新形式）
    timeseries_cache = {
        "params": {"top_n": top_n, "rebalance": rebalance},
        "cached_at": datetime.now().isoformat(),
        "summary": summary,
        "weekly": weekly_series,
        "monthly": monthly_list,
        "nikkei_weekly": nikkei_weekly_series,
        "holdings": holdings_list,
        "all_trades": trades_log,  # 後続処理用
        "equity_curve": equity_curve,
        "nikkei_start": n_start,
        "initial_capital": 1_000_000,
    }
    with open(out_dir / "timeseries_cache.json", "w") as f:
        json.dump(timeseries_cache, f, ensure_ascii=False, indent=2)

    print("\n=== バックテスト結果 ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=(datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d"))
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--rebalance", default="daily", choices=["daily", "weekly", "biweekly", "monthly"])
    args = parser.parse_args()
    run_backtest(args.start, args.end, args.top_n, args.rebalance)
