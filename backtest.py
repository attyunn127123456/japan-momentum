"""
Backtest: hold top-N momentum stocks, rebalance weekly.
Compares vs Nikkei 225 buy-and-hold.

Usage:
  python backtest.py
  python backtest.py --start 2022-01-01 --end 2025-01-01 --top-n 5 --rebalance weekly
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from stocks import TICKERS, BENCHMARK
from momentum import calculate_momentum_score, apply_filters


def fetch_all_prices(tickers: list, start: str, end: str) -> dict[str, pd.Series]:
    """Fetch close prices for all tickers. Returns dict of ticker -> pd.Series."""
    print(f"Fetching price data for {len(tickers)+1} tickers ({start} to {end})...")
    all_tickers = tickers + [BENCHMARK]
    df = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=True)

    prices = {}
    close = df["Close"] if "Close" in df else df.xs("Close", axis=1, level=0)
    for t in all_tickers:
        if t in close.columns:
            s = close[t].dropna()
            if len(s) > 50:
                prices[t] = s
    return prices


def get_rebalance_dates(start: str, end: str, freq: str) -> list:
    """Generate rebalance dates (Mondays for weekly, etc.)"""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if freq == "weekly":
        dates = pd.date_range(s, e, freq="W-MON")
    elif freq == "biweekly":
        dates = pd.date_range(s, e, freq="2W-MON")
    elif freq == "monthly":
        dates = pd.date_range(s, e, freq="MS")
    else:
        dates = pd.date_range(s, e, freq="W-MON")
    return [d for d in dates if d <= e]


def score_on_date(ticker: str, prices: dict, nikkei_prices: pd.Series, date: pd.Timestamp) -> float | None:
    """Score a ticker using only data available up to `date`."""
    if ticker not in prices:
        return None
    p = prices[ticker].loc[:date]
    if BENCHMARK not in prices:
        return None
    n = nikkei_prices.loc[:date]
    if len(p) < 60:
        return None
    v = pd.Series(np.ones(len(p)))  # volume not available in bulk fetch, use proxy

    passes, _ = apply_filters(ticker, p, v, market_cap=100_000_000_000)  # skip mktcap filter in backtest
    if not passes:
        return None
    scores = calculate_momentum_score(p, v, n)
    return scores["total"] if scores["valid"] else None


def run_backtest(start: str, end: str, top_n: int, rebalance: str) -> dict:
    prices = fetch_all_prices(TICKERS, start, end)
    nikkei = prices.pop(BENCHMARK, None)
    if nikkei is None:
        raise RuntimeError("No Nikkei data")

    rebalance_dates = get_rebalance_dates(start, end, rebalance)
    trading_days = nikkei.index

    portfolio_value = 1_000_000.0
    transaction_cost = 0.001  # 0.1% each way
    holdings: dict[str, float] = {}  # ticker -> shares
    equity_curve = []
    trades_log = []
    weekly_holdings = []

    print(f"\nRunning backtest: {start} to {end}, top-{top_n}, {rebalance} rebalance")

    for i, reb_date in enumerate(rebalance_dates):
        # Find closest trading day
        future_days = trading_days[trading_days >= reb_date]
        if len(future_days) == 0:
            continue
        trade_date = future_days[0]

        # Score all tickers on this date
        scores = {}
        for t in prices:
            s = score_on_date(t, prices, nikkei, trade_date)
            if s is not None:
                scores[t] = s

        if not scores:
            continue

        # Top N by score
        top_tickers = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_n]

        # Current portfolio value
        current_value = 0.0
        for t, shares in holdings.items():
            if t in prices:
                p_series = prices[t].loc[:trade_date]
                if len(p_series) > 0:
                    current_value += shares * p_series.iloc[-1]
        if not holdings:
            current_value = portfolio_value

        # Sell all (apply cost)
        sell_cost = current_value * transaction_cost * len(holdings) if holdings else 0
        cash = current_value - sell_cost

        # Buy top N equally
        per_stock = cash / top_n
        new_holdings = {}
        buy_cost = 0.0
        for t in top_tickers:
            p_series = prices[t].loc[:trade_date]
            if len(p_series) == 0:
                continue
            price = p_series.iloc[-1]
            cost = per_stock * transaction_cost
            shares = (per_stock - cost) / price
            new_holdings[t] = shares
            buy_cost += cost

        holdings = new_holdings
        portfolio_value = cash - buy_cost

        weekly_holdings.append({
            "date": str(trade_date.date()),
            "tickers": top_tickers,
            "scores": {t: round(scores[t], 1) for t in top_tickers},
        })

        trades_log.append({
            "date": str(trade_date.date()),
            "action": "rebalance",
            "portfolio_value": round(portfolio_value, 0),
            "top_n": top_tickers,
        })

        if i % 10 == 0:
            print(f"  {trade_date.date()} | Portfolio: ¥{portfolio_value:,.0f} | Top: {top_tickers[:3]}")

    # Calculate final portfolio value
    end_ts = pd.Timestamp(end)
    final_value = 0.0
    for t, shares in holdings.items():
        if t in prices:
            p_series = prices[t].loc[:end_ts]
            if len(p_series) > 0:
                final_value += shares * p_series.iloc[-1]

    # Build equity curve (weekly)
    eq_dates = trading_days[trading_days >= pd.Timestamp(start)]
    eq_dates = eq_dates[::5]  # sample weekly

    nikkei_start = nikkei.loc[pd.Timestamp(start):].iloc[0] if len(nikkei.loc[pd.Timestamp(start):]) > 0 else 1
    equity_data = []
    for d in eq_dates:
        n_val = nikkei.loc[:d].iloc[-1] / nikkei_start * 1_000_000 if len(nikkei.loc[:d]) > 0 else 0
        equity_data.append({
            "date": str(d.date()),
            "nikkei": round(float(n_val), 0),
        })

    # Summary stats
    total_return = (final_value / 1_000_000 - 1) * 100
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365
    annual_return = ((final_value / 1_000_000) ** (1 / years) - 1) * 100 if years > 0 else 0

    nikkei_end_val = nikkei.loc[:end_ts].iloc[-1] if len(nikkei.loc[:end_ts]) > 0 else nikkei_start
    nikkei_return = (nikkei_end_val / nikkei_start - 1) * 100

    summary = {
        "start": start,
        "end": end,
        "top_n": top_n,
        "rebalance": rebalance,
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
        "equity_curve": equity_data,
        "weekly_holdings": weekly_holdings[-52:],  # last year
        "trades": trades_log[-20:],
    }

    # Save
    out_dir = Path("backtest")
    out_dir.mkdir(exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    out_path = out_dir / f"results_{today}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(out_dir / "latest.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n=== バックテスト結果 ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n結果保存: {out_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=(datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d"))
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--rebalance", default="weekly", choices=["weekly", "biweekly", "monthly"])
    args = parser.parse_args()

    run_backtest(args.start, args.end, args.top_n, args.rebalance)
