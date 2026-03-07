"""
Layer 1 相場の空気シグナル - コマンドラインランナー

Usage:
  python3 run_regime_signal.py
      → 今日（最新取引日）のレジームを表示

  python3 run_regime_signal.py --history 90
      → 直近90日のレジーム推移を表示

  python3 run_regime_signal.py --output json
      → JSON 形式で出力

  python3 run_regime_signal.py --date 2024-01-15
      → 指定日のレジームを表示
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# プロジェクトルートに合わせて sys.path 調整
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from market_regime import detect_regime, load_prices_dict
from backtest import get_nikkei_history

DATA_DIR = PROJECT_DIR / "data"
SIGNAL_HISTORY_PATH = DATA_DIR / "regime_signal_history.json"
UNIVERSE_PATH = DATA_DIR / "universe_cache_500.json"
FINS_PATH = DATA_DIR / "fundamentals" / "fins_summary.parquet"
INVESTOR_PATH = DATA_DIR / "fundamentals" / "investor_types.parquet"


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def load_universe() -> list:
    with open(UNIVERSE_PATH) as f:
        d = json.load(f)
    return d.get("tickers", [])


def load_fundamentals():
    fins = pd.read_parquet(FINS_PATH)
    investor = pd.read_parquet(INVESTOR_PATH)
    return fins, investor


def get_latest_trading_date(nikkei: pd.Series) -> str:
    return nikkei.index.max().strftime("%Y-%m-%d")


def load_signal_history() -> dict:
    if SIGNAL_HISTORY_PATH.exists():
        with open(SIGNAL_HISTORY_PATH) as f:
            return json.load(f)
    return {}


def save_signal_history(history: dict):
    with open(SIGNAL_HISTORY_PATH, "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _emoji(regime: str) -> str:
    return {
        "bull_trend": "🟢",
        "choppy": "🟡",
        "bear_trend": "🔴",
        "crash": "💥",
    }.get(regime, "⚪")


def print_regime(result: dict, verbose: bool = True):
    regime = result["regime"]
    score = result["score"]
    date = result["date"]
    strat = result["recommended_strategy"]
    sigs = result["signals"]

    print(f"\n{'='*50}")
    print(f"  {_emoji(regime)} 相場レジーム: {regime.upper()}   (スコア: {score:.2f})")
    print(f"  日付: {date}")
    print(f"{'='*50}")
    if verbose:
        print(f"\n📊 シグナル:")
        print(f"  市場幅 (Breadth)     : {sigs['breadth']:+.3f}")
        print(f"  日経5日モメンタム    : {sigs['mom5']:+.2%}")
        print(f"  日経20日モメンタム   : {sigs['mom20']:+.2%}")
        print(f"  日経60日モメンタム   : {sigs['mom60']:+.2%}")
        print(f"  外国人フロー        : {sigs['foreign_flow']:+.3f}")
        print(f"  銘柄間ボラ          : {sigs['cross_vol']:.4f}")
        print(f"  決算モメンタム      : {sigs['earnings_momentum']:.2%}")
        print(f"\n💡 推奨ストラテジー:")
        if strat.get("use_cash"):
            print(f"  ⚠️  全キャッシュ保持推奨（市場エクスポージャーを避ける）")
        else:
            print(f"  Top-N銘柄数    : {strat['top_n']}")
            print(f"  Lookback期間   : {strat['lookback']}日")
            print(f"  トレイリングSTOP: {strat['trailing_stop']:.1%}")
    print()


def run_single(date_str: str, output_fmt: str = "text") -> dict:
    """指定日のレジームを計算して返す。"""
    print(f"[INFO] データ読み込み中...", file=sys.stderr)

    tickers = load_universe()
    fins, investor = load_fundamentals()

    # データ範囲（当日より180日前から取得）
    date = pd.Timestamp(date_str)
    start = (date - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    end = date_str

    print(f"[INFO] 日経データ取得...", file=sys.stderr)
    nikkei = get_nikkei_history(start, end)

    print(f"[INFO] {len(tickers)}銘柄のOHLCVロード...", file=sys.stderr)
    prices = load_prices_dict(tickers, start=start, end=end)
    print(f"[INFO] ロード完了: {len(prices)}銘柄", file=sys.stderr)

    result = detect_regime(prices, nikkei, investor, fins, date)

    # 履歴に保存
    history = load_signal_history()
    history[result["date"]] = result
    save_signal_history(history)
    print(f"[INFO] signal_history.json に保存しました", file=sys.stderr)

    return result


def run_history(days: int, output_fmt: str = "text"):
    """直近 N 日のレジーム推移を表示する。"""
    print(f"[INFO] データ読み込み中...", file=sys.stderr)

    tickers = load_universe()
    fins, investor = load_fundamentals()

    end_date = datetime.today()
    start_date = end_date - timedelta(days=days + 90)  # モメンタム計算バッファ

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    print(f"[INFO] 日経データ取得...", file=sys.stderr)
    nikkei = get_nikkei_history(start_str, end_str)

    print(f"[INFO] {len(tickers)}銘柄のOHLCVロード...", file=sys.stderr)
    prices = load_prices_dict(tickers, start=start_str, end=end_str)
    print(f"[INFO] ロード完了: {len(prices)}銘柄", file=sys.stderr)

    # 対象日一覧（営業日ベース）
    target_start = (end_date - timedelta(days=days)).strftime("%Y-%m-%d")
    trading_days = nikkei.loc[target_start:end_str].index.tolist()

    history = load_signal_history()
    results = []

    for d in trading_days:
        d_str = d.strftime("%Y-%m-%d")
        # キャッシュがあれば再計算しない
        if d_str in history:
            results.append(history[d_str])
        else:
            r = detect_regime(prices, nikkei, investor, fins, d)
            history[d_str] = r
            results.append(r)

    save_signal_history(history)

    if output_fmt == "json":
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"  直近{days}日 レジーム推移")
        print(f"{'='*60}")
        for r in results:
            regime = r["regime"]
            score = r["score"]
            date = r["date"]
            strat = r["recommended_strategy"]
            use_cash = strat.get("use_cash", False)
            top_n = strat.get("top_n", 0)
            print(
                f"  {date}  {_emoji(regime)} {regime:<12}  "
                f"score={score:.2f}  "
                f"{'CASH' if use_cash else f'top_n={top_n}'}"
            )
        print()

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Layer1 相場レジームシグナル")
    parser.add_argument("--date", default=None, help="対象日 (YYYY-MM-DD). 省略時は最新取引日")
    parser.add_argument("--history", type=int, default=None, help="直近N日の推移を表示")
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="出力形式",
    )
    args = parser.parse_args()

    if args.history:
        run_history(args.history, output_fmt=args.output)
        return

    # 単日モード
    if args.date:
        date_str = args.date
    else:
        # 最新取引日を取得するためにまず日経データを小さく取得
        today = datetime.today()
        start_tmp = (today - timedelta(days=10)).strftime("%Y-%m-%d")
        end_tmp = today.strftime("%Y-%m-%d")
        nikkei_tmp = get_nikkei_history(start_tmp, end_tmp)
        date_str = get_latest_trading_date(nikkei_tmp)
        print(f"[INFO] 最新取引日: {date_str}", file=sys.stderr)

    result = run_single(date_str, output_fmt=args.output)

    if args.output == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_regime(result, verbose=True)


if __name__ == "__main__":
    main()
