"""
Layer 1 レジームシグナル - テスト・検証スクリプト

テストケース:
  1. 2020年コロナ暴落 (2020-03-13) → crash を期待
  2. 2023年AI相場   (2023-06-30) → bull_trend を期待
  3. 2022年弱気相場  (2022-10-14) → bear_trend を期待

Usage:
  python3 test_regime_signal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from market_regime import detect_regime, load_prices_dict
from backtest import get_nikkei_history

DATA_DIR = PROJECT_DIR / "data"
RESULT_PATH = PROJECT_DIR / "backtest" / "regime_signal_test.json"

UNIVERSE_PATH = DATA_DIR / "universe_cache_500.json"
FINS_PATH = DATA_DIR / "fundamentals" / "fins_summary.parquet"
INVESTOR_PATH = DATA_DIR / "fundamentals" / "investor_types.parquet"


TEST_CASES = [
    {
        "label": "2020年コロナ暴落",
        "date": "2020-03-13",
        "expected": "crash",
        "note": "日経平均が1週間で-15%超の急落局面",
    },
    {
        "label": "2023年AI相場",
        "date": "2023-06-30",
        "expected": "bull_trend",
        "note": "ChatGPTブーム・日経平均が年初から+25%超",
    },
    {
        "label": "2022年弱気相場",
        "date": "2022-10-14",
        "expected": "bear_trend",
        "note": "米金利上昇・円安・インフレによる調整局面",
    },
]


def run_test(test_case: dict, prices: dict, nikkei: pd.Series, investor, fins) -> dict:
    date = test_case["date"]
    expected = test_case["expected"]

    result = detect_regime(prices, nikkei, investor, fins, date)
    actual = result["regime"]
    passed = actual == expected

    print(
        f"  {'✅' if passed else '❌'} [{test_case['label']}] "
        f"期待={expected}  実際={actual}  スコア={result['score']:.2f}"
    )
    if not passed:
        print(f"     シグナル: {result['signals']}")

    return {
        "label": test_case["label"],
        "date": date,
        "expected": expected,
        "actual": actual,
        "passed": passed,
        "score": result["score"],
        "signals": result["signals"],
        "note": test_case.get("note", ""),
    }


def main():
    import json as _json

    print("\n" + "=" * 60)
    print("  Layer1 レジームシグナル テスト")
    print("=" * 60)

    # データロード（全テストで共有する広い期間）
    print("\n[INFO] データロード中 (2019-01-01 〜 2024-01-01)...")

    with open(UNIVERSE_PATH) as f:
        universe = _json.load(f)
    tickers = universe.get("tickers", [])

    fins = pd.read_parquet(FINS_PATH)
    investor = pd.read_parquet(INVESTOR_PATH)

    nikkei = get_nikkei_history("2019-01-01", "2024-01-01")
    prices = load_prices_dict(tickers, start="2019-01-01", end="2024-01-01")
    print(f"[INFO] {len(prices)}銘柄ロード完了\n")

    # テスト実行
    results = []
    for tc in TEST_CASES:
        r = run_test(tc, prices, nikkei, investor, fins)
        results.append(r)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n結果: {passed}/{total} passed")

    # 保存
    RESULT_PATH.parent.mkdir(exist_ok=True)
    summary = {
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 3) if total > 0 else 0,
        "cases": results,
    }
    with open(RESULT_PATH, "w") as f:
        _json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 結果を {RESULT_PATH} に保存しました\n")

    # 失敗ケースの詳細分析
    failed = [r for r in results if not r["passed"]]
    if failed:
        print("\n⚠️  失敗ケースの分析:")
        for r in failed:
            print(f"\n  [{r['label']}] {r['date']}")
            sigs = r["signals"]
            print(f"    mom5={sigs['mom5']:+.2%}, mom20={sigs['mom20']:+.2%}, mom60={sigs['mom60']:+.2%}")
            print(f"    breadth={sigs['breadth']:+.3f}, foreign_flow={sigs['foreign_flow']:+.3f}")
            print(f"    期待={r['expected']}, 実際={r['actual']}")
        print("\n  ※ 判定ルールのしきい値調整や日付のずれが原因の可能性あり")

    return summary


if __name__ == "__main__":
    main()
