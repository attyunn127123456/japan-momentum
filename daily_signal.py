"""
売買代金Top500全銘柄のスクリーニング結果を前日比較してシグナルを出す。
screener.pyの結果(results/*.json)を2日分比較するだけ。追加APIコールなし。

使い方:
  python3 daily_signal.py
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

RESULTS_DIR = Path("results")
SIGNAL_LOG_PATH = Path("data/signal_history.json")


def load_results(date: str) -> dict[str, dict]:
    """results/YYYY-MM-DD.json を {code: row} で返す"""
    p = RESULTS_DIR / f"{date}.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    rows = data.get("results", [])
    return {(r.get("code") or r.get("ticker","").replace(".T","")): r for r in rows}


def get_available_dates() -> list:
    return sorted([f.stem for f in RESULTS_DIR.glob("????-??-??.json")], reverse=True)


def run_signal_check() -> dict:
    dates = get_available_dates()
    if len(dates) < 2:
        print("スクリーニング結果が2日分ありません。先に screener.py を2日分実行してください。")
        return {}

    today_str = dates[0]
    yesterday_str = dates[1]

    today_data = load_results(today_str)
    yesterday_data = load_results(yesterday_str)

    if not today_data:
        print(f"本日({today_str})の結果がありません")
        return {}

    print(f"シグナル計算: {yesterday_str} → {today_str} ({len(today_data)}銘柄)")

    results = []
    for code, row in today_data.items():
        prev = yesterday_data.get(code)
        score_today = row.get("score", 0)
        score_yesterday = prev.get("score", 0) if prev else score_today
        delta = score_today - score_yesterday

        # ランク変化
        rank_today = list(today_data.keys()).index(code) + 1 if code in today_data else 999
        rank_yesterday = list(yesterday_data.keys()).index(code) + 1 if code in yesterday_data else 999

        # シグナル判定
        signals = []
        if delta >= 10:
            signals.append(("🚀", f"スコア急伸 +{delta:.0f}"))
        elif delta >= 5:
            signals.append(("📈", f"スコア上昇 +{delta:.0f}"))
        elif delta <= -15:
            signals.append(("🔴", f"スコア急落 {delta:.0f}"))
        elif delta <= -8:
            signals.append(("🟡", f"スコア低下 {delta:.0f}"))

        if rank_today <= 20 and rank_yesterday > 20:
            signals.append(("⭐", "Top20に新規ランクイン"))
        elif rank_today > 20 and rank_yesterday <= 20:
            signals.append(("⬇️", "Top20から脱落"))

        vol = row.get("volume_acceleration", 0)
        prev_vol = prev.get("volume_acceleration", 0) if prev else vol
        if vol >= 0.67 and prev_vol < 0.67:
            signals.append(("💹", "出来高加速"))
        elif vol < 0.33 and prev_vol >= 0.67:
            signals.append(("📉", "出来高急減"))

        # 総合判定
        has_surge = any(e in ["🚀","📈","⭐"] for e, _ in signals)
        has_warn = any(e in ["🔴","🟡","⬇️"] for e, _ in signals)
        if delta <= -15 or (any(e=="🔴" for e,_ in signals)):
            verdict = "SELL"
        elif has_warn:
            verdict = "WATCH"
        elif has_surge:
            verdict = "BUY"
        else:
            verdict = "HOLD"

        results.append({
            "code": code,
            "ticker": row.get("ticker", f"{code}.T"),
            "sector": row.get("sector", ""),
            "score_today": round(score_today, 1),
            "score_yesterday": round(score_yesterday, 1),
            "score_delta": round(delta, 1),
            "rank_today": rank_today,
            "rank_yesterday": rank_yesterday,
            "return_5_25d": row.get("return_5_25d", 0),
            "volume_acceleration": round(vol, 2),
            "green_day_ratio": row.get("green_day_ratio", 0),
            "rs_score": row.get("rs_score", 0),
            "price": row.get("price", 0),
            "signals": signals,
            "verdict": verdict,
        })

    # スコア順でソート
    results.sort(key=lambda x: x["score_today"], reverse=True)

    output = {
        "date": today_str,
        "prev_date": yesterday_str,
        "total": len(results),
        "results": results,
        # サマリー
        "buy": [r for r in results if r["verdict"] == "BUY"],
        "sell": [r for r in results if r["verdict"] == "SELL"],
        "watch": [r for r in results if r["verdict"] == "WATCH"],
        "top20": [r for r in results if r["rank_today"] <= 20],
        "new_entries": [r for r in results if r["rank_today"] <= 20 and r["rank_yesterday"] > 20],
        "drop_outs": [r for r in results if r["rank_today"] > 20 and r["rank_yesterday"] <= 20],
    }

    # 保存
    SIGNAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(SIGNAL_LOG_PATH.read_text()) if SIGNAL_LOG_PATH.exists() else []
    # 同日分は上書き
    history = [h for h in history if h.get("date") != today_str]
    history.append(output)
    SIGNAL_LOG_PATH.write_text(json.dumps(history[-30:], ensure_ascii=False))

    # サマリー表示
    print(f"\n=== シグナルサマリー {today_str} ===")
    print(f"  🚀 急伸(BUY):  {len(output['buy'])}銘柄")
    print(f"  🔴 急落(SELL): {len(output['sell'])}銘柄")
    print(f"  🟡 要注意:     {len(output['watch'])}銘柄")
    print(f"  ⭐ Top20新規:  {len(output['new_entries'])}銘柄 → {[r['code'] for r in output['new_entries']]}")
    print(f"  ⬇️  Top20脱落:  {len(output['drop_outs'])}銘柄 → {[r['code'] for r in output['drop_outs']]}")
    print(f"\n--- Top20 ---")
    for r in output["top20"][:20]:
        delta_str = f"{r['score_delta']:+.0f}"
        rank_str = f"#{r['rank_today']}"
        if r['rank_yesterday'] != r['rank_today']:
            rank_str += f"(前{r['rank_yesterday']})"
        print(f"  {rank_str:12} {r['code']:6} [{r['sector'][:8]:8}] {r['score_today']:5.1f} ({delta_str:>4}) | {r['return_5_25d']:+.1f}%")

    return output


if __name__ == "__main__":
    run_signal_check()
