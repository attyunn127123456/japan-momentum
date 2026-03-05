"""
保有銘柄のモメンタム劣化を日次チェックして売買シグナルを出す。

使い方:
  python3 daily_signal.py --holdings 5803,7011,5401
  python3 daily_signal.py  # portfolio.json から自動読み込み
"""
import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from jquants import get_daily_quotes_code
from momentum import calculate_momentum_score, calculate_rs_score

API_KEY = os.environ.get("JQUANTS_API_KEY", "cph3PdiF8zxH9GxClcFfShcJdSUzuNpV9ho_zMPm4a8")
PORTFOLIO_PATH = Path("portfolio.json")
SIGNAL_LOG_PATH = Path("data/signal_history.json")


# ── ポートフォリオ管理 ──────────────────────────────────────────

def load_portfolio() -> dict:
    """portfolio.json から保有銘柄を読む。なければ空。"""
    if PORTFOLIO_PATH.exists():
        return json.loads(PORTFOLIO_PATH.read_text())
    return {"holdings": [], "updated": ""}


def save_portfolio(holdings: list[str]):
    PORTFOLIO_PATH.write_text(json.dumps(
        {"holdings": holdings, "updated": datetime.now().isoformat()},
        ensure_ascii=False, indent=2
    ))


# ── 指数データ ──────────────────────────────────────────────────

def get_index_prices(start: str, end: str) -> pd.Series:
    """TOPIX or 日経をフォールバック付きで取得"""
    try:
        all_data = []
        params = {"from": start, "to": end}
        while True:
            r = requests.get(
                "https://api.jquants.com/v2/indices/bars/daily",
                headers={"x-api-key": API_KEY},
                params={**params, "code": "0028"},  # TOPIX
            )
            if not r.ok:
                break
            d = r.json()
            all_data.extend(d.get("data", []))
            if not d.get("pagination_key"):
                break
            params["pagination_key"] = d["pagination_key"]
        if all_data:
            df = pd.DataFrame(all_data)
            df["Date"] = pd.to_datetime(df["Date"])
            return df.set_index("Date").sort_index()["C"].dropna()
    except Exception:
        pass
    # フォールバック
    import yfinance as yf
    df = yf.download("^N225", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].dropna()


# ── シグナル判定 ────────────────────────────────────────────────

def analyze_holding(code: str, nikkei: pd.Series) -> dict:
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")

    df = get_daily_quotes_code(code, start, end)
    if df is None or len(df) < 30:
        return {"code": code, "error": "データ不足"}

    prices = df["AdjC"].dropna() if "AdjC" in df else df["C"].dropna()
    volumes = df["AdjVo"].dropna() if "AdjVo" in df else df["Vo"].dropna()
    n = nikkei.reindex(prices.index, method="ffill").dropna()

    # 今日のスコア
    scores_today = calculate_momentum_score(prices, volumes, n)
    # 昨日のスコア（1日前のデータで計算）
    scores_yesterday = calculate_momentum_score(prices.iloc[:-1], volumes.iloc[:-1], n.iloc[:-1])
    # 5日前のスコア
    scores_5d = calculate_momentum_score(prices.iloc[:-5], volumes.iloc[:-5], n.iloc[:-5]) if len(prices) > 10 else None

    score_today = scores_today.get("total", 0)
    score_yesterday = scores_yesterday.get("total", 0)
    score_5d = scores_5d.get("total", 0) if scores_5d else score_today
    score_delta_1d = score_today - score_yesterday
    score_delta_5d = score_today - score_5d

    # 対日経パフォーマンス（直近3・5日）
    def rel_perf(days: int) -> float:
        if len(prices) < days + 1 or len(n) < days + 1:
            return 0.0
        stock_ret = (prices.iloc[-1] / prices.iloc[-days] - 1) * 100
        nikkei_ret = (n.iloc[-1] / n.iloc[-days] - 1) * 100
        return round(stock_ret - nikkei_ret, 2)

    rel_3d = rel_perf(3)
    rel_5d = rel_perf(5)

    # 出来高トレンド変化
    vol_today = scores_today.get("volume_acceleration", 0)
    vol_yesterday = scores_yesterday.get("volume_acceleration", 0)
    vol_delta = vol_today - vol_yesterday

    # 50日MA チェック
    ma50_broken = len(prices) >= 50 and float(prices.iloc[-1]) < float(prices.iloc[-50:].mean())

    # ── シグナル判定 ──
    signals = []
    if score_delta_1d <= -15:
        signals.append(("🔴", f"スコア急落 {score_yesterday:.0f}→{score_today:.0f} ({score_delta_1d:+.0f})"))
    elif score_delta_1d <= -8:
        signals.append(("🟡", f"スコア低下 {score_yesterday:.0f}→{score_today:.0f} ({score_delta_1d:+.0f})"))

    if rel_3d <= -3.0:
        signals.append(("🔴", f"対日経3日: {rel_3d:+.1f}%"))
    elif rel_3d <= -1.5:
        signals.append(("🟡", f"対日経3日: {rel_3d:+.1f}%"))
    elif rel_3d >= 1.0:
        signals.append(("✅", f"対日経3日: {rel_3d:+.1f}%"))

    if vol_delta <= -0.33 and vol_today < 0.33:
        signals.append(("🟡", "出来高が細ってきた"))

    if ma50_broken:
        signals.append(("🔴", "50日MA割れ → トレンド崩壊"))

    if not signals:
        if score_today >= 65 and rel_3d >= 0:
            signals.append(("✅", "モメンタム継続"))
        else:
            signals.append(("🟢", "異常なし"))

    # 総合判定
    reds = sum(1 for s, _ in signals if s == "🔴")
    yellows = sum(1 for s, _ in signals if s == "🟡")
    if reds >= 2 or ma50_broken:
        verdict = "SELL"
    elif reds == 1 or yellows >= 2:
        verdict = "WATCH"
    else:
        verdict = "HOLD"

    return {
        "code": code,
        "score_today": round(score_today, 1),
        "score_yesterday": round(score_yesterday, 1),
        "score_delta_1d": round(score_delta_1d, 1),
        "score_delta_5d": round(score_delta_5d, 1),
        "rel_3d": rel_3d,
        "rel_5d": rel_5d,
        "vol_acceleration": round(vol_today, 2),
        "ma50_broken": ma50_broken,
        "price": round(float(prices.iloc[-1]), 1),
        "signals": signals,
        "verdict": verdict,
    }


# ── フォーマット ────────────────────────────────────────────────

VERDICT_EMOJI = {"SELL": "🔴", "WATCH": "🟡", "HOLD": "✅"}

def format_report(results: list[dict], date: str) -> str:
    lines = [f"📊 **保有銘柄シグナル {date}**\n"]
    sells, watches, holds = [], [], []
    for r in results:
        v = r.get("verdict", "HOLD")
        if v == "SELL": sells.append(r)
        elif v == "WATCH": watches.append(r)
        else: holds.append(r)

    for group, label in [(sells, "売り検討"), (watches, "要注意"), (holds, "継続保有")]:
        if not group:
            continue
        emoji = {"売り検討": "🔴", "要注意": "🟡", "継続保有": "✅"}[label]
        lines.append(f"{emoji} **{label}**")
        for r in group:
            if "error" in r:
                lines.append(f"  {r['code']}: ❌ {r['error']}")
                continue
            delta = r['score_delta_1d']
            lines.append(
                f"  **{r['code']}** スコア: {r['score_today']:.0f} ({delta:+.0f}d) | "
                f"対日経3日: {r['rel_3d']:+.1f}% | ¥{r['price']:,.0f}"
            )
            for sig_emoji, sig_text in r["signals"]:
                lines.append(f"    {sig_emoji} {sig_text}")
        lines.append("")

    # 乗り換え候補（最新スクリーニング結果から）
    latest_path = Path("results/latest.json")
    if latest_path.exists():
        latest = json.loads(latest_path.read_text())
        holding_codes = {r["code"] for r in results}
        candidates = [x for x in latest.get("top", [])[:5] if x["code"] not in holding_codes]
        if candidates:
            lines.append("💡 **乗り換え候補（最新スクリーニング上位）**")
            for c in candidates[:3]:
                lines.append(f"  {c['code']} [{c.get('sector','')}] スコア:{c['score']:.0f} | 中期:{c['return_5_25d']:+.1f}%")

    return "\n".join(lines)


# ── メイン ──────────────────────────────────────────────────────

def run_signal_check(holdings: list[str]) -> list[dict]:
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    print(f"指数データ取得中...")
    nikkei = get_index_prices(start, end)

    print(f"{len(holdings)}銘柄チェック中...")
    results = []
    for code in holdings:
        print(f"  {code}...", end=" ", flush=True)
        r = analyze_holding(code, nikkei)
        results.append(r)
        print(r.get("verdict", "ERROR"))

    # シグナル履歴保存
    SIGNAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(SIGNAL_LOG_PATH.read_text()) if SIGNAL_LOG_PATH.exists() else []
    history.append({
        "date": datetime.now().isoformat(),
        "results": results,
    })
    SIGNAL_LOG_PATH.write_text(json.dumps(history[-30:], ensure_ascii=False, indent=2))  # 30日分保持

    today = datetime.now().strftime("%Y-%m-%d")
    report = format_report(results, today)
    print("\n" + report)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdings", help="銘柄コードをカンマ区切りで指定 例: 5803,7011,5401")
    args = parser.parse_args()

    if args.holdings:
        codes = [c.strip() for c in args.holdings.split(",")]
        save_portfolio(codes)
    else:
        portfolio = load_portfolio()
        codes = portfolio.get("holdings", [])

    if not codes:
        print("保有銘柄を指定してください: --holdings 5803,7011")
        exit(1)

    run_signal_check(codes)
