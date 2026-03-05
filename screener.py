"""Daily momentum screener using J-Quants V2 API"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from jquants import get_master, get_daily_quotes_date, get_daily_quotes_code
from momentum import calculate_momentum_score, apply_filters

RESULTS_DIR = Path("results")


def get_nikkei(start: str, end: str) -> pd.Series:
    """日経225を個別銘柄として取得（コード: 0000）"""
    # J-Quantsの指数エンドポイントを使う
    import requests, os
    API_KEY = os.environ.get("JQUANTS_API_KEY", "cph3PdiF8zxH9GxClcFfShcJdSUzuNpV9ho_zMPm4a8")
    all_data = []
    params = {"code": "0000", "from": start, "to": end}  # TOPIXは0028、日経は別途
    # 日経225は indices/bars/daily で取得
    while True:
        r = requests.get(
            "https://api.jquants.com/v2/indices/bars/daily",
            headers={"x-api-key": API_KEY},
            params=params
        )
        if not r.ok:
            break
        d = r.json()
        all_data.extend(d.get("data", []))
        pk = d.get("pagination_key")
        if not pk:
            break
        params = {**params, "pagination_key": pk}

    if not all_data:
        # フォールバック: yfinance
        import yfinance as yf
        df = yf.download("^N225", start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df["Close"].dropna()

    df = pd.DataFrame(all_data)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    # 指数の終値カラム
    close_col = "C" if "C" in df.columns else "Close"
    return df[close_col].dropna()


def score_ticker(code: str, start: str, end: str, nikkei: pd.Series, market_cap: float) -> dict | None:
    try:
        df = get_daily_quotes_code(code, start, end)
    except Exception as e:
        return None

    if df is None or len(df) < 60:
        return None

    prices = df["AdjC"].dropna() if "AdjC" in df.columns else df.get("C", pd.Series()).dropna()
    volumes = df["AdjVo"].dropna() if "AdjVo" in df.columns else df.get("Vo", pd.Series()).dropna()

    if len(prices) < 60:
        return None

    # Align nikkei to same dates
    n_aligned = nikkei.reindex(prices.index, method="ffill").dropna()

    passes, reason = apply_filters(code, prices, volumes, market_cap)
    if not passes:
        return None

    scores = calculate_momentum_score(prices, volumes, n_aligned)
    if not scores["valid"]:
        return None

    return {
        "ticker": f"{code}.T",
        "code": code,
        "score": scores["total"],
        "rs_score": round(scores["rs_score"], 2),
        "return_5_25d": round(scores["return_5_25d"], 2),
        "volume_acceleration": round(scores["volume_acceleration"], 2),
        "green_day_ratio": round(scores["green_day_ratio"], 2),
        "rs_acceleration": round(scores["rs_acceleration"], 2),
        "price": round(float(prices.iloc[-1]), 2),
        "market_cap": market_cap,
    }


def run_screener(top_n: int = 20, market: str = "プライム") -> list[dict]:
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")  # 余裕を持って

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 銘柄マスタ取得中...")
    master = get_master()

    # プライム市場のみ
    if market:
        master = master[master["MktNm"] == market]

    # 時価総額フィルタ用にセクター情報を保持（マスタに時価総額はないので後でスキップ）
    codes = master["Code"].tolist()
    sector_map = dict(zip(master["Code"], master["S17Nm"]))
    print(f"  対象銘柄数: {len(codes)}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 日経225取得中...")
    nikkei = get_nikkei(start, end)
    print(f"  日経データ: {len(nikkei)}日")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] スコア計算中...")
    results = []
    with ThreadPoolExecutor(max_workers=10) as exe:
        futures = {exe.submit(score_ticker, c, start, end, nikkei, 0): c for c in codes}
        done = 0
        for f in as_completed(futures):
            done += 1
            r = f.result()
            if r:
                code = r["code"]
                r["sector"] = sector_map.get(code, "")
                results.append(r)
            if done % 100 == 0:
                print(f"  {done}/{len(codes)} 処理済, {len(results)} 有効")

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:top_n]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    for fname in [f"{today}.json", "latest.json"]:
        with open(RESULTS_DIR / fname, "w") as f:
            json.dump({"date": today, "results": results, "top": top}, f, ensure_ascii=False, indent=2)

    print(f"\n=== TOP {top_n} モメンタム銘柄 ({today}) ===")
    for i, r in enumerate(top, 1):
        print(f"{i:2}. {r['code']:6} [{r['sector'][:8]}] スコア:{r['score']:5.1f} | "
              f"RS:{r['rs_score']:+.1f} | 中期:{r['return_5_25d']:+.1f}% | 出来高:{r['volume_acceleration']:.2f}")

    return top


if __name__ == "__main__":
    run_screener()
