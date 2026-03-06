"""FastAPI dashboard for Japan Momentum Screener"""
import json
import math
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()
BASE = Path(__file__).parent.parent


def sanitize(obj):
    """NaN/Inf をNoneに置換してJSONシリアライズ可能にする"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(i) for i in obj]
    return obj


@app.get("/api/today")
def today():
    p = BASE / "results/latest.json"
    if not p.exists():
        return JSONResponse({"error": "no data"}, status_code=404)
    data = json.loads(p.read_text())
    # total_return_pct 降順でソート
    if isinstance(data, list):
        data = sorted(data, key=lambda x: x.get('total_return_pct', 0), reverse=True)
        return JSONResponse(sanitize({'best10': data[:10], 'all': data, 'total': len(data)}))
    if 'all' in data:
        data['all'] = sorted(data['all'], key=lambda x: x.get('total_return_pct', 0), reverse=True)
        data['best10'] = data['all'][:10]
    return JSONResponse(sanitize(data))

@app.get("/api/history")
def history():
    d = BASE / "results"
    if not d.exists():
        return JSONResponse(sanitize([]))
    return JSONResponse(sanitize(sorted([f.stem for f in d.glob("????-??-??.json")], reverse=True)))

@app.get("/api/results/{date}")
def results(date: str):
    p = BASE / f"results/{date}.json"
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    data = json.loads(p.read_text())
    # total_return_pct 降順でソート
    if isinstance(data, list):
        data = sorted(data, key=lambda x: x.get('total_return_pct', 0), reverse=True)
        return JSONResponse(sanitize({'best10': data[:10], 'all': data, 'total': len(data)}))
    if 'all' in data:
        data['all'] = sorted(data['all'], key=lambda x: x.get('total_return_pct', 0), reverse=True)
        data['best10'] = data['all'][:10]
    return JSONResponse(sanitize(data))

@app.get("/api/backtest")
def backtest():
    p = BASE / "backtest/latest.json"
    if not p.exists():
        return JSONResponse({"error": "no backtest data"}, status_code=404)
    data = json.loads(p.read_text())
    # total_return_pct 降順でソート
    if isinstance(data, list):
        data = sorted(data, key=lambda x: x.get('total_return_pct', 0), reverse=True)
        return JSONResponse(sanitize({'best10': data[:10], 'all': data, 'total': len(data)}))
    if 'all' in data:
        data['all'] = sorted(data['all'], key=lambda x: x.get('total_return_pct', 0), reverse=True)
        data['best10'] = data['all'][:10]
    return JSONResponse(sanitize(data))

@app.get("/api/backtest/optimize")
def backtest_optimize():
    p = BASE / "backtest/optimize_latest.json"
    if not p.exists():
        return JSONResponse({"error": "no optimize data"}, status_code=404)
    data = json.loads(p.read_text())
    # total_return_pct 降順でソート
    if isinstance(data, list):
        data = sorted(data, key=lambda x: x.get('total_return_pct', 0), reverse=True)
        return JSONResponse(sanitize({'best10': data[:10], 'all': data, 'total': len(data)}))
    if 'all' in data:
        data['all'] = sorted(data['all'], key=lambda x: x.get('total_return_pct', 0), reverse=True)
        data['best10'] = data['all'][:10]
    return JSONResponse(sanitize(data))

@app.get("/api/backtest/hypothesis")
def backtest_hypothesis():
    p = BASE / "backtest/hypothesis_queue.json"
    if not p.exists():
        return JSONResponse({"error": "no hypothesis data"}, status_code=404)
    data = json.loads(p.read_text())
    # total_return_pct 降順でソート
    if isinstance(data, list):
        data = sorted(data, key=lambda x: x.get('total_return_pct', 0), reverse=True)
        return JSONResponse(sanitize({'best10': data[:10], 'all': data, 'total': len(data)}))
    if 'all' in data:
        data['all'] = sorted(data['all'], key=lambda x: x.get('total_return_pct', 0), reverse=True)
        data['best10'] = data['all'][:10]
    return JSONResponse(sanitize(data))

@app.get("/api/signals")
def signals():
    # signal_library.json (new design)
    p = BASE / "backtest/signal_library.json"
    if p.exists():
        data = json.loads(p.read_text())
        return JSONResponse(sanitize(data))
    # Fallback: legacy signal_history.json
    p2 = BASE / "data/signal_history.json"
    if not p2.exists():
        return JSONResponse({"error": "no signal data"}, status_code=404)
    history = json.loads(p2.read_text())
    if not history:
        return JSONResponse({"error": "empty"}, status_code=404)
    return JSONResponse(sanitize(history[-1]))

@app.get("/api/signals/history")
def signals_history():
    p = BASE / "data/signal_history.json"
    if not p.exists():
        return JSONResponse(sanitize([]))
    data = json.loads(p.read_text())
    # total_return_pct 降順でソート
    if isinstance(data, list):
        data = sorted(data, key=lambda x: x.get('total_return_pct', 0), reverse=True)
        return JSONResponse(sanitize({'best10': data[:10], 'all': data, 'total': len(data)}))
    if 'all' in data:
        data['all'] = sorted(data['all'], key=lambda x: x.get('total_return_pct', 0), reverse=True)
        data['best10'] = data['all'][:10]
    return JSONResponse(sanitize(data))

@app.get("/api/regime")
def regime():
    p = BASE / "backtest/regime_weights.json"
    if not p.exists():
        return JSONResponse({"error": "no regime data"}, status_code=404)
    data = json.loads(p.read_text())
    # total_return_pct 降順でソート
    if isinstance(data, list):
        data = sorted(data, key=lambda x: x.get('total_return_pct', 0), reverse=True)
        return JSONResponse(sanitize({'best10': data[:10], 'all': data, 'total': len(data)}))
    if 'all' in data:
        data['all'] = sorted(data['all'], key=lambda x: x.get('total_return_pct', 0), reverse=True)
        data['best10'] = data['all'][:10]
    return JSONResponse(sanitize(data))

@app.get("/api/backtest/timeseries")
def backtest_timeseries():
    """週次ポートフォリオ時系列と月次リターンを返す。timeseries_cache.jsonを使う。"""
    import sys
    import time
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    cache_path = BASE / "backtest/timeseries_cache.json"

    def build_timeseries(raw: dict) -> dict:
        """timeseries_cache.jsonのrawデータを整形してフロント用に返す"""
        import pandas as pd

        all_trades = raw.get("all_trades", [])
        equity_curve = raw.get("equity_curve", [])  # [{date, nikkei}, ...]
        initial_capital = raw.get("initial_capital", 1_000_000)
        summary = raw.get("summary", {})

        if not all_trades:
            return {"error": "no trades data", "weekly": [], "monthly": [], "nikkei_weekly": []}

        # ポートフォリオ週次（100スタートに正規化）
        weekly = [
            {"date": t["date"], "value": round(t["portfolio_value"] / initial_capital * 100, 2)}
            for t in all_trades
        ]

        # 日経週次（100スタートに正規化）
        # equity_curve の nikkei 値は既に initial_capital スタートで正規化済み
        nikkei_weekly = []
        if equity_curve:
            nikkei_weekly = [
                {"date": e["date"], "value": round(e["nikkei"] / initial_capital * 100, 2)}
                for e in equity_curve
                if e.get("nikkei") is not None
            ]

        # 月次リターン計算
        monthly = []
        try:
            if len(all_trades) >= 2:
                portfolio_values = {t["date"]: t["portfolio_value"] for t in all_trades}
                series = pd.Series(portfolio_values)
                series.index = pd.to_datetime(series.index)
                series = series.sort_index()
                # 月末値でリサンプル
                monthly_end = series.resample("ME").last().dropna()
                if len(monthly_end) >= 2:
                    monthly_returns = monthly_end.pct_change().dropna() * 100
                    monthly = [
                        {"month": str(d)[:7], "return_pct": round(float(r), 2)}
                        for d, r in zip(monthly_returns.index, monthly_returns.values)
                    ]
        except Exception as e:
            monthly = []

        return {
            "weekly": weekly,
            "monthly": monthly,
            "nikkei_weekly": nikkei_weekly,
            "params": summary,
        }

    # キャッシュが存在すれば使う（24時間以内）
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < 86400:
            raw = json.loads(cache_path.read_text())
            return JSONResponse(sanitize(build_timeseries(raw)))

    # キャッシュなし → バックテスト実行（重い処理）
    try:
        sys.path.insert(0, str(BASE))
        from backtest import run_backtest
        from datetime import datetime as _dt, timedelta as _td

        # ベースラインパラメータ取得
        q_path = BASE / "backtest/hypothesis_queue.json"
        if not q_path.exists():
            return JSONResponse({"error": "hypothesis_queue.json not found"}, status_code=404)

        q = json.loads(q_path.read_text())
        baseline = q.get("baseline", {})
        params = baseline.get("params", {})
        if not params:
            return JSONResponse({"error": "baseline params not found"}, status_code=404)

        end = _dt.now().strftime("%Y-%m-%d")
        start = (_dt.now() - _td(days=365 * 3)).strftime("%Y-%m-%d")

        # バックテスト実行（timeseries_cache.jsonも保存される）
        import os
        orig_dir = os.getcwd()
        os.chdir(str(BASE))
        try:
            run_backtest(
                start, end,
                params.get("top_n", 5),
                params.get("rebalance", "weekly")
            )
        finally:
            os.chdir(orig_dir)

        # 生成されたキャッシュを読む
        if cache_path.exists():
            raw = json.loads(cache_path.read_text())
            return JSONResponse(sanitize(build_timeseries(raw)))
        else:
            return JSONResponse({"error": "timeseries cache not generated"}, status_code=500)

    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


@app.get("/api/backtest/evolution")
def backtest_evolution():
    p = BASE / "backtest/evolution_log.json"
    if not p.exists():
        return JSONResponse(sanitize({"best10": [], "all": [], "total": 0}))
    data = json.loads(p.read_text())

    def dedup_by_return(entries):
        """total_return_pct が小数第2位まで同じエントリーは最初の1件のみ残す"""
        seen = set()
        result = []
        for e in entries:
            key = round(e.get('total_return_pct', 0), 2)
            if key not in seen:
                seen.add(key)
                result.append(e)
        return result

    # total_return_pct 降順でソート
    if isinstance(data, list):
        data = sorted(data, key=lambda x: x.get('total_return_pct', 0), reverse=True)
        deduped = dedup_by_return(data)
        return JSONResponse(sanitize({'best10': deduped[:10], 'all': data, 'total': len(data)}))
    if 'all' in data:
        data['all'] = sorted(data['all'], key=lambda x: x.get('total_return_pct', 0), reverse=True)
        deduped = dedup_by_return(data['all'])
        data['best10'] = deduped[:10]
    return JSONResponse(sanitize(data))


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

@app.get("/")
def index():
    r = FileResponse(Path(__file__).parent / "static/index.html", media_type="text/html")
    r.headers["Cache-Control"] = "no-store"
    return r

@app.get("/backtest")
def backtest_page():
    r = FileResponse(Path(__file__).parent / "static/index.html", media_type="text/html")
    r.headers["Cache-Control"] = "no-store"
    return r

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
