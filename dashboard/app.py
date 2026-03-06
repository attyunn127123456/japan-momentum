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
    """週次ポートフォリオ時系列・月次リターン・保有銘柄を返す。
    generate_dashboard_cache.py が生成した JSON を読んで返すだけ。
    """
    cache_path = BASE / "backtest/timeseries_cache.json"
    if not cache_path.exists():
        return JSONResponse({"error": "キャッシュ未生成。generate_dashboard_cache.py を実行してください。",
                             "weekly": [], "monthly": [], "holdings": []}, status_code=404)
    return JSONResponse(sanitize(json.loads(cache_path.read_text())))


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


@app.get("/api/backtest/oos")
def backtest_oos():
    """最新OOS検証結果を返す。hypothesis_queue.json の baseline.oos_result から読む。"""
    q_path = BASE / "backtest/hypothesis_queue.json"
    if not q_path.exists():
        return JSONResponse({"error": "no hypothesis data"}, status_code=404)
    try:
        q = json.loads(q_path.read_text())
        oos = q.get("baseline", {}).get("oos_result")
        if not oos:
            return JSONResponse({"error": "OOS結果なし（まだ実行されていないか、データ不足）"}, status_code=404)
        baseline = q.get("baseline", {})
        return JSONResponse(sanitize({
            "oos_result": oos,
            "train_total_pct": baseline.get("total_pct"),
            "train_sharpe": baseline.get("sharpe"),
            "oos_start": "2020-01-01",
            "params": baseline.get("params"),
        }))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/weekly_picks")
async def get_weekly_picks():
    """週次推奨銘柄を返す。generate_dashboard_cache.py が生成した JSON を読んで返すだけ。"""
    cache_path = BASE / "backtest/weekly_picks_cache.json"
    if not cache_path.exists():
        return JSONResponse({"error": "キャッシュ未生成。generate_dashboard_cache.py を実行してください。",
                             "recommended": [], "changes": {}, "ranking": []}, status_code=404)
    return JSONResponse(sanitize(json.loads(cache_path.read_text())))


@app.get("/api/ranking")
async def get_ranking():
    """銘柄ランキングを返す。generate_dashboard_cache.py が生成した JSON を読んで返すだけ。"""
    cache_path = BASE / "backtest/ranking_cache.json"
    if not cache_path.exists():
        return JSONResponse({"error": "キャッシュ未生成。generate_dashboard_cache.py を実行してください。",
                             "rankings": []}, status_code=404)
    return JSONResponse(sanitize(json.loads(cache_path.read_text())))


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
