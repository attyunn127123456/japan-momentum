"""FastAPI dashboard for Japan Momentum Screener"""
import asyncio
import json
import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

app = FastAPI()
BASE = Path(__file__).parent.parent

@app.get("/api/today")
def today():
    p = BASE / "results/latest.json"
    if not p.exists():
        return JSONResponse({"error": "no data"}, status_code=404)
    return json.loads(p.read_text())

@app.get("/api/history")
def history():
    d = BASE / "results"
    if not d.exists():
        return []
    return sorted([f.stem for f in d.glob("????-??-??.json")], reverse=True)

@app.get("/api/results/{date}")
def results(date: str):
    p = BASE / f"results/{date}.json"
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return json.loads(p.read_text())

@app.get("/api/backtest")
def backtest():
    p = BASE / "backtest/latest.json"
    if not p.exists():
        return JSONResponse({"error": "no backtest data"}, status_code=404)
    return json.loads(p.read_text())

@app.get("/api/backtest/optimize")
def backtest_optimize():
    p = BASE / "backtest/optimize_latest.json"
    if not p.exists():
        return JSONResponse({"error": "no optimize data"}, status_code=404)
    return json.loads(p.read_text())

@app.get("/api/backtest/hypothesis")
def backtest_hypothesis():
    p = BASE / "backtest/hypothesis_queue.json"
    if not p.exists():
        return JSONResponse({"error": "no hypothesis data"}, status_code=404)
    return json.loads(p.read_text())

@app.get("/api/signals")
def signals():
    p = BASE / "data/signal_history.json"
    if not p.exists():
        return JSONResponse({"error": "no signal data"}, status_code=404)
    history = json.loads(p.read_text())
    return history[-1] if history else JSONResponse({"error": "empty"}, status_code=404)

@app.get("/api/signals/history")
def signals_history():
    p = BASE / "data/signal_history.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())

@app.get("/api/events")
async def events(request: Request):
    """SSE endpoint: sends 'update' when watched files change, 'ping' every 30s."""
    watch_files = [
        BASE / "backtest/evolution_log.json",
        BASE / "backtest/hypothesis_queue.json",
    ]

    def get_mtimes():
        mtimes = {}
        for p in watch_files:
            try:
                mtimes[str(p)] = os.path.getmtime(p)
            except OSError:
                mtimes[str(p)] = 0
        return mtimes

    async def generator():
        last_mtimes = get_mtimes()
        elapsed = 0
        try:
            while True:
                if await request.is_disconnected():
                    break
                await asyncio.sleep(3)
                elapsed += 3
                current_mtimes = get_mtimes()
                if current_mtimes != last_mtimes:
                    last_mtimes = current_mtimes
                    yield "data: update\n\n"
                    elapsed = 0
                elif elapsed >= 30:
                    yield "data: ping\n\n"
                    elapsed = 0
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

@app.get("/")
def index():
    r = FileResponse(Path(__file__).parent / "static/index.html")
    r.headers["Cache-Control"] = "no-store"
    return r

@app.get("/backtest")
def backtest_page():
    r = FileResponse(Path(__file__).parent / "static/backtest.html")
    r.headers["Cache-Control"] = "no-store"
    return r

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)

@app.get("/api/backtest/evolution")
def backtest_evolution():
    p = BASE / "backtest/evolution_log.json"
    if not p.exists():
        return JSONResponse({"best10": [], "all": [], "total": 0})
    return json.loads(p.read_text())
