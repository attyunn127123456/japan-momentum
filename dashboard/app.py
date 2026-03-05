"""FastAPI dashboard for Japan Momentum Screener"""
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

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

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
