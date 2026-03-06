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
    optimize.py の eval_params()（Optunaベース）で生成・キャッシュする。
    """
    import sys
    import os
    import time
    import pandas as pd
    from datetime import datetime as _dt, timedelta as _td

    cache_path = BASE / "backtest/timeseries_cache.json"

    # キャッシュ確認（24時間以内なら再利用）
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
            cached_at_str = cache.get("cached_at", "2000-01-01")
            cached_at = _dt.fromisoformat(cached_at_str)
            if _dt.now() - cached_at < _td(hours=24):
                # 新形式キャッシュ（weekly/monthly/holdingsが直接ある）
                if cache.get("weekly") or cache.get("holdings"):
                    return JSONResponse(sanitize(cache))
        except Exception:
            pass

    # キャッシュなし or 期限切れ → eval_params()で再生成
    try:
        sys.path.insert(0, str(BASE))
        from optimize import precompute, eval_params
        from backtest import get_nikkei_history, get_rebalance_dates
        from universe import get_top_liquid_tickers
        from fetch_cache import read_ohlcv

        # ベースラインパラメータ取得
        q_path = BASE / "backtest/hypothesis_queue.json"
        if not q_path.exists():
            return JSONResponse({"error": "hypothesis_queue.json not found"}, status_code=404)
        q = json.loads(q_path.read_text())
        params = q.get("baseline", {}).get("params", {})
        if not params:
            return JSONResponse({"error": "baseline params not found"}, status_code=404)

        START = "2023-01-01"
        END = _dt.now().strftime("%Y-%m-%d")
        lb = params.get("lookback", 60)
        rebalance = params.get("rebalance", "weekly")

        orig_dir = os.getcwd()
        os.chdir(str(BASE))
        try:
            # データ読み込み
            codes = get_top_liquid_tickers(2000)
            warmup = (_dt.strptime(START, "%Y-%m-%d") - _td(days=200)).strftime("%Y-%m-%d")
            prices_dict = {}
            for c in codes:
                df = read_ohlcv(c, warmup, END)
                if df is not None and not df.empty and "AdjC" in df.columns:
                    prices_dict[c] = df

            nikkei = get_nikkei_history(warmup, END)
            # precompute: prices_dict(dict), nikkei, lookbacks(list) → factor_dfs
            factor_dfs = precompute(prices_dict, nikkei, [lb])

            rebal_dates = get_rebalance_dates(warmup, END, rebalance)
            all_prices = pd.DataFrame({c: df["AdjC"] for c, df in prices_dict.items()})
            return_df = all_prices.pct_change()

            result = eval_params(params, factor_dfs, prices_dict, rebal_dates, nikkei, START, return_df)
        finally:
            os.chdir(orig_dir)

        if not result or "equity_curve" not in result:
            return JSONResponse({"error": "backtest failed or no equity_curve"}, status_code=500)

        equity_curve = result["equity_curve"]

        # 月次リターン計算
        df_eq = pd.DataFrame(equity_curve).set_index("date")
        df_eq.index = pd.to_datetime(df_eq.index)
        monthly_end = df_eq["value"].resample("ME").last()
        monthly_start = monthly_end.shift(1).fillna(0)
        monthly_return = ((monthly_end - monthly_start) / (100 + monthly_start) * 100).round(2)
        monthly = [{"month": str(d)[:7], "return_pct": float(r)} for d, r in monthly_return.items()]

        # 銘柄名マッピング
        name_map = {}
        try:
            master_path = BASE / "data/fundamentals/equities_master.parquet"
            if master_path.exists():
                master = pd.read_parquet(master_path)
                # 利用可能な名前カラムを探す
                name_col = next(
                    (c for c in ["CoName", "CompanyName", "Name", "name", "CompanyNameInEnglish"]
                     if c in master.columns),
                    None
                )
                if name_col:
                    name_map = dict(zip(master["Code"].astype(str), master[name_col].astype(str)))
        except Exception:
            pass

        # holdings（各週の保有銘柄、新しい順）
        holdings = []
        for entry in reversed(equity_curve):
            stocks = [
                {"code": code, "name": name_map.get(str(code), str(code))}
                for code in entry["holdings"]
            ]
            holdings.append({"date": entry["date"], "stocks": stocks})

        # 日経225週次累積指数（START以降、100スタート）
        nikkei_weekly = []
        try:
            nikkei_from_start = nikkei[nikkei.index >= pd.Timestamp(START)]
            nikkei_w = nikkei_from_start.resample("W-MON").last().dropna()
            if len(nikkei_w) > 0:
                base_val = nikkei_w.iloc[0]
                nikkei_weekly = [
                    {"date": str(d.date()), "value": round(float(v) / base_val * 100, 2)}
                    for d, v in nikkei_w.items()
                ]
        except Exception:
            nikkei_weekly = []

        cache_data = {
            "params": params,
            "cached_at": _dt.now().isoformat(),
            "total_return_pct": result.get("total_return_pct"),
            "weekly": [{"date": e["date"], "value": e["value"] + 100} for e in equity_curve],
            "nikkei_weekly": nikkei_weekly,
            "monthly": monthly,
            "holdings": holdings,
        }
        cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2))
        return JSONResponse(sanitize(cache_data))

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
    import sys
    sys.path.insert(0, str(BASE))
    import json as _json
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    # キャッシュ（1時間）
    cache_path = BASE / "backtest/weekly_picks_cache.json"
    if cache_path.exists():
        try:
            cache = _json.loads(cache_path.read_text())
            cached_at = datetime.fromisoformat(cache.get("cached_at", "2000-01-01"))
            if datetime.now() - cached_at < timedelta(hours=1):
                return JSONResponse(sanitize(cache))
        except Exception:
            pass

    try:
        from optimize import precompute, build_score_df
        from backtest import get_nikkei_history
        from fetch_cache import read_ohlcv
        from universe import get_top_liquid_tickers
    except Exception as e:
        return JSONResponse({"error": f"モジュールインポート失敗: {e}"}, status_code=500)

    # ベースラインパラメータ取得
    q_path = BASE / "backtest/hypothesis_queue.json"
    if not q_path.exists():
        return JSONResponse({"error": "hypothesis_queue.json が見つかりません"}, status_code=404)
    q = _json.loads(q_path.read_text())
    params = q.get("baseline", {}).get("params", {})
    lb = params.get("lookback", 60)
    top_n = params.get("top_n", 2)

    # データロード
    END = datetime.now().strftime("%Y-%m-%d")
    warmup = (datetime.now() - timedelta(days=lb * 2 + 50)).strftime("%Y-%m-%d")

    try:
        codes = get_top_liquid_tickers(2000)
    except Exception as e:
        return JSONResponse({"error": f"ユニバース取得失敗: {e}"}, status_code=500)

    prices_dict = {}
    for c in codes:
        try:
            df = read_ohlcv(c, warmup, END)
            if df is not None and not df.empty and 'AdjC' in df.columns:
                prices_dict[c] = df
        except Exception:
            pass

    if not prices_dict:
        return JSONResponse({"error": "株価データが取得できませんでした"}, status_code=500)

    try:
        nikkei = get_nikkei_history(warmup, END)
    except Exception as e:
        return JSONResponse({"error": f"日経データ取得失敗: {e}"}, status_code=500)

    # precompute
    try:
        factor_dfs = precompute(prices_dict, nikkei, [lb])
    except Exception as e:
        return JSONResponse({"error": f"precompute失敗: {e}"}, status_code=500)

    # ウェイト取得（ベースラインパラメータ全ファクター対応）
    weight_keys = ["ret", "rs", "green", "smooth", "resilience",
                   "short_momentum", "high52", "omega", "close_location",
                   "range_expand", "win_streak", "sector_momentum",
                   "overnight_return", "volume_acceleration", "higher_lows",
                   "body_strength", "vol_return_corr", "accumulation",
                   "momentum_consistency", "upside_capture", "gap_momentum",
                   "volume_confirm", "ret_skip", "cluster_boost",
                   "return_autocorr", "volume_slope", "clean_momentum"]
    weights = {k: params.get(k + "_w", 0.0) for k in weight_keys}

    try:
        score_df = build_score_df(factor_dfs, lb, weights)
    except Exception as e:
        return JSONResponse({"error": f"スコア計算失敗: {e}"}, status_code=500)

    if score_df is None or score_df.empty:
        return JSONResponse({"error": "スコア計算失敗（空のスコアDF）"}, status_code=500)

    # 最新日のスコア
    latest_date = score_df.index[-1]
    latest_scores = score_df.loc[latest_date].dropna().sort_values(ascending=False)

    # 1週前のスコア（前回推奨との比較）
    prev_date = score_df.index[-2] if len(score_df) >= 2 else latest_date
    prev_scores = score_df.loc[prev_date].dropna().sort_values(ascending=False) if prev_date != latest_date else latest_scores
    prev_top = prev_scores.nlargest(top_n).index.tolist()
    curr_top = latest_scores.nlargest(top_n).index.tolist()

    # 銘柄名マッピング
    name_map = {}
    try:
        master = pd.read_parquet(str(BASE / "data/fundamentals/equities_master.parquet"))
        name_col = next((c for c in ["CoName", "CompanyName", "Name"] if c in master.columns), None)
        if name_col:
            name_map = dict(zip(master["Code"].astype(str), master[name_col].astype(str)))
    except Exception:
        pass

    def stock_info(code):
        df = prices_dict.get(code)
        price = None
        ret5d = None
        if df is not None and not df.empty:
            try:
                price = round(float(df['AdjC'].iloc[-1]), 0)
            except Exception:
                pass
            try:
                if len(df) > 5:
                    ret5d = round(float(df['AdjC'].pct_change(5).iloc[-1] * 100), 1)
            except Exception:
                pass
        return {
            "code": str(code),
            "name": name_map.get(str(code), str(code)),
            "price": price,
            "score": round(float(latest_scores.get(code, 0)), 3),
            "ret5d_pct": ret5d,
        }

    # 上位20のランキング
    top20 = latest_scores.nlargest(20)
    ranking = []
    for rank, (code, score) in enumerate(top20.items(), 1):
        info = stock_info(code)
        info["rank"] = rank
        info["is_recommended"] = code in curr_top
        ranking.append(info)

    # 持ち替え推奨
    sell = [c for c in prev_top if c not in curr_top]
    buy = [c for c in curr_top if c not in prev_top]
    hold = [c for c in curr_top if c in prev_top]

    result = {
        "cached_at": datetime.now().isoformat(),
        "as_of": str(latest_date.date()),
        "params": {
            "lookback": lb,
            "top_n": top_n,
            "rebalance": params.get("rebalance", "weekly"),
            "total_return_pct": q.get("baseline", {}).get("total_pct", 0),
        },
        "recommended": [stock_info(c) for c in curr_top],
        "changes": {
            "buy": [stock_info(c) for c in buy],
            "sell": [stock_info(c) for c in sell],
            "hold": [stock_info(c) for c in hold],
        },
        "ranking": ranking,
    }

    try:
        cache_path.write_text(_json.dumps(result, ensure_ascii=False, default=str, indent=2))
    except Exception:
        pass

    return JSONResponse(sanitize(result))


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
