"""FastAPI dashboard for Japan Momentum Screener"""
import json
import math
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
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


@app.get("/api/stock/{code}")
async def get_stock_info(code: str):
    """特定銘柄のモメンタム情報を返す（コード or 銘柄名で検索）"""
    signal_path = BASE / "backtest/daily_signal_output.json"

    # 銘柄名マッピング
    master_path = BASE / "data/fundamentals/equities_master.parquet"
    name_map = {}
    try:
        if master_path.exists():
            master = pd.read_parquet(master_path)
            name_col = next((c for c in ["CoName", "CompanyName", "Name"] if c in master.columns), None)
            if name_col:
                name_map = dict(zip(master["Code"].astype(str), master[name_col].astype(str)))
    except Exception:
        pass

    query = code.strip()

    # 名前検索：全角→半角変換して部分一致
    def normalize(s):
        """全角英数字→半角、大文字→小文字に変換"""
        result = []
        for c in s:
            cp = ord(c)
            if 0xFF01 <= cp <= 0xFF5E:  # 全角英数記号
                result.append(chr(cp - 0xFEE0))
            else:
                result.append(c)
        return ''.join(result).lower()

    target_code = None

    # 1. コード直接指定（数字のみ）
    # J-Quantsは4桁TSEコード末尾に"0"を付けた5桁形式 (例: 6227 → 62270)
    if query.isdigit():
        if len(query) == 4:
            target_code = query + "0"
        elif len(query) == 5:
            target_code = query
        else:
            target_code = query[-5:]  # 長すぎる場合は末尾5桁
    else:
        # 2. 名前で検索（正規化して部分一致）
        q_norm = normalize(query)
        for c, name in name_map.items():
            if q_norm in normalize(name):
                target_code = c
                break

    if not target_code:
        raise HTTPException(status_code=404, detail=f"銘柄が見つかりません: {query}")

    name = name_map.get(target_code, "不明")

    # all_scoresからスコアを取得
    score = None
    rank = None
    is_top = False
    top_score = None

    if signal_path.exists():
        sig = json.loads(signal_path.read_text())
        top_codes = [s['code'] for s in sig.get('recommended', [])]
        all_scores = sig.get('all_scores', sig.get('top20', []))  # フォールバック

        top_score = all_scores[0]['score'] if all_scores else None

        for i, s in enumerate(all_scores):
            if s['code'] == target_code:
                score = s['score']
                rank = i + 1
                is_top = target_code in top_codes
                break

    return JSONResponse(sanitize({
        "code": target_code,
        "name": name,
        "score": score,
        "rank": rank,
        "is_top": is_top,
        "in_ranking": score is not None,
        "top_score": top_score,
    }))


@app.get("/api/paper-trade/history")
def paper_trade_history():
    """ペーパートレード履歴を返す。ファイルがなければ空配列。"""
    p = BASE / "backtest/paper_trade_log.json"
    if not p.exists():
        return JSONResponse({"entries": []})
    return JSONResponse(sanitize(json.loads(p.read_text())))


@app.post("/api/paper-trade/record")
def paper_trade_record():
    """今週のTOP2銘柄をペーパートレードログに記録する。"""
    try:
        # ranking_cache.json からTOP銘柄を取得
        ranking_path = BASE / "backtest/ranking_cache.json"
        if not ranking_path.exists():
            return JSONResponse({"error": "ranking_cache.json が見つかりません"}, status_code=404)
        ranking = json.loads(ranking_path.read_text())
        top_n = ranking.get("params", {}).get("top_n", 2)
        top_codes = ranking.get("top_codes", [])[:top_n]
        if not top_codes:
            return JSONResponse({"error": "TOP銘柄が見つかりません"}, status_code=404)
        as_of = ranking.get("as_of", "")

        # 週の月曜日を week キーにする
        from datetime import datetime, timedelta
        try:
            dt = datetime.strptime(as_of, "%Y-%m-%d")
        except Exception:
            dt = datetime.today()
        monday = dt - timedelta(days=dt.weekday())
        week_key = monday.strftime("%Y-%m-%d")

        # daily_signal_output.json からエントリー価格（price）を取得
        signal_path = BASE / "backtest/daily_signal_output.json"
        entry_prices = {}
        if signal_path.exists():
            sig = json.loads(signal_path.read_text())
            all_scores = sig.get("all_scores", sig.get("top20", sig.get("recommended", [])))
            price_map = {s["code"]: s.get("price") for s in all_scores if s.get("price") is not None}
            for code in top_codes:
                if code in price_map:
                    entry_prices[code] = price_map[code]

        # 既存ログを読み込み
        log_path = BASE / "backtest/paper_trade_log.json"
        if log_path.exists():
            log = json.loads(log_path.read_text())
        else:
            log = {"entries": []}

        # 同じ週がすでに存在する場合はスキップ
        existing_weeks = [e["week"] for e in log.get("entries", [])]
        if week_key in existing_weeks:
            return JSONResponse({"message": f"week {week_key} はすでに記録済みです", "week": week_key})

        # 新エントリーを追加
        new_entry = {
            "week": week_key,
            "holdings": top_codes,
            "entry_prices": entry_prices,
            "exit_prices": {},
            "status": "open",
            "return_pct": None,
        }
        log.setdefault("entries", []).append(new_entry)
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2))

        return JSONResponse(sanitize({"message": "記録しました", "entry": new_entry}))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/backtest/is-oos-compare")
def backtest_is_oos_compare():
    """IS（学習期間）とOOS（検証期間）のエクイティカーブを返す。"""
    result = {}

    # IS: timeseries_cache.json
    ts_path = BASE / "backtest/timeseries_cache.json"
    if ts_path.exists():
        ts = json.loads(ts_path.read_text())
        weekly = ts.get("weekly", [])
        is_curve = [{"date": d["date"], "value": d["value"]} for d in weekly if "date" in d and "value" in d]
        is_sharpe = ts.get("sharpe")
        is_total = ts.get("total_return_pct")
        result["is"] = {
            "label": "IS (学習期間 2023〜)",
            "curve": is_curve,
            "sharpe": is_sharpe,
            "total_pct": is_total,
        }
    else:
        result["is"] = {"label": "IS (学習期間 2023〜)", "curve": [], "sharpe": None, "total_pct": None}

    # OOS: hypothesis_queue.json > baseline.oos_result
    q_path = BASE / "backtest/hypothesis_queue.json"
    if q_path.exists():
        try:
            q = json.loads(q_path.read_text())
            oos_result = q.get("baseline", {}).get("oos_result", {})
            oos_curve_raw = oos_result.get("equity_curve", [])
            oos_curve = [{"date": d["date"], "value": d["value"]} for d in oos_curve_raw if "date" in d and "value" in d]
            oos_sharpe = oos_result.get("sharpe")
            oos_total = oos_result.get("total_return_pct", oos_result.get("total_pct"))
            result["oos"] = {
                "label": "OOS (検証期間 2020〜)",
                "curve": oos_curve,
                "sharpe": oos_sharpe,
                "total_pct": oos_total,
            }
        except Exception:
            result["oos"] = {"label": "OOS (検証期間 2020〜)", "curve": [], "sharpe": None, "total_pct": None}
    else:
        result["oos"] = {"label": "OOS (検証期間 2020〜)", "curve": [], "sharpe": None, "total_pct": None}

    return JSONResponse(sanitize(result))


@app.get("/api/backtest/is-oos-compare")
def backtest_is_oos_compare():
    """IS（学習期間）とOOS（検証期間）のエクイティカーブを両方返す。"""
    q_path = BASE / "backtest/hypothesis_queue.json"
    ts_path = BASE / "backtest/timeseries_cache.json"
    result = {}
    # IS
    if ts_path.exists():
        ts = json.loads(ts_path.read_text())
        result["is"] = {
            "label": "IS（学習期間）",
            "curve": ts.get("weekly", []),
            "sharpe": ts.get("sharpe"),
            "total_pct": ts.get("total_return_pct"),
        }
    # OOS
    if q_path.exists():
        q = json.loads(q_path.read_text())
        baseline = q.get("baseline", {})
        oos = baseline.get("oos_result", {})
        if oos:
            result["oos"] = {
                "label": "OOS（検証期間 2020〜）",
                "curve": oos.get("equity_curve", []),
                "sharpe": oos.get("sharpe"),
                "total_pct": oos.get("total_return_pct"),
                "alpha_pct": oos.get("alpha_pct"),
            }
        result["train_sharpe"] = baseline.get("sharpe")
        result["train_total_pct"] = baseline.get("total_pct")
    return JSONResponse(sanitize(result))


@app.get("/api/paper-trade/history")
def paper_trade_history():
    p = BASE / "backtest/paper_trade_log.json"
    if not p.exists():
        return JSONResponse({"entries": []})
    return JSONResponse(sanitize(json.loads(p.read_text())))


@app.post("/api/paper-trade/record")
def paper_trade_record():
    """今週のTOP2シグナルをペーパートレードログに記録する。"""
    import datetime
    ranking_path = BASE / "backtest/ranking_cache.json"
    log_path = BASE / "backtest/paper_trade_log.json"
    if not ranking_path.exists():
        return JSONResponse({"error": "ranking_cache.json なし"}, status_code=404)
    ranking = json.loads(ranking_path.read_text())
    rankings = ranking.get("rankings", [])[:2]
    holdings = [r.get("code") or r.get("ticker", "") for r in rankings]
    entry_prices = {r.get("code") or r.get("ticker", ""): r.get("close") or r.get("price") for r in rankings}
    week = datetime.date.today().strftime("%Y-%m-%d")
    entry = {
        "week": week,
        "holdings": holdings,
        "entry_prices": entry_prices,
        "exit_prices": {},
        "status": "open",
        "return_pct": None,
    }
    log = {"entries": []}
    if log_path.exists():
        log = json.loads(log_path.read_text())
    # 同じ週の重複防止
    log["entries"] = [e for e in log["entries"] if e.get("week") != week]
    log["entries"].append(entry)
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2))
    return JSONResponse({"ok": True, "entry": entry})


# ─── マルチストラテジー拡張エンドポイント ───

@app.get("/api/portfolio")
def portfolio():
    """全ポッドの統合ポートフォリオ状況"""
    try:
        import sys
        sys.path.insert(0, str(BASE))
        from portfolio_engine import PortfolioEngine, create_default_engine
        engine = create_default_engine()
        if not engine.load_state():
            # 状態ファイルがなければデフォルト構成を返す
            pass
        return JSONResponse(sanitize(engine.summary()))
    except Exception as e:
        return JSONResponse({"error": str(e), "hint": "portfolio_engine.py が必要です"}, status_code=500)


@app.get("/api/paper_trades")
def paper_trades():
    """ペーパートレードの履歴とPnL"""
    try:
        import sys
        sys.path.insert(0, str(BASE))
        from paper_trading_engine import PaperTradingEngine
        engine = PaperTradingEngine()
        if engine.load_state():
            return JSONResponse(sanitize(engine.dashboard_data()))
        # フォールバック: 旧形式の paper_trade_log.json
        old_path = BASE / "backtest/paper_trade_log.json"
        if old_path.exists():
            return JSONResponse(sanitize(json.loads(old_path.read_text())))
        return JSONResponse({"nav": 0, "positions": [], "trade_history": [], "message": "ペーパートレード未開始"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/risk")
def risk():
    """リスクメトリクス（DD・VaR・Sharpe・相関）"""
    try:
        import sys
        sys.path.insert(0, str(BASE))
        from portfolio_engine import PortfolioEngine, create_default_engine
        engine = create_default_engine()
        engine.load_state()
        metrics = engine.compute_risk_metrics()
        correlation = engine.compute_correlation_matrix()
        alerts = engine.check_all_limits()
        return JSONResponse(sanitize({
            "portfolio": metrics,
            "correlation": correlation,
            "alerts": alerts,
            "as_of": metrics.get("as_of", ""),
        }))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/signals/all")
def signals_all():
    """全戦略のシグナル一覧"""
    result = {"strategies": []}

    # 1. Momentum シグナル（既存）
    signal_path = BASE / "backtest/daily_signal_output.json"
    if signal_path.exists():
        sig = json.loads(signal_path.read_text())
        result["strategies"].append({
            "name": "momentum",
            "display_name": "クロスセクショナルモメンタム",
            "signals": sig.get("recommended", []),
            "changes": sig.get("changes", {}),
            "as_of": sig.get("as_of", ""),
            "market_regime": sig.get("market_regime_filter", {}),
        })

    # 2. Signal Library
    lib_path = BASE / "backtest/signal_library.json"
    if lib_path.exists():
        lib = json.loads(lib_path.read_text())
        active_signals = [s for s in lib.get("signals", []) if s.get("status") == "active"]
        result["strategies"].append({
            "name": "signal_library",
            "display_name": "シグナルライブラリ",
            "signals": active_signals,
            "total_signals": len(lib.get("signals", [])),
            "active_count": len(active_signals),
        })

    # 3. Evolution Log（最新ベスト）
    evo_path = BASE / "backtest/evolution_log.json"
    if evo_path.exists():
        evo = json.loads(evo_path.read_text())
        best = evo.get("best10", [])[:3]
        result["strategies"].append({
            "name": "evolution",
            "display_name": "進化エンジン最新",
            "top_results": best,
        })

    return JSONResponse(sanitize(result))


import datetime as _dt

ARCHIVE_FILE = BASE / "backtest/archived_hypotheses.json"
BOOKMARKS_FILE = BASE / "backtest/bookmarks.json"

def _load_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except:
            pass
    return default

def _save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

@app.post("/api/archive/{hypothesis_id}")
def archive_hypothesis(hypothesis_id: str):
    archived = _load_json(ARCHIVE_FILE, [])
    if hypothesis_id not in archived:
        archived.append(hypothesis_id)
        _save_json(ARCHIVE_FILE, archived)
    return JSONResponse({"ok": True, "archived": archived})

@app.delete("/api/archive/{hypothesis_id}")
def unarchive_hypothesis(hypothesis_id: str):
    archived = _load_json(ARCHIVE_FILE, [])
    archived = [x for x in archived if x != hypothesis_id]
    _save_json(ARCHIVE_FILE, archived)
    return JSONResponse({"ok": True, "archived": archived})

@app.get("/api/archive")
def get_archived():
    return JSONResponse(_load_json(ARCHIVE_FILE, []))

@app.post("/api/bookmarks")
async def add_bookmark(request: Request):
    body = await request.json()
    bookmarks = _load_json(BOOKMARKS_FILE, [])
    # 重複チェック
    key = body.get("code") or body.get("name")
    if not any((b.get("code") or b.get("name")) == key for b in bookmarks):
        body["bookmarked_at"] = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        bookmarks.append(body)
        _save_json(BOOKMARKS_FILE, bookmarks)
    return JSONResponse({"ok": True, "total": len(bookmarks)})

@app.delete("/api/bookmarks/{key}")
def remove_bookmark(key: str):
    bookmarks = _load_json(BOOKMARKS_FILE, [])
    bookmarks = [b for b in bookmarks if (b.get("code") or b.get("name")) != key]
    _save_json(BOOKMARKS_FILE, bookmarks)
    return JSONResponse({"ok": True, "total": len(bookmarks)})

@app.get("/api/bookmarks")
def get_bookmarks():
    return JSONResponse(_load_json(BOOKMARKS_FILE, []))

@app.get("/api/hypotheses")
def hypotheses():
    p = BASE / "backtest/macro_hypotheses.json"
    if not p.exists():
        return JSONResponse({"hypotheses": [], "updated_at": None, "market_summary": {}})
    return JSONResponse(json.loads(p.read_text()))

@app.get("/api/candidates")
def candidates():
    p = BASE / "backtest/fundamental_candidates.json"
    if not p.exists():
        return JSONResponse({"candidates": [], "updated_at": None, "total": 0})
    data = json.loads(p.read_text())
    if "candidates" in data:
        data["candidates"] = sorted(data["candidates"], key=lambda x: x.get("score", 0), reverse=True)
    return JSONResponse(sanitize(data))


EVALUATED_FILE = BASE / "backtest/evaluated_hypotheses.json"

@app.get("/api/top_picks")
def top_picks():
    if not EVALUATED_FILE.exists():
        return JSONResponse({"ranked_hypotheses": [], "updated_at": None, "total": 0})
    return JSONResponse(sanitize(json.loads(EVALUATED_FILE.read_text())))

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
