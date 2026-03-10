"""
ファンダメンタルスクリーナー - 仮説に合致する銘柄を抽出・スコアリング
"""
import json
import math
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
HYPOTHESES_FILE = BASE / "backtest/macro_hypotheses.json"
OUTPUT = BASE / "backtest/fundamental_candidates.json"
FINS_CACHE = BASE / "data/fundamentals/fins_summary.parquet"

def load_fins_data() -> pd.DataFrame:
    if FINS_CACHE.exists():
        df = pd.read_parquet(FINS_CACHE)
        print(f"キャッシュ読み込み: {len(df)}行")
        return df
    print("fins_summaryキャッシュなし。macro_hypothesis_engine.pyを先に実行してください")
    return pd.DataFrame()

def compute_fundamental_scores(df: pd.DataFrame = None) -> list:
    if df is None:
        df = load_fins_data()
    if df.empty:
        return []
    
    df = df.copy()
    # 数値変換
    for col in ["Sales", "OP", "NP", "Eq", "EPS", "FSales", "FOP", "FNP", "FEPS"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["DiscDate"] = pd.to_datetime(df["DiscDate"], errors="coerce")
    
    # 直近2期分取得
    df_sorted = df.sort_values(["Code", "DiscDate"])
    latest = df_sorted.groupby("Code").last().reset_index()
    prev = df_sorted.groupby("Code").nth(-2).reset_index()
    
    merged = latest.merge(prev[["Code", "Sales", "NP", "EPS"]], on="Code", suffixes=("", "_prev"))
    
    # 指標計算
    merged["op_margin"] = (merged["OP"] / merged["Sales"] * 100).where(merged["Sales"] > 0)
    merged["roe"] = (merged["NP"] / merged["Eq"] * 100).where(merged["Eq"] > 0)
    merged["sales_growth_yoy"] = ((merged["Sales"] - merged["Sales_prev"]) / merged["Sales_prev"].abs() * 100).where(merged["Sales_prev"].abs() > 0)
    merged["eps_growth"] = ((merged["EPS"] - merged["EPS_prev"]) / merged["EPS_prev"].abs() * 100).where(merged["EPS_prev"].abs() > 0)
    merged["forecast_sales_growth"] = ((merged["FSales"] - merged["Sales"]) / merged["Sales"].abs() * 100).where(merged["Sales"].abs() > 0)
    
    # スコアリング（各指標を0-1に正規化して合計）
    def normalize(series, min_val=None, max_val=None):
        s = series.clip(min_val, max_val) if min_val is not None else series
        rng = s.max() - s.min()
        if rng == 0:
            return pd.Series(0.5, index=series.index)
        return (s - s.min()) / rng
    
    merged["score_roe"] = normalize(merged["roe"], -50, 50)
    merged["score_op_margin"] = normalize(merged["op_margin"], -20, 40)
    merged["score_sales_growth"] = normalize(merged["sales_growth_yoy"], -30, 50)
    merged["score_forecast"] = normalize(merged["forecast_sales_growth"], -20, 40)
    
    merged["score"] = (
        merged["score_roe"] * 0.35 +
        merged["score_op_margin"] * 0.25 +
        merged["score_sales_growth"] * 0.25 +
        merged["score_forecast"] * 0.15
    ).round(4)
    
    # 有効なデータのみ（ROEと営業利益率が計算できた銘柄）
    valid = merged[merged["roe"].notna() & merged["op_margin"].notna()].copy()
    valid = valid.sort_values("score", ascending=False)
    
    # 上位100件をJSON変換
    candidates = []
    for _, row in valid.head(100).iterrows():
        candidates.append({
            "code": str(row["Code"]),
            "disc_date": str(row["DiscDate"].date()) if pd.notna(row["DiscDate"]) else "",
            "roe": round(float(row["roe"]), 2) if pd.notna(row["roe"]) else None,
            "op_margin": round(float(row["op_margin"]), 2) if pd.notna(row["op_margin"]) else None,
            "sales_growth_yoy": round(float(row["sales_growth_yoy"]), 2) if pd.notna(row["sales_growth_yoy"]) else None,
            "eps_growth": round(float(row["eps_growth"]), 2) if pd.notna(row["eps_growth"]) else None,
            "forecast_sales_growth": round(float(row["forecast_sales_growth"]), 2) if pd.notna(row["forecast_sales_growth"]) else None,
            "score": float(row["score"]),
            "hypothesis_id": "general",
        })
    
    return candidates

def run():
    print("ファンダスクリーナー起動...")
    candidates = compute_fundamental_scores()
    
    output = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total": len(candidates),
        "candidates": candidates
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"候補{len(candidates)}件保存: {OUTPUT}")
    return candidates

if __name__ == "__main__":
    run()
