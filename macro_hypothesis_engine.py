"""
マクロ仮説エンジン - ファンダデータから投資テーマ仮説を生成
"""
import json
import os
import httpx
import pandas as pd
from pathlib import Path
from datetime import datetime
from fetch_fundamentals import fetch_fins_summary

BASE = Path(__file__).parent
OUTPUT = BASE / "backtest/macro_hypotheses.json"

def get_openrouter_key():
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        secret_path = BASE / ".secrets/accounts.json"
        if secret_path.exists():
            d = json.loads(secret_path.read_text())
            key = d.get("openrouter", {}).get("api_key", "")
    return key

def aggregate_by_sector(df: pd.DataFrame) -> dict:
    """業種別の財務指標を集計"""
    # fins/summaryのカラム: Sales, OP, NP, Eq, EPS, FSales, FOP, FNP, FEPS
    df = df.copy()
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df["OP"] = pd.to_numeric(df["OP"], errors="coerce")
    df["NP"] = pd.to_numeric(df["NP"], errors="coerce")
    df["Eq"] = pd.to_numeric(df["Eq"], errors="coerce")
    df["FSales"] = pd.to_numeric(df["FSales"], errors="coerce")
    
    # 直近決算のみ（DiscDateが最新）
    df["DiscDate"] = pd.to_datetime(df["DiscDate"], errors="coerce")
    latest = df.sort_values("DiscDate").groupby("Code").last().reset_index()
    
    # 基本指標計算
    latest["op_margin"] = (latest["OP"] / latest["Sales"] * 100).round(2)
    latest["roe"] = (latest["NP"] / latest["Eq"] * 100).round(2)
    
    # 全体サマリー
    summary = {
        "total_companies": len(latest),
        "median_op_margin": float(latest["op_margin"].median()) if not latest["op_margin"].empty else 0,
        "median_roe": float(latest["roe"].median()) if not latest["roe"].empty else 0,
        "high_roe_count": int((latest["roe"] > 15).sum()),
        "high_growth_count": int((latest["FSales"] > latest["Sales"] * 1.1).sum()),
    }
    return summary

def generate_hypotheses_with_llm(summary: dict) -> list:
    """OpenRouter経由でOpusに仮説生成させる"""
    key = get_openrouter_key()
    if not key:
        # APIキーなし時はサンプル仮説を返す
        return [
            {
                "id": "h_sample_001",
                "theme": "AIインフラ投資継続でSIerが恩恵",
                "logic": "生成AI導入が加速する中、日本企業のDX投資需要が高まり、ITサービス企業の売上・利益が向上する見込み",
                "target_conditions": {"min_roe": 10, "min_op_margin": 5, "sector_hint": "情報・通信"},
                "confidence": 0.75,
                "created_at": datetime.now().strftime("%Y-%m-%d")
            },
            {
                "id": "h_sample_002", 
                "theme": "円安継続で輸出製造業の利益拡大",
                "logic": "構造的な円安環境下で、海外売上比率が高い製造業は為替差益を享受し利益率が改善",
                "target_conditions": {"min_roe": 8, "min_op_margin": 6},
                "confidence": 0.65,
                "created_at": datetime.now().strftime("%Y-%m-%d")
            },
            {
                "id": "h_sample_003",
                "theme": "成果報酬型SaaSはAI普及で効率化恩恵",
                "logic": "LLMによる業務自動化で成果ベース課金モデルのSaaS企業は内部コスト削減しつつ顧客価値は向上、利益率が改善",
                "target_conditions": {"min_roe": 12, "min_op_margin": 8, "sector_hint": "情報・通信"},
                "confidence": 0.80,
                "created_at": datetime.now().strftime("%Y-%m-%d")
            }
        ]
    
    prompt = f"""あなたはトップクラスのファンダメンタル投資アナリストです。
以下の日本株市場の最新財務データサマリーを基に、今後6〜12ヶ月で注目すべき投資テーマ・仮説を3〜5個生成してください。

## 現在の市場データ（最新決算集計）
- 上場企業数: {summary['total_companies']}社
- 営業利益率中央値: {summary['median_op_margin']:.1f}%
- ROE中央値: {summary['median_roe']:.1f}%
- ROE>15%の企業数: {summary['high_roe_count']}社
- 来期増収予想企業数: {summary['high_growth_count']}社

## 現在の市場環境（2026年3月）
- 日本銀行が段階的利上げを継続中
- 生成AI・エージェントAIの急速な普及
- 円相場は不安定（米関税政策の影響）
- 東証PBR改善要請が継続中

## 出力形式（JSON配列）
```json
[
  {{
    "id": "h_001",
    "theme": "（一言で表すテーマ）",
    "logic": "（なぜこのテーマが有望か、逆張り的な洞察も含めて200字以内）",
    "target_conditions": {{
      "min_roe": 10,
      "min_op_margin": 5,
      "sector_hint": "（業種ヒント、省略可）"
    }},
    "confidence": 0.75
  }}
]
```

JSONのみを出力してください。"""

    try:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "anthropic/claude-opus-4-6", "messages": [{"role": "user", "content": prompt}]},
            timeout=120
        )
        content = resp.json()["choices"][0]["message"]["content"]
        # JSON抽出
        import re
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            hypotheses = json.loads(match.group())
            for h in hypotheses:
                h["created_at"] = datetime.now().strftime("%Y-%m-%d")
            return hypotheses
    except Exception as e:
        print(f"LLM呼び出しエラー: {e}")
    
    return []

def run():
    print("マクロ仮説エンジン起動...")
    
    # 既存キャッシュがあれば使う
    from jquants import get_master
    try:
        master = get_master()
        codes = master["Code"].tolist() if not master.empty else []
    except:
        codes = []
    
    if not codes:
        print("銘柄マスター取得失敗、サンプルデータで続行")
        summary = {"total_companies": 3800, "median_op_margin": 5.2, "median_roe": 7.8, "high_roe_count": 450, "high_growth_count": 1200}
    else:
        print(f"{len(codes)}銘柄のファンダデータ取得中...")
        df = fetch_fins_summary(codes[:500])  # コスト抑制のため上位500社
        if df.empty:
            summary = {"total_companies": 3800, "median_op_margin": 5.2, "median_roe": 7.8, "high_roe_count": 450, "high_growth_count": 1200}
        else:
            summary = aggregate_by_sector(df)
    
    print(f"集計完了: {summary}")
    
    hypotheses = generate_hypotheses_with_llm(summary)
    
    output = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "market_summary": summary,
        "hypotheses": hypotheses
    }
    
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"仮説{len(hypotheses)}件保存: {OUTPUT}")
    return output

if __name__ == "__main__":
    run()
