"""
HFグレード仮説評価エンジン

各仮説の候補銘柄に対して:
1. Perplexity Sonar Pro で現在株価・PER・アナリストTP取得
2. Opus で利益インパクト定量試算 → 目標株価（bear/base/bull）
3. Alpha Score = 確信度 × アップサイド × 未織り込み度 ÷ タイムライン
4. Kelly Fraction でポジションサイズ提案
出力: backtest/evaluated_hypotheses.json
"""

import json
import re
import httpx
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
HYPO_FILE  = BASE / "backtest/macro_hypotheses.json"
OUTPUT     = BASE / "backtest/evaluated_hypotheses.json"
SECRETS    = BASE / ".secrets/accounts.json"
OR_URL     = "https://openrouter.ai/api/v1/chat/completions"
MAX_STOCKS = 2  # 1仮説あたり評価する銘柄数（コスト削減）

def get_key():
    if SECRETS.exists():
        return json.loads(SECRETS.read_text())["openrouter"]["api_key"]
    return ""

def call_llm(model, messages, temperature=0.3, timeout=120):
    key = get_key()
    r = httpx.post(OR_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=timeout
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def extract_json_obj(text):
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass
    return {}

def extract_json_arr(text):
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass
    return []

# ───────────────────────────────────────────────
# STEP A: 市場データ取得（Perplexity）
# ───────────────────────────────────────────────
def fetch_market_data(code: str, name: str) -> dict:
    prompt = f"""東証上場企業「{name}（証券コード{code}）」について以下を調べて、JSONのみで返してください。

{{
  "current_price": 現在の株価（円、整数）,
  "per": 現在のPER（倍）,
  "pbr": 現在のPBR（倍）,
  "analyst_target_price": アナリストコンセンサス目標株価（円）,
  "analyst_consensus": "Buy" または "Hold" または "Sell",
  "buy_ratio_pct": Buyレーティング比率（%）,
  "price_change_1y_pct": 過去1年の株価変化率（%）
}}

数値が不明な場合はnullを入れてください。JSONのみ出力。"""

    try:
        res = call_llm("perplexity/sonar-pro", [{"role": "user", "content": prompt}])
        return extract_json_obj(res)
    except Exception as e:
        print(f"    市場データ取得失敗 {name}: {e}")
        return {}

# ───────────────────────────────────────────────
# STEP B: 利益インパクト試算（Opus）
# ───────────────────────────────────────────────
def estimate_earnings_impact(hypothesis: dict, stock: dict, market_data: dict) -> dict:
    current_price = market_data.get("current_price") or 0
    per = market_data.get("per") or 15

    prompt = f"""あなたはプロのヘッジファンドアナリストです。以下の投資仮説が実現した場合の業績インパクトを定量試算し、目標株価を算出してください。

## 投資仮説
テーマ: {hypothesis.get('theme')}
洞察: {hypothesis.get('insight')}
論拠: {hypothesis.get('logic', '')}
タイムライン: {hypothesis.get('timeline', '2年')}

## 対象銘柄
コード: {stock.get('code')} / 名前: {stock.get('name')}
現在株価: {current_price}円
現在PER: {per}倍
なぜこの仮説の受益者か: {stock.get('reason', '')}
なぜ今まだ割安か: {stock.get('current_concern', '')}

## 要求

以下をJSONで返してください（数字は必ず埋める、不確かなら幅を持たせる）:

{{
  "revenue_impact_pct": {{"bear": -5, "base": 15, "bull": 40}},
  "op_margin_impact_pp": {{"bear": 0, "base": 1.5, "bull": 4}},
  "eps_impact_pct": {{"bear": -5, "base": 20, "bull": 55}},
  "target_price": {{
    "bear": 0,
    "base": 0,
    "bull": 0,
    "basis": "想定PER × 修正後予想EPSの根拠を一言"
  }},
  "market_pricing_pct": 0.2,
  "pricing_rationale": "なぜこの織り込み度か（一言）",
  "key_catalyst": "株価が動く具体的なトリガーイベント",
  "catalyst_timeline": "2026年Q3",
  "position_size_suggestion": "2-3%",
  "stop_loss_pct": -15,
  "risk_comment": "最大のリスクシナリオ（一言）"
}}

目標株価の計算根拠を必ず明記。JSONのみ出力。"""

    try:
        res = call_llm("anthropic/claude-opus-4-6", [{"role": "user", "content": prompt}], temperature=0.4)
        return extract_json_obj(res)
    except Exception as e:
        print(f"    インパクト試算失敗 {stock.get('name')}: {e}")
        return {}

# ───────────────────────────────────────────────
# STEP C: Alpha Score 計算
# ───────────────────────────────────────────────
def parse_timeline_years(text: str) -> float:
    """タイムライン文字列から年数を推定"""
    if not text:
        return 2.0
    # 年が含まれる場合
    m = re.search(r'(\d+)\s*年', text)
    if m:
        year = int(m.group(1))
        current_year = 2026
        return max(0.5, year - current_year + 0.5)
    # 「Q1〜Q4」の場合
    if 'Q' in text:
        return 1.0
    return 2.0

def calc_alpha_score(hypothesis: dict, stock_eval: dict) -> dict:
    confidence     = hypothesis.get('confidence', 0.5)
    upside_pct     = stock_eval.get('upside_base_pct', 0)
    downside_pct   = abs(stock_eval.get('stop_loss_pct', 15))
    pricing_gap    = 1.0 - min(1.0, max(0.0, stock_eval.get('market_pricing_pct', 0.5)))
    timeline_years = parse_timeline_years(hypothesis.get('timeline', ''))

    # 年率化期待リターン（プロキシ）
    expected_annual = (upside_pct / 100 * confidence * pricing_gap) / max(0.5, timeline_years) * 100

    # Kelly fraction (half-Kelly で保守的に)
    if downside_pct > 0:
        b = (upside_pct / 100) / (downside_pct / 100)
        p = confidence
        kelly_full = max(0.0, (p * b - (1 - p)) / b)
        kelly_half = kelly_full * 0.5  # half-Kelly
    else:
        kelly_half = 0.0

    # Alpha Score: 0-1にスケール（年率20%超 → 1.0）
    alpha_score = min(1.0, expected_annual / 20.0)

    return {
        'alpha_score': round(alpha_score, 3),
        'expected_annual_return_pct': round(expected_annual, 1),
        'kelly_fraction': round(kelly_half, 3),
    }

# ───────────────────────────────────────────────
# メイン評価ループ
# ───────────────────────────────────────────────
def evaluate_stock(hypothesis: dict, stock: dict) -> dict:
    code = stock.get('code', '')
    name = stock.get('name', '')
    print(f"    📊 {code} {name}")

    # A: 市場データ
    mdata = fetch_market_data(code, name)
    current_price = mdata.get('current_price') or 0

    # B: インパクト試算
    impact = estimate_earnings_impact(hypothesis, stock, mdata)

    # アップサイド計算
    target_base = impact.get('target_price', {}).get('base', 0) or 0
    upside_base = ((target_base - current_price) / current_price * 100) if current_price > 0 and target_base > 0 else 0
    impact['upside_base_pct'] = round(upside_base, 1)

    # C: Alpha Score
    scores = calc_alpha_score(hypothesis, impact)

    return {
        **stock,
        'current_price': current_price,
        'per': mdata.get('per'),
        'pbr': mdata.get('pbr'),
        'analyst_target_price': mdata.get('analyst_target_price'),
        'analyst_consensus': mdata.get('analyst_consensus'),
        'buy_ratio_pct': mdata.get('buy_ratio_pct'),
        'price_change_1y_pct': mdata.get('price_change_1y_pct'),
        **impact,
        **scores,
    }

def evaluate_hypothesis(hypothesis: dict) -> dict:
    stocks = hypothesis.get('candidate_stocks', [])[:MAX_STOCKS]
    if not stocks:
        return {**hypothesis, 'evaluated_stocks': [], 'alpha_score': 0, 'top_pick': None}

    evaluated = []
    for s in stocks:
        ev = evaluate_stock(hypothesis, s)
        evaluated.append(ev)

    # アルファスコア順にソート
    evaluated.sort(key=lambda x: x.get('alpha_score', 0), reverse=True)
    top = evaluated[0] if evaluated else None

    return {
        **hypothesis,
        'evaluated_stocks': evaluated,
        'alpha_score': top.get('alpha_score', 0) if top else 0,
        'expected_annual_return_pct': top.get('expected_annual_return_pct', 0) if top else 0,
        'top_pick': top,
    }

def evaluate_all(hypotheses=None):
    if hypotheses is None:
        if not HYPO_FILE.exists():
            print("仮説ファイルなし")
            return
        data = json.loads(HYPO_FILE.read_text())
        hypotheses = data.get('hypotheses', [])

    if not hypotheses:
        print("仮説なし")
        return

    print(f"\n{'='*55}")
    print(f"🏦 HFグレード評価エンジン 起動 ({len(hypotheses)}仮説)")
    print(f"{'='*55}")

    ranked = []
    for i, h in enumerate(hypotheses):
        print(f"\n[{i+1}/{len(hypotheses)}] {h.get('theme', '—')}")
        ev = evaluate_hypothesis(h)
        ranked.append(ev)

    # アルファスコア順
    ranked.sort(key=lambda x: x.get('alpha_score', 0), reverse=True)
    for i, h in enumerate(ranked):
        h['rank'] = i + 1

    output = {
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'total': len(ranked),
        'ranked_hypotheses': ranked,
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    print(f"\n{'='*55}")
    print(f"✅ 評価完了 → {OUTPUT.name}")
    print(f"\n🏆 Top 3 仮説:")
    for h in ranked[:3]:
        tp = h.get('top_pick') or {}
        code = tp.get('code', '—')
        upside = tp.get('upside_base_pct', 0)
        print(f"  #{h['rank']} {h.get('theme')} | α={h.get('alpha_score',0):.2f} | 年率{h.get('expected_annual_return_pct',0):.0f}% | Top:{code} +{upside:.0f}%")
    print(f"{'='*55}\n")

    return ranked

if __name__ == "__main__":
    evaluate_all()
