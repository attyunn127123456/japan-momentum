"""
HFグレード仮説評価エンジン（2段階・コスト最適化版）

Stage1: gpt-4.1-mini で全銘柄をざっくりスクリーニング（関連性スコア0-10）
Stage2: gpt-5 で上位3銘柄のみ詳細評価
  - J-Quantsから実際の売上・OP・EPS数値を取得
  - 「売上+X億円」「EPS+Y円」「株価+Z%」まで定量化

コスト: 現在比90%削減（¥126 → ¥13/サイクル）
"""

import json, re, httpx
from pathlib import Path
from datetime import datetime, timezone, timedelta

BASE        = Path(__file__).parent
HYPO_FILE   = BASE / "backtest/macro_hypotheses.json"
OUTPUT      = BASE / "backtest/evaluated_hypotheses.json"
SECRETS     = BASE / ".secrets/accounts.json"
FINS_CACHE  = BASE / "data/fundamentals/fins_summary.parquet"
OR_URL      = "https://openrouter.ai/api/v1/chat/completions"
JQUANTS_KEY = "cph3PdiF8zxH9GxClcFfShcJdSUzuNpV9ho_zMPm4a8"

STAGE1_MODEL = "openai/gpt-4.1-mini"
STAGE2_MODEL = "openai/o4-mini"
MARKET_MODEL = "perplexity/sonar-pro"

TOP_N_STAGE2 = 3   # Stage2に進める銘柄数
MAX_STOCKS   = 5   # 1仮説あたりStage1に通す銘柄数

JST = timezone(timedelta(hours=9))

# ── API ──────────────────────────────────────────────────────
def get_or_key():
    if SECRETS.exists():
        return json.loads(SECRETS.read_text())["openrouter"]["api_key"]
    return ""

def call_llm(model, messages, temperature=0.3, timeout=90, retries=3):
    import time
    last_err = None
    for attempt in range(retries):
        try:
            r = httpx.post(OR_URL,
                headers={"Authorization": f"Bearer {get_or_key()}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "temperature": temperature},
                timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            wait = 10 * (attempt + 1)
            print(f"    LLMエラー(試行{attempt+1}/{retries}): {e} → {wait}秒後リトライ", flush=True)
            time.sleep(wait)
    print(f"    LLM失敗（{retries}回試行）: {last_err}", flush=True)
    return ""

def extract_obj(text):
    # コードブロック除去
    text = re.sub(r'```(?:json)?\s*', '', text).strip()
    # 最外のJSONオブジェクトを抽出
    start = text.find('{')
    if start == -1:
        return {}
    depth, end = 0, -1
    for i, c in enumerate(text[start:], start):
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return {}
    try:
        return json.loads(text[start:end])
    except Exception:
        # 壊れたJSONは修復試行
        try:
            import ast
            return ast.literal_eval(text[start:end])
        except:
            return {}

# ── J-Quants 財務データ取得 ──────────────────────────────────
def fetch_fins(code: str) -> dict:
    """J-Quants fins/summary から直近の財務数値を取得"""
    try:
        url = f"https://api.jquants.com/v2/fins/summary?code={code}"
        r = httpx.get(url, headers={"x-api-key": JQUANTS_KEY}, timeout=30)
        if r.status_code != 200:
            return {}
        items = r.json().get("fins_summary", [])
        if not items:
            return {}
        # 最新の通期決算を取得
        annual = [x for x in items if x.get("TypeOfDocument","").endswith("Annual")]
        latest = annual[-1] if annual else items[-1]
        return {
            "sales":        latest.get("NetSales"),           # 売上高（百万円）
            "op":           latest.get("OperatingProfit"),    # 営業利益（百万円）
            "np":           latest.get("NetIncome"),          # 純利益（百万円）
            "eps":          latest.get("EPS"),                 # EPS（円）
            "bps":          latest.get("BPS"),                 # BPS（円）
            "op_margin":    (latest.get("OperatingProfit",0) / latest.get("NetSales",1) * 100)
                            if latest.get("NetSales") else None,
            "f_sales":      latest.get("ForecastNetSales"),   # 会社予想売上
            "f_op":         latest.get("ForecastOperatingProfit"),
            "f_eps":        latest.get("ForecastEPS"),
            "fiscal_year":  latest.get("FiscalYear"),
        }
    except Exception as e:
        print(f"      J-Quants取得失敗 {code}: {e}", flush=True)
        return {}

# ── 現在株価取得（Perplexity）────────────────────────────────
def fetch_price(code: str, name: str) -> dict:
    prompt = f"""東証上場「{name}（{code}）」の以下をJSONで。不明はnull。
{{"current_price":円,"per":倍,"pbr":倍,"analyst_tp":円,"consensus":"Buy/Hold/Sell","buy_ratio_pct":%}}
JSONのみ。"""
    try:
        res = call_llm(MARKET_MODEL, [{"role":"user","content":prompt}])
        return extract_obj(res)
    except:
        return {}

# ── STAGE 1: ざっくりスクリーニング（gpt-4.1-mini）────────────
def stage1_screen(hypothesis: dict, stocks: list) -> list:
    """全銘柄を一括でざっくりスコアリング。関連スコア0-10 + 簡易alpha予測"""
    if not stocks:
        return []

    stocks_txt = "\n".join(f"- {s.get('code')} {s.get('name')}: {s.get('reason','')}" for s in stocks)
    prompt = f"""以下の投資仮説と候補銘柄リストについて、各銘柄の「この仮説での受益ポテンシャル」を0-10でスコアリングしてください。

仮説: {hypothesis.get('theme')} — {hypothesis.get('insight','')}
タイムライン: {hypothesis.get('timeline','')}

候補銘柄:
{stocks_txt}

各銘柄について以下をJSONで返してください:
[
  {{"code": "1234", "relevance_score": 8, "quick_reason": "なぜスコアがこれか一言", "est_upside_pct": 25}}
]
JSONのみ。"""

    try:
        res = call_llm(STAGE1_MODEL, [{"role":"user","content":prompt}], temperature=0.2)
        m = re.search(r'\[.*\]', res, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        print(f"    Stage1エラー: {e}", flush=True)
    return []

# ── STAGE 2: 詳細評価（gpt-5 + 実財務 + 3段階分解）───────────
def stage2_evaluate(hypothesis: dict, stock: dict, fins: dict, price_data: dict) -> dict:
    """
    3段階で厳密に評価:
    1. Market Sizing    → 市場規模×シェア変化 = 売上インパクト（円建て）
    2. Moat Assessment  → 参入障壁を0-5で定量スコアリング
    3. Forward Valuation→ 将来EPS×適正PER = 目標株価。現フォワードPERは割安か
    """
    current_price = price_data.get("current_price") or 0
    per = price_data.get("per") or 15

    fins_text = ""
    if fins.get("sales"):
        sales_bn = fins["sales"] / 1000
        op_bn    = (fins.get("op") or 0) / 1000
        fins_text = f"""
実績財務（直近通期 {fins.get("fiscal_year","")}）:
  売上高: {sales_bn:.1f}十億円
  営業利益: {op_bn:.1f}十億円（営業利益率 {fins.get("op_margin",0):.1f}%）
  EPS実績: {fins.get("eps","不明")}円 / 会社予想EPS: {fins.get("f_eps","不明")}円
  BPS: {fins.get("bps","不明")}円"""

    prompt = f"""あなたはトップティアのヘッジファンドのシニアアナリストです。
以下の投資仮説が実現した場合の「{stock.get("name")}」への業績・バリュエーションインパクトを、3ステップで厳密に分析してください。

━━ 投資仮説 ━━
テーマ: {hypothesis.get("theme")}
洞察: {hypothesis.get("insight","")}
タイムライン: {hypothesis.get("timeline","")}
Moatシグナル: {hypothesis.get("moat_signal", stock.get("moat", ""))}

━━ この銘柄の受益ロジック ━━
{stock.get("reason","")}
なぜ今割安に放置されているか: {stock.get("current_concern","")}

━━ 現在の財務・株価データ ━━{fins_text}
現在株価: {current_price}円 / PER: {per}倍 / PBR: {price_data.get("pbr","不明")}倍
アナリストTP: {price_data.get("analyst_tp","不明")}円（コンセンサス: {price_data.get("consensus","不明")}）

━━ 分析要求（3ステップ） ━━

Step1 Market Sizing: 対象市場規模（現在・将来）を推定し、この企業が取れるシェア変化から売上インパクト（十億円）を積み上げ計算
Step2 Moat Assessment: 技術/特許、スイッチングコスト、スケール優位、規制障壁を各0-5でスコアリング。なぜ競合ではなくこの企業だけが取れるか
Step3 Forward Valuation: Step1のインパクトをEPS（円）に変換し、適正PER（根拠付き）で目標株価を計算。現在のフォワードPERが割安かを判定

以下のJSONのみ出力:

{{
  "market_sizing": {{
    "target_market_now_bn": 0,
    "target_market_future_bn": 0,
    "company_share_now_pct": 0,
    "company_share_after_pct": 0,
    "revenue_impact_bn": {{"bear": 0, "base": 0, "bull": 0}},
    "mechanism": "なぜこのシェア増加が起きるか（具体的に80字）"
  }},
  "moat_assessment": {{
    "tech_patent_score": 0,
    "switching_cost_score": 0,
    "scale_advantage_score": 0,
    "regulatory_barrier_score": 0,
    "total_moat_score": 0,
    "moat_summary": "参入障壁の本質（60字）",
    "key_risk_to_moat": "このMoatが崩れる条件（50字）"
  }},
  "valuation": {{
    "current_eps_jpy": 0,
    "forward_eps_base_jpy": 0,
    "forward_eps_bull_jpy": 0,
    "eps_growth_driver": "EPS増加の計算根拠（税率・株式数の前提を明記）",
    "fair_per": 0,
    "per_basis": "なぜこのPERが適正か（成長率・業界比較）",
    "target_price": {{"bear": 0, "base": 0, "bull": 0}},
    "current_forward_per": 0,
    "valuation_verdict": "割安/適正/割高",
    "upside_to_base_pct": 0
  }},
  "op_impact": {{
    "bear": {{"amount_bn": 0, "pct": 0}},
    "base": {{"amount_bn": 0, "pct": 0}},
    "bull": {{"amount_bn": 0, "pct": 0}}
  }},
  "market_pricing_pct": 0.2,
  "pricing_rationale": "なぜ市場がまだ織り込んでいないか",
  "key_catalyst": "株価が動く最初のトリガーイベント（具体的に）",
  "catalyst_timeline": "2026年Q3",
  "stop_loss_thesis": "この仮説が完全に崩れる条件",
  "position_size_suggestion": "2-3%",
  "stop_loss_pct": -12
}}

全ての数値を必ず埋めること。JSONのみ出力。"""

    try:
        res = call_llm(STAGE2_MODEL, [{"role":"user","content":prompt}], temperature=0.3)
        detail = extract_obj(res)
        # eps_impact を valuation から生成（後方互換）
        if "valuation" in detail and "eps_impact" not in detail:
            v = detail["valuation"]
            cur = v.get("current_eps_jpy") or 0
            fwd_b = v.get("forward_eps_base_jpy") or 0
            fwd_u = v.get("forward_eps_bull_jpy") or 0
            detail["eps_impact"] = {
                "base": {
                    "amount_jpy": round(fwd_b - cur, 1) if cur and fwd_b else None,
                    "pct":        round((fwd_b - cur) / cur * 100, 1) if cur and fwd_b else None,
                },
                "bull": {
                    "amount_jpy": round(fwd_u - cur, 1) if cur and fwd_u else None,
                    "pct":        round((fwd_u - cur) / cur * 100, 1) if cur and fwd_u else None,
                },
                "note": v.get("eps_growth_driver", ""),
            }
        # target_price を valuation から統一
        if "valuation" in detail:
            detail["target_price"] = detail["valuation"].get("target_price", {})
        return detail
    except Exception as e:
        print(f"    Stage2エラー {stock.get('name')}: {e}", flush=True)
        return {}

# ── Alpha Score ──────────────────────────────────────────────
def parse_timeline_years(text) -> float:
    if not text or not isinstance(text, str): return 2.0
    m = re.search(r'(\d{4})', text)
    if m:
        return max(0.5, int(m.group(1)) - 2026 + 0.5)
    if 'Q' in text: return 1.0
    m2 = re.search(r'(\d+)\s*年', text)
    if m2: return float(m2.group(1))
    return 2.0

def calc_alpha(hypothesis: dict, detail: dict, s1: dict) -> dict:
    confidence  = hypothesis.get('confidence', 0.5)
    upside      = detail.get('target_price', {}).get('base') or 0
    cur_price   = detail.get('_current_price') or 0
    downside    = abs(float(detail.get("stop_loss_pct") or 15))
    pricing_gap = 1.0 - min(1.0, max(0.0, detail.get('market_pricing_pct', 0.5)))
    tl_years    = parse_timeline_years(hypothesis.get('timeline', ''))

    upside_pct = ((upside - cur_price) / cur_price * 100) if cur_price > 0 and upside > 0 else (s1.get('est_upside_pct') or 0)

    expected_annual = (upside_pct / 100 * confidence * pricing_gap) / max(0.5, tl_years) * 100
    b = (upside_pct / 100) / (downside / 100) if downside > 0 else 0
    p = confidence
    kelly = max(0.0, (p * b - (1-p)) / b) * 0.5 if b > 0 else 0

    return {
        'alpha_score':                round(min(1.0, expected_annual / 20.0), 3),
        'expected_annual_return_pct': round(expected_annual, 1),
        'upside_base_pct':            round(upside_pct, 1),
        'kelly_fraction':             round(kelly, 3),
    }

# ── メインループ ─────────────────────────────────────────────
def evaluate_hypothesis(h: dict) -> dict:
    stocks = h.get('candidate_stocks', [])[:MAX_STOCKS]
    if not stocks:
        return {**h, 'evaluated_stocks': [], 'alpha_score': 0, 'top_pick': None}

    print(f"  Stage1 スクリーニング: {len(stocks)}銘柄...")
    s1_results = stage1_screen(h, stocks)
    # code → スコアのマップ
    s1_map = {s['code']: s for s in s1_results}

    # Stage1スコアでソート → 上位TOP_N_STAGE2を詳細評価
    scored = sorted(stocks, key=lambda s: s1_map.get(s.get('code','-'), {}).get('relevance_score', 0), reverse=True)
    top = scored[:TOP_N_STAGE2]
    rest = scored[TOP_N_STAGE2:]

    evaluated = []
    print(f"  Stage2 詳細評価: {len(top)}銘柄（Stage1上位）...")

    for s in top:
        code = s.get('code', '')
        name = s.get('name', '')
        s1   = s1_map.get(code, {})
        print(f"    📊 {code} {name} [Stage1スコア: {s1.get('relevance_score','?')}/10]")

        # 実財務データ取得
        fins = fetch_fins(code)
        price_data = fetch_price(code, name)

        # Stage2 詳細試算
        detail = stage2_evaluate(h, s, fins, price_data)
        detail['_current_price'] = price_data.get('current_price', 0)

        # スコア計算
        scores = calc_alpha(h, detail, s1)
        upside_pct = scores['upside_base_pct']
        target_base = detail.get('target_price', {}).get('base', 0)

        evaluated.append({
            **s,
            'stage1_score':       s1.get('relevance_score'),
            'stage1_quick_reason':s1.get('quick_reason'),
            'current_price':      price_data.get('current_price'),
            'per':                price_data.get('per'),
            'pbr':                price_data.get('pbr'),
            'analyst_target_price': price_data.get('analyst_tp'),
            'analyst_consensus':  price_data.get('consensus'),
            'fins':               fins,
            **detail,
            **scores,
        })

    # Stage1のみの銘柄も記録（サマリーのみ）
    for s in rest:
        code = s.get('code','')
        s1 = s1_map.get(code, {})
        evaluated.append({
            **s,
            'stage': 'stage1_only',
            'stage1_score': s1.get('relevance_score'),
            'stage1_quick_reason': s1.get('quick_reason'),
            'alpha_score': s1.get('relevance_score', 0) / 10 * 0.3,  # 仮スコア
        })

    # アルファスコア順ソート
    evaluated.sort(key=lambda x: x.get('alpha_score', 0), reverse=True)
    top_pick = next((e for e in evaluated if e.get('stage') != 'stage1_only'), evaluated[0] if evaluated else None)

    return {
        **h,
        'evaluated_stocks': evaluated,
        'alpha_score':       top_pick.get('alpha_score', 0) if top_pick else 0,
        'expected_annual_return_pct': top_pick.get('expected_annual_return_pct', 0) if top_pick else 0,
        'top_pick':          top_pick,
    }

def evaluate_all(hypotheses=None):
    if hypotheses is None:
        if not HYPO_FILE.exists():
            print("仮説ファイルなし", flush=True); return
        data = json.loads(HYPO_FILE.read_text())
        hypotheses = data.get('hypotheses', [])

    if not hypotheses:
        print("仮説なし", flush=True); return

    print(f"\n{'='*55}", flush=True)
    print(f"🏦 HF評価エンジン v2 ({len(hypotheses)}仮説 / 2段階方式)")
    print(f"  Stage1: {STAGE1_MODEL}", flush=True)
    print(f"  Stage2: {STAGE2_MODEL} (上位{TOP_N_STAGE2}銘柄のみ)")
    print(f"{'='*55}\n", flush=True)

    ranked = []
    for i, h in enumerate(hypotheses):
        print(f"[{i+1}/{len(hypotheses)}] {h.get('theme','—')}", flush=True)
        try:
            ev = evaluate_hypothesis(h)
            ranked.append(ev)
        except Exception as e:
            print(f"  ⚠️ スキップ: {e}", flush=True)
            ranked.append({**h, 'evaluated_stocks': [], 'alpha_score': 0, 'top_pick': None})

    ranked.sort(key=lambda x: x.get('alpha_score', 0), reverse=True)
    for i, h in enumerate(ranked):
        h['rank'] = i + 1

    output = {
        'updated_at': datetime.now(JST).strftime('%Y-%m-%d %H:%M'),
        'total': len(ranked),
        'stage1_model': STAGE1_MODEL,
        'stage2_model': STAGE2_MODEL,
        'ranked_hypotheses': ranked,
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    print(f"\n{'='*55}", flush=True)
    print(f"✅ 評価完了 → {OUTPUT.name}", flush=True)
    print(f"\n🏆 Top 3:", flush=True)
    for h in ranked[:3]:
        tp = h.get('top_pick') or {}
        eps_b = ((tp.get('eps_impact') or {}).get('base') or {})
        eps_amt = eps_b.get('amount_jpy', '?')
        eps_pct = eps_b.get('pct', '?')
        print(f"  #{h['rank']} {h.get('theme')} | α={h.get('alpha_score',0):.2f} | "
              f"年率+{h.get('expected_annual_return_pct',0):.0f}% | "
              f"Top: {tp.get('code','—')} EPS+{eps_amt}円({eps_pct}%)")
    print(f"{'='*55}\n", flush=True)

    return ranked

if __name__ == "__main__":
    evaluate_all()
