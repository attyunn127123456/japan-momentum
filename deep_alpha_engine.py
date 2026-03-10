"""
Deep Alpha Engine - 専門的知見×未来予測×市場の見落としから深い投資仮説を生成

アーキテクチャ:
1. Perplexity Sonar Pro（web検索内蔵）で業界最先端の情報収集
2. Opus が「市場コンセンサスとのギャップ」を分析して鋭い仮説を生成
3. Devil's Advocate で反論検証
4. J-Quants で具体銘柄へ落とし込み
"""

import json
import os
import httpx
import random
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
OUTPUT = BASE / "backtest/macro_hypotheses.json"
SECRETS = BASE / ".secrets/accounts.json"

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"

# 探索するドメイン（毎回違う角度から掘る）
DOMAINS = [
    {
        "name": "半導体製造・素材",
        "angles": [
            "EUV High-NAの量産移行で何が変わるか。レジスト材料・ペリクル・光学系部品の供給制約",
            "Gate-All-Around（GAA）トランジスタへの移行でエッチング・成膜装置の何が変わるか",
            "HBM（高帯域幅メモリ）のTSV工程で日本の化学・素材メーカーが持つ寡占的ポジション",
            "半導体パッケージング（CoWoS・SoIC）の拡大で恩恵を受ける日本の材料・装置メーカー",
            "光電融合（シリコンフォトニクス）量産化で必要な新素材・新プロセス",
        ]
    },
    {
        "name": "AI・データセンターインフラ",
        "angles": [
            "推論専用チップ時代に学習インフラと何が変わるか。電力・冷却・ネットワーク",
            "日本のデータセンター電力不足と再エネ調達競争。受益する電力・不動産・建設会社",
            "エッジAI・小型モデルへの移行でどんなハードウェアが必要になるか",
            "AIエージェント普及でAPIコール急増。インフラの何がボトルネックになるか",
        ]
    },
    {
        "name": "エネルギー転換",
        "angles": [
            "全固体電池の量産ロードマップと日本の素材メーカーの立ち位置",
            "洋上風力の日本展開で最もボトルネックになる部品・素材・工事会社",
            "核融合（ITER・民間企業）が2030年代に与える電力インフラへの影響",
            "系統用蓄電池の急拡大でLFP vs NMC、日本企業のポジション",
        ]
    },
    {
        "name": "医療・バイオ",
        "angles": [
            "ADC（抗体薬物複合体）の製造ボトルネック。リンカー・ペイロードの原料供給",
            "核医学治療（放射性同位体療法）の量産化で必要なインフラ・原料",
            "AIによる創薬スクリーニングが製薬会社の研究開発コストに与える影響",
        ]
    },
    {
        "name": "製造・ロボティクス",
        "angles": [
            "ヒューマノイドロボット量産化で最も需要が増えるアクチュエータ・センサー・素材",
            "中国製造業の自動化加速で日本の工作機械・FA機器に何が起きるか",
            "サプライチェーン再編（中国+1）で恩恵を受ける日本の製造拠点・企業",
        ]
    },
    {
        "name": "金融・資本市場",
        "angles": [
            "日本の金利上昇局面で真に恩恵を受ける金融業態（単純な銀行以外）",
            "東証PBR改善要請の第2フェーズ。自社株買い・M&A・事業売却の次に来るもの",
            "インバウンド消費の構造変化。訪日客の消費パターンと受益する意外な業種",
        ]
    }
]


def get_key():
    if SECRETS.exists():
        return json.loads(SECRETS.read_text())["openrouter"]["api_key"]
    return os.environ.get("OPENROUTER_API_KEY", "")


def call_llm(model: str, messages: list, temperature: float = 0.7) -> str:
    key = get_key()
    resp = httpx.post(
        OPENROUTER_BASE,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=180
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def research_domain(domain: dict, angle: str) -> str:
    """Perplexity Sonar Proでウェブ検索込みのリサーチ"""
    print(f"  🔍 リサーチ中: {angle[:50]}...")
    prompt = f"""以下のテーマについて、業界の最先端動向・技術ロードマップ・グローバルサプライチェーンの観点から深く調査してください。

テーマ: {angle}

特に以下を調査:
1. 現在の技術的・物理的限界と次世代への移行タイムライン
2. グローバルサプライチェーンでの日本企業のポジション（寡占・ニッチ・脆弱性）
3. 直近1〜2年の業界の動き（M&A・投資・技術発表・規制変化）
4. 5〜10年後に何が起こるかの専門的予測

できるだけ具体的な企業名・製品名・数値を含めて回答してください。"""

    return call_llm("perplexity/sonar-pro", [{"role": "user", "content": prompt}])


def generate_alpha_hypothesis(domain: dict, research_results: list) -> str:
    """Opusで市場の見落としから深い仮説を生成"""
    print(f"  🧠 Opusが仮説を生成中...")
    
    research_text = "\n\n---\n\n".join(research_results)
    
    prompt = f"""あなたは世界トップレベルのファンダメンタル投資家です。以下のリサーチ結果を読んで、日本株への投資仮説を生成してください。

## リサーチ結果
{research_text}

## あなたのタスク

**重要**: 「みんなが知っていること」ではなく「まだ市場が気づいていないこと」を探してください。

以下の観点で深く考えてください:
- コンセンサスと現実のギャップはどこか
- 二次・三次効果（直接の受益者ではなく、その先の受益者）
- サプライチェーンの中でボトルネックになる部分
- 市場が短期的な悪材料で見落としている長期的なポジション
- 技術移行期に必ず生じる「古い技術に依存している会社」「新技術で唯一解になる会社」

## 出力形式（JSON）

```json
{{
  "hypotheses": [
    {{
      "id": "alpha_YYYYMMDD_001",
      "theme": "（一言で表すテーマ、20字以内）",
      "insight": "（市場が見落としている核心的な洞察、200字程度）",
      "logic": "（なぜこの仮説が成立するか、技術的・構造的な論拠、400字程度）",
      "timeline": "（この仮説が顕在化するタイムライン、例: '2026年後半〜2027年'）",
      "target_company_type": "（どういう特性の会社を探すべきか）",
      "key_risks": ["（最大のリスク1）", "（最大のリスク2）"],
      "confidence": 0.0〜1.0,
      "domain": "{domain['name']}"
    }}
  ]
}}
```

仮説は1〜3個。深さ最優先。浅い仮説は出さないこと。JSONのみ出力。"""

    return call_llm("anthropic/claude-opus-4-6", [{"role": "user", "content": prompt}], temperature=0.8)


def devils_advocate(hypothesis: dict) -> dict:
    """反論で仮説を鍛える"""
    print(f"  ⚔️  反論検証: {hypothesis.get('theme', '')}")
    
    prompt = f"""以下の投資仮説に対して、最も鋭い反論を3つ挙げてください。そして、その反論を踏まえて仮説の確信度を再評価してください。

## 仮説
テーマ: {hypothesis.get('theme')}
洞察: {hypothesis.get('insight')}
論拠: {hypothesis.get('logic')}

## 出力（JSON）
```json
{{
  "counterarguments": ["反論1", "反論2", "反論3"],
  "revised_confidence": 0.0〜1.0,
  "revised_logic": "（反論を踏まえて修正・強化された論拠）",
  "leading_indicators": ["（この仮説が正しければ先に動く指標1）", "（指標2）"]
}}
```

JSONのみ出力。"""

    result = call_llm("anthropic/claude-opus-4-6", [{"role": "user", "content": prompt}], temperature=0.5)
    
    try:
        import re
        match = re.search(r'\{.*\}', result, re.DOTALL)
        if match:
            review = json.loads(match.group())
            hypothesis["counterarguments"] = review.get("counterarguments", [])
            hypothesis["confidence"] = review.get("revised_confidence", hypothesis.get("confidence", 0.5))
            hypothesis["logic"] = review.get("revised_logic", hypothesis.get("logic", ""))
            hypothesis["leading_indicators"] = review.get("leading_indicators", [])
    except Exception as e:
        print(f"    反論パースエラー: {e}")
    
    return hypothesis


def find_jp_stocks(hypothesis: dict) -> list:
    """仮説から具体的な日本株候補を探す（東証上場のみ）"""
    print(f"  📊 銘柄マッピング中...")

    prompt = f"""以下の投資仮説に合致する日本の上場企業を3〜5社挙げてください。

## 仮説
テーマ: {hypothesis.get('theme')}
洞察: {hypothesis.get('insight')}
対象企業タイプ: {hypothesis.get('target_company_type', '')}

## 厳守条件
- **東証上場企業のみ**（プライム・スタンダード・グロース問わず）
- 米国株・未上場企業は絶対に含めない
- 証券コード（4桁または5桁）を必ず記載
- なぜ今この仮説で割安・見落とされているかを具体的に

## 出力（JSON配列のみ）
[
  {{
    "code": "1234",
    "name": "会社名",
    "reason": "なぜこの仮説の受益者か（具体的に100字程度）",
    "current_concern": "なぜ今まだ市場に見落とされているか"
  }}
]

JSONのみ出力。"""

    result = call_llm("anthropic/claude-opus-4-6", [{"role": "user", "content": prompt}], temperature=0.6)

    try:
        import re
        match = re.search(r'\[.*\]', result, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"    銘柄マッピングエラー: {e}")

    return []


def find_startups(hypothesis: dict) -> list:
    """仮説から有望な未上場スタートアップを探す"""
    print(f"  🚀 スタートアップリサーチ中...")

    prompt = f"""以下の投資仮説に関連する分野で、特に有望な日本の未上場スタートアップを1〜3社挙げてください。

## 仮説
テーマ: {hypothesis.get('theme')}
洞察: {hypothesis.get('insight')}

## 条件
- 日本の未上場スタートアップ（VC投資済み、または注目されている企業）
- 上場企業は含めない
- 実在する会社のみ（不確かな場合は記載しない）
- この仮説のテーマに直接関連する事業を持つこと

## 出力（JSON配列のみ）
[
  {{
    "name": "スタートアップ名",
    "founded": "設立年（わかれば）",
    "business": "事業内容（50字程度）",
    "why_promising": "なぜこの仮説で有望か（100字程度）",
    "stage": "シリーズA/B/C等（わかれば）",
    "investors": "主要投資家（わかれば）"
  }}
]

確実に存在する会社のみ。不明な場合は空配列[]を返す。JSONのみ出力。"""

    result = call_llm("perplexity/sonar-pro", [{"role": "user", "content": prompt}], temperature=0.5)

    try:
        import re
        match = re.search(r'\[.*\]', result, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return [s for s in data if s.get("name")]  # 名前があるものだけ
    except Exception as e:
        print(f"    スタートアップリサーチエラー: {e}")

    return []


def run():
    print("=" * 60)
    print("🔬 Deep Alpha Engine 起動")
    print("=" * 60)
    
    # 今回掘るドメインをローテーション（前回の履歴から）
    history_file = BASE / "backtest/alpha_domain_history.json"
    history = json.loads(history_file.read_text()) if history_file.exists() else {"last_domains": []}
    
    # 前回使っていないドメインを優先
    available = [d for d in DOMAINS if d["name"] not in history["last_domains"][-3:]]
    if not available:
        available = DOMAINS
    
    domain = random.choice(available)
    angles = random.sample(domain["angles"], min(2, len(domain["angles"])))  # 2角度から掘る
    
    print(f"\n📌 今回のドメイン: {domain['name']}")
    print(f"📌 探索角度: {len(angles)}個\n")
    
    # Phase 1: リサーチ
    research_results = []
    for angle in angles:
        result = research_domain(domain, angle)
        research_results.append(f"### {angle}\n\n{result}")
    
    # Phase 2: 仮説生成
    hypothesis_raw = generate_alpha_hypothesis(domain, research_results)
    
    # JSONパース
    import re
    hypotheses = []
    match = re.search(r'\{.*\}', hypothesis_raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            hypotheses = data.get("hypotheses", [])
        except:
            pass
    
    if not hypotheses:
        print("⚠️ 仮説生成失敗")
        return
    
    print(f"\n💡 生成された仮説: {len(hypotheses)}個")
    
    # Phase 3: Devil's Advocate
    refined = []
    for h in hypotheses:
        h["id"] = f"alpha_{datetime.now().strftime('%Y%m%d')}_{len(refined)+1:03d}"
        h["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        h = devils_advocate(h)
        
        # Phase 4: 銘柄マッピング（東証上場）
        h["candidate_stocks"] = find_jp_stocks(h)
        # Phase 5: スタートアップリサーチ
        h["startups"] = find_startups(h)
        refined.append(h)
        
        print(f"\n  ✅ {h.get('theme')} (確信度: {h.get('confidence', 0):.0%})")
    
    # 既存の仮説と合算（最新30件を保持）
    existing = []
    if OUTPUT.exists():
        try:
            old = json.loads(OUTPUT.read_text())
            existing = old.get("hypotheses", [])
        except:
            pass
    
    all_hypotheses = refined + existing
    all_hypotheses = all_hypotheses[:30]  # 最新30件
    
    output = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "last_domain": domain["name"],
        "hypotheses": all_hypotheses
    }
    
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    
    # ドメイン履歴更新
    history["last_domains"].append(domain["name"])
    history["last_domains"] = history["last_domains"][-10:]
    history_file.write_text(json.dumps(history))
    
    print(f"\n{'='*60}")
    print(f"✨ 完了: 仮説{len(refined)}件を生成・保存")
    print(f"{'='*60}")
    
    return refined


if __name__ == "__main__":
    run()
