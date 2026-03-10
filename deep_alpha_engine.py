"""
Deep Alpha Engine v2
マクロトレンド起点の15ドメインから深い投資仮説を生成
"""

import json, httpx, random, re
from pathlib import Path
from datetime import datetime

BASE    = Path(__file__).parent
OUTPUT  = BASE / "backtest/macro_hypotheses.json"
SECRETS = BASE / ".secrets/accounts.json"
OR_URL  = "https://openrouter.ai/api/v1/chat/completions"

def get_key():
    if SECRETS.exists():
        return json.loads(SECRETS.read_text())["openrouter"]["api_key"]
    return ""

def call_llm(model, messages, temperature=0.7, timeout=180):
    r = httpx.post(OR_URL,
        headers={"Authorization": f"Bearer {get_key()}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ── 15ドメイン（マクロトレンド起点） ─────────────────────────
DOMAINS = [
    {
        "name": "半導体製造・素材",
        "macro_driver": "AI/DC投資拡大と技術世代交代",
        "angles": [
            "EUV High-NA量産移行でレジスト・ペリクル・光学系部品の供給制約は何が起きるか",
            "Gate-All-Around（GAA）トランジスタ移行でエッチング・成膜装置市場はどう変わるか",
            "HBMのTSV工程で日本の化学・素材メーカーが持つ寡占的ポジション",
            "半導体パッケージング（CoWoS・SoIC）拡大で恩恵を受ける日本の材料・装置メーカー",
            "光電融合（シリコンフォトニクス）量産化で必要な新素材・新プロセスの日本企業優位",
        ]
    },
    {
        "name": "AI・データセンターインフラ",
        "macro_driver": "推論需要爆発とエネルギー制約",
        "angles": [
            "推論専用チップ時代に電力・冷却・ネットワークの何がボトルネックになるか",
            "日本のデータセンター電力不足と再エネ調達競争。受益する電力・不動産・建設会社",
            "AIエージェント普及でAPIコール急増。インフラで真に不足するものは何か",
            "液冷・浸漬冷却への移行で必要な新素材・部品・メンテ企業",
        ]
    },
    {
        "name": "エネルギー転換",
        "macro_driver": "脱炭素規制強化と電力需要増大",
        "angles": [
            "全固体電池の量産ロードマップと日本の素材メーカーの立ち位置",
            "洋上風力の日本展開で最もボトルネックになる部品・素材・工事会社",
            "系統用蓄電池の急拡大でLFP vs NMCの競合構造はどう変わるか",
            "電力系統の超高圧直流送電（HVDC）拡大で受益する日本の重電メーカー",
        ]
    },
    {
        "name": "医療・バイオ",
        "macro_driver": "高齢化加速とAI創薬革命",
        "angles": [
            "ADC（抗体薬物複合体）の製造ボトルネック。リンカー・ペイロードの原料供給寡占",
            "核医学治療（放射性同位体療法）の量産化で必要なインフラ・原料の日本優位",
            "AIによる創薬スクリーニングが製薬会社の研究開発コスト構造に与える影響",
            "後発品（バイオシミラー）普及で淘汰される製薬会社と勝ち残る会社の違い",
        ]
    },
    {
        "name": "製造・ロボティクス",
        "macro_driver": "人手不足と中国自動化加速",
        "angles": [
            "ヒューマノイドロボット量産化で最も需要が増えるアクチュエータ・センサー・素材",
            "中国製造業の自動化加速で日本の工作機械・FA機器に何が起きるか",
            "サプライチェーン再編（中国+1）で恩恵を受ける日本の製造拠点・企業",
        ]
    },
    {
        "name": "金融・資本市場",
        "macro_driver": "金利正常化とPBR改革第2フェーズ",
        "angles": [
            "日本の金利上昇局面で真に恩恵を受ける金融業態（単純な銀行以外）",
            "東証PBR改革第2フェーズ。自社株買い・M&A・事業売却の次に来るもの",
            "インバウンド消費の構造変化。訪日客の消費パターンと受益する意外な業種",
        ]
    },
    {
        "name": "地政学・サプライチェーン再編",
        "macro_driver": "米中対立の長期化と日本の戦略的地位向上",
        "angles": [
            "米国の対中輸出規制強化で日本の半導体・素材輸出規制の抜け穴と代替需要",
            "台湾有事リスクを織り込んで日本国内に製造拠点を移す動きの恩恵企業",
            "レアアース・重要鉱物の中国依存から脱却しようとする動きで受益する日本企業",
            "日本の防衛費倍増（GDP2%）で最も受益する防衛関連・デュアルユース企業",
        ]
    },
    {
        "name": "宇宙・衛星インフラ",
        "macro_driver": "衛星コンステレーション競争と宇宙経済の商業化",
        "angles": [
            "H3ロケット量産化で日本の宇宙アクセスコストがどこまで下がるか",
            "低軌道衛星コンステレーション（Starlink競合）で恩恵を受ける日本の部品・素材メーカー",
            "衛星データ（SAR・光学）の商業利用拡大で必要なグラウンドインフラと解析企業",
            "宇宙デブリ除去（アクティブデブリリムーバル）市場の立ち上がりと日本の優位性",
        ]
    },
    {
        "name": "量子コンピューティング",
        "macro_driver": "量子優位性実証と実用化競争",
        "angles": [
            "量子コンピュータの冷却（希釈冷凍機）・制御（マイクロ波）・配線で日本企業が持つ優位性",
            "量子暗号通信の実用化で恩恵を受ける日本の光ファイバー・通信機器メーカー",
            "量子センシング（重力センサー・磁気センサー）の商業応用で最初に市場が立つ分野",
        ]
    },
    {
        "name": "次世代原子力・核融合",
        "macro_driver": "電力安定供給ニーズとネットゼロ目標の矛盾解決",
        "angles": [
            "小型モジュール炉（SMR）の2030年代量産化で日本の重工・素材メーカーが担う部分",
            "核融合（民間企業）のタイムラインが前倒しになった場合に最初に恩恵を受ける部品・素材",
            "原発再稼働の加速で日本の電力会社の収益構造がどう変わるか。隠れた受益者は誰か",
        ]
    },
    {
        "name": "水・食料・農業テック",
        "macro_driver": "気候変動による食料安保リスクと農業人口減少",
        "angles": [
            "スマート農業（ドローン散布・精密農業）の普及で受益する日本の農業機械・センサー企業",
            "植物工場・垂直農業の量産化で必要な照明・環境制御・栄養液の日本メーカー優位",
            "水処理・海水淡水化技術の新興国需要拡大で日本の膜・ポンプ企業のポジション",
        ]
    },
    {
        "name": "高齢化・介護・ヘルスケアテック",
        "macro_driver": "2025年問題と介護人材危機",
        "angles": [
            "介護ロボット（移乗・見守り・排泄支援）の普及加速で受益する日本の企業",
            "在宅医療・訪問看護のデジタル化で恩恵を受けるヘルスIT・医療機器企業",
            "認知症・慢性疾患のデジタル療法（DTx）市場の立ち上がりと日本企業の勝算",
        ]
    },
    {
        "name": "デジタルインフラ・通信",
        "macro_driver": "データトラフィック爆発と6G競争",
        "angles": [
            "海底ケーブルの新設・増強競争で恩恵を受ける日本の光ファイバー・ケーブル船企業",
            "6G（2030年代）の基地局・端末に必要な新素材・部品で日本企業が取れるポジション",
            "ローカル5G・プライベートネットワーク普及で恩恵を受ける日本の通信機器・SI企業",
        ]
    },
    {
        "name": "生成AI × 業務DX",
        "macro_driver": "ホワイトカラー生産性革命",
        "angles": [
            "業種特化型AI（法律・医療・会計・設計）の普及で受益する垂直SaaS企業",
            "企業の基幹システム（ERP・会計）のAI刷新需要で恩恵を受けるSI・パッケージ企業",
            "AIによる自動化で需要が増える逆説的ポジション（データラベリング・品質保証）",
        ]
    },
    {
        "name": "インバウンド・観光再構築",
        "macro_driver": "円安定着と訪日客3000万人超",
        "angles": [
            "富裕層インバウンドの高単価消費で恩恵を受ける宿泊・体験・コンテンツ企業",
            "訪日客の決済・免税・電子チケット需要で受益するフィンテック・IT企業",
            "地方観光地の再生（オーバーツーリズム対策）で新たに浮上する観光インフラ企業",
        ]
    },
]


def research_domain(domain: dict, angle: str) -> str:
    print(f"  🔍 {angle[:55]}...", flush=True)
    prompt = f"""以下のテーマについて、業界の最前線動向・技術ロードマップ・グローバルサプライチェーンの観点から深く調査してください。

マクロドライバー: {domain.get('macro_driver', '')}
調査テーマ: {angle}

特に以下を調査:
1. 現在の技術的・物理的限界と次世代への移行タイムライン
2. グローバルサプライチェーンでの日本企業のポジション（寡占・ニッチ・脆弱性）
3. 直近1〜2年の業界の動き（M&A・投資・技術発表・規制変化）
4. 5〜10年後に何が起こるかの専門的予測

できるだけ具体的な企業名・製品名・数値を含めてください。"""
    return call_llm("perplexity/sonar-pro", [{"role": "user", "content": prompt}])


def generate_hypotheses(domain: dict, research_results: list) -> list:
    print(f"  🧠 Opusが仮説生成中...", flush=True)
    research_text = "\n\n---\n\n".join(research_results)

    prompt = f"""あなたは世界トップレベルのヘッジファンドリサーチャーです。以下のリサーチ結果を読んで、日本株への投資仮説を生成してください。

## マクロドライバー
{domain.get('macro_driver', '')}

## リサーチ結果
{research_text}

## 重要な指針

「みんなが知っている話」は仮説ではありません。以下の視点で考えてください:
- **二次・三次効果**: 直接受益者ではなく、その先の受益者
- **ボトルネック**: サプライチェーンの中で唯一解になる企業
- **逆説的ポジション**: 「一見ネガティブ」に見える変化で実は勝つ企業
- **市場の認知lag**: 実態変化が株価に反映されるまでのタイムラグ
- **Moatの存在**: 技術・特許・顧客関係・規制で守られた参入障壁

## 出力（JSON）

```json
{{
  "hypotheses": [
    {{
      "theme": "（20字以内のキャッチーなテーマ名）",
      "insight": "（市場がまだ気づいていない核心的洞察、200字程度）",
      "logic": "（技術的・構造的な論拠、400字程度）",
      "moat_signal": "（なぜ他社ではなくこの種の企業だけが取れるか、100字）",
      "timeline": "（この仮説が顕在化するタイムライン、例: '2026年後半〜2027年'）",
      "target_company_type": "（どういう特性の会社を探すべきか）",
      "key_risks": ["リスク1", "リスク2"],
      "confidence": 0.0から1.0,
      "domain": "{domain['name']}"
    }}
  ]
}}
```

仮説は1〜3個。浅い仮説は出さない。JSONのみ出力。"""

    res = call_llm("anthropic/claude-opus-4-6", [{"role": "user", "content": prompt}], temperature=0.8)
    m = re.search(r'\{.*\}', res, re.DOTALL)
    if m:
        try:
            return json.loads(m.group()).get("hypotheses", [])
        except:
            pass
    return []


def devils_advocate(h: dict) -> dict:
    print(f"  ⚔️  反論検証: {h.get('theme', '')}", flush=True)
    prompt = f"""以下の投資仮説に対して最も鋭い反論を3つ挙げ、反論を踏まえて仮説を強化してください。

テーマ: {h.get('theme')}
洞察: {h.get('insight')}
論拠: {h.get('logic')}

```json
{{
  "counterarguments": ["反論1", "反論2", "反論3"],
  "revised_confidence": 0.0から1.0,
  "revised_logic": "（反論を踏まえて強化された論拠）",
  "leading_indicators": ["先行指標1", "先行指標2"]
}}
```
JSONのみ出力。"""

    res = call_llm("anthropic/claude-opus-4-6", [{"role": "user", "content": prompt}], temperature=0.5)
    m = re.search(r'\{.*\}', res, re.DOTALL)
    if m:
        try:
            review = json.loads(m.group())
            h["counterarguments"]  = review.get("counterarguments", [])
            h["confidence"]        = review.get("revised_confidence", h.get("confidence", 0.5))
            h["logic"]             = review.get("revised_logic", h.get("logic", ""))
            h["leading_indicators"]= review.get("leading_indicators", [])
        except:
            pass
    return h


def find_stocks(h: dict) -> list:
    print(f"  📊 銘柄マッピング中...", flush=True)
    prompt = f"""以下の投資仮説に合致する東証上場企業を3〜5社挙げてください。

テーマ: {h.get('theme')}
洞察: {h.get('insight')}
Moatシグナル: {h.get('moat_signal', '')}
対象企業タイプ: {h.get('target_company_type', '')}

厳守: 東証上場企業のみ（米国株・未上場不可）。証券コード必須。

```json
[{{
  "code": "1234",
  "name": "会社名",
  "reason": "なぜ仮説の受益者か（具体的に100字）",
  "moat": "この企業固有の参入障壁（50字）",
  "current_concern": "なぜ今まだ割安に放置されているか"
}}]
```
JSONのみ出力。"""

    res = call_llm("anthropic/claude-opus-4-6", [{"role": "user", "content": prompt}], temperature=0.6)
    m = re.search(r'\[.*\]', res, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass
    return []


def find_startups(h: dict) -> list:
    print(f"  🚀 スタートアップリサーチ中...", flush=True)
    prompt = f"""以下の投資仮説に関連する日本の有望未上場スタートアップを1〜3社。

テーマ: {h.get('theme')}
洞察: {h.get('insight')}

条件: 実在する日本の未上場企業のみ。不確かなら空配列。

```json
[{{
  "name": "スタートアップ名",
  "founded": "設立年",
  "business": "事業内容（50字）",
  "why_promising": "なぜ有望か（100字）",
  "stage": "シリーズA等",
  "investors": "主要投資家"
}}]
```
JSONのみ出力。"""

    res = call_llm("perplexity/sonar-pro", [{"role": "user", "content": prompt}], temperature=0.5)
    m = re.search(r'\[.*\]', res, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            return [s for s in data if s.get("name")]
        except:
            pass
    return []


def run():
    print("=" * 60)
    print("🔬 Deep Alpha Engine v2 起動")
    print("=" * 60)

    history_file = BASE / "backtest/alpha_domain_history.json"
    history = json.loads(history_file.read_text()) if history_file.exists() else {"last_domains": []}

    # 直近3回使っていないドメインを優先
    available = [d for d in DOMAINS if d["name"] not in history["last_domains"][-3:]]
    if not available:
        available = DOMAINS

    domain = random.choice(available)
    angles = random.sample(domain["angles"], min(2, len(domain["angles"])))

    print(f"\n📌 ドメイン: {domain['name']}")
    print(f"📌 マクロドライバー: {domain['macro_driver']}")
    print(f"📌 探索角度: {len(angles)}個\n")

    # Phase 1: リサーチ
    research_results = []
    for angle in angles:
        result = research_domain(domain, angle)
        research_results.append(f"### {angle}\n\n{result}")

    # Phase 2: 仮説生成
    hypotheses = generate_hypotheses(domain, research_results)
    if not hypotheses:
        print("⚠️ 仮説生成失敗")
        return []

    print(f"\n💡 生成仮説: {len(hypotheses)}件")

    refined = []
    for h in hypotheses:
        h["id"]         = f"alpha_{datetime.now().strftime('%Y%m%d')}_{len(refined)+1:03d}"
        h["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        h = devils_advocate(h)
        h["candidate_stocks"] = find_stocks(h)
        h["startups"]         = find_startups(h)
        refined.append(h)
        print(f"\n  ✅ {h.get('theme')} (確信度: {h.get('confidence', 0):.0%})", flush=True)

    # 保存（最新30件）
    existing = []
    if OUTPUT.exists():
        try:
            existing = json.loads(OUTPUT.read_text()).get("hypotheses", [])
        except:
            pass

    all_hyps = refined + existing
    all_hyps = all_hyps[:30]

    OUTPUT.write_text(json.dumps({
        "updated_at":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "last_domain":   domain["name"],
        "hypotheses":    all_hyps,
    }, ensure_ascii=False, indent=2))

    history["last_domains"].append(domain["name"])
    history["last_domains"] = history["last_domains"][-10:]
    history_file.write_text(json.dumps(history))

    print(f"\n{'='*60}")
    print(f"✨ 完了: 仮説{len(refined)}件生成")
    print(f"{'='*60}\n")
    return refined


if __name__ == "__main__":
    run()
