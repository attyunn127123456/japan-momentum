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

# ── Catalyst Chain スキーマ定義 ─────────────────────────────────────
CATALYST_CHAIN_SCHEMA = {
    "next_catalyst_event": "(str) 次回決算発表/展示会/規制発効等の具体的イベント名",
    "catalyst_date": "(str) YYYY-MM-DD or 'TBD'",
    "days_until_catalyst": "(int|null) catalyst_dateまでの日数。TBDの場合はnull",
    "key_kpi_to_watch": "(str) 受注残/ASP/稼働率等、注目すべきKPI",
    "kpi_surprise_rationale": "(str) 市場予想を上回る根拠",
    "expected_price_mechanism": "(str) 株価反応の想定メカニズム"
}

# ── Short Candidates スキーマ定義 ───────────────────────────────────
SHORT_CANDIDATES_SCHEMA = {
    "company_name": "(str) 企業名",
    "ticker": "(str) ティッカーシンボル（日本上場企業優先、例: 6502.T）",
    "impacted_segment": "(str) 毀損されるセグメント名",
    "damage_mechanism": "(str) なぜ構造的に毀損するか",
    "estimated_downside_pct": "(float) 想定下落率（例: -15.0）"
}

CATALYST_CHAIN_SYSTEM_INSTRUCTION = (
    "Catalyst Chainが具体的でない仮説は投資判断に使えない。"
    "必ず3ヶ月以内の具体的イベントを特定し、そのイベントで開示されるKPIと市場予想との乖離の根拠を明記せよ。"
    "カタリストが特定できない場合はcatalyst_dateを'TBD'とし、confidence_scoreを0.15減点せよ。"
    "出力JSONには必ず以下のcatalyst_chainオブジェクトを含めること:\n"
    "  catalyst_chain: {\n"
    "    next_catalyst_event: (str) 次回決算発表/展示会/規制発効等,\n"
    "    catalyst_date: (str) YYYY-MM-DD or 'TBD',\n"
    "    days_until_catalyst: (int|null),\n"
    "    key_kpi_to_watch: (str) 受注残/ASP/稼働率等,\n"
    "    kpi_surprise_rationale: (str) 市場予想を上回る根拠,\n"
    "    expected_price_mechanism: (str) 株価反応の想定メカニズム\n"
    "  }\n"
)

SHORT_CANDIDATES_SYSTEM_INSTRUCTION = (
    "各仮説について、この構造変化によって売上・利益が構造的に毀損される企業を1〜3社特定せよ。"
    "ロング候補の受益メカニズムの裏返しとして、代替される技術・製品・サービスを提供する企業、"
    "またはサプライチェーン上で需要が減少する企業を具体的に挙げよ。"
    "日本上場企業を優先し、ティッカーを明記せよ。"
    "出力JSONには必ず以下のshort_candidates配列を含めること（最大3社）:\n"
    "  short_candidates: [\n"
    "    {\n"
    "      company_name: (str) 企業名,\n"
    "      ticker: (str) ティッカーシンボル（例: 6502.T）,\n"
    "      impacted_segment: (str) 毀損されるセグメント名,\n"
    "      damage_mechanism: (str) なぜ構造的に毀損するか,\n"
    "      estimated_downside_pct: (float) 想定下落率（例: -15.0）\n"
    "    }\n"
    "  ]\n"
)

# ── Post-processing: Catalyst Chain enrichment & penalty ────────────
def _enrich_hypothesis(hypothesis):
    """catalyst_chain の days_until_catalyst を自動計算し、ペナルティを適用する。"""
    cc = hypothesis.get("catalyst_chain")
    if not cc:
        # catalyst_chain が欠落している場合はデフォルトを補完
        cc = {
            "next_catalyst_event": "Unknown",
            "catalyst_date": "TBD",
            "days_until_catalyst": None,
            "key_kpi_to_watch": "N/A",
            "kpi_surprise_rationale": "N/A",
            "expected_price_mechanism": "N/A"
        }
        hypothesis["catalyst_chain"] = cc

    catalyst_date_str = cc.get("catalyst_date", "TBD")
    apply_penalty = False

    if catalyst_date_str and catalyst_date_str != "TBD":
        try:
            catalyst_dt = datetime.strptime(catalyst_date_str, "%Y-%m-%d")
            days_until = (catalyst_dt - datetime.now()).days
            cc["days_until_catalyst"] = days_until
            if days_until > 90:
                apply_penalty = True
        except (ValueError, TypeError):
            cc["days_until_catalyst"] = None
            apply_penalty = True
    else:
        cc["catalyst_date"] = "TBD"
        cc["days_until_catalyst"] = None
        apply_penalty = True

    if apply_penalty:
        current_score = hypothesis.get("confidence_score", 0.5)
        if isinstance(current_score, (int, float)):
            hypothesis["confidence_score"] = round(current_score * 0.85, 4)

    # short_candidates が欠落している場合はデフォルトを補完
    if "short_candidates" not in hypothesis or not isinstance(hypothesis.get("short_candidates"), list):
        hypothesis["short_candidates"] = []

    # short_candidates の各要素を検証・補完
    validated_shorts = []
    for sc in hypothesis.get("short_candidates", []):
        if isinstance(sc, dict):
            validated_sc = {
                "company_name": sc.get("company_name", "Unknown"),
                "ticker": sc.get("ticker", "N/A"),
                "impacted_segment": sc.get("impacted_segment", "N/A"),
                "damage_mechanism": sc.get("damage_mechanism", "N/A"),
                "estimated_downside_pct": sc.get("estimated_downside_pct", 0.0)
            }
            # estimated_downside_pct が数値であることを保証
            if not isinstance(validated_sc["estimated_downside_pct"], (int, float)):
                try:
                    validated_sc["estimated_downside_pct"] = float(validated_sc["estimated_downside_pct"])
                except (ValueError, TypeError):
                    validated_sc["estimated_downside_pct"] = 0.0
            validated_shorts.append(validated_sc)
    hypothesis["short_candidates"] = validated_shorts[:3]  # 最大3社

    return hypothesis

# ── ドメイン定義（高市政権重点領域 × グローバルトレンド）────────────────
#  大カテゴリ（高市政権の4大柱）
#    A. 経済安全保障  B. エネルギー安全保障  C. 防衛安全保障  D. 成長戦略
#  実行順序: policy_tailwind「強」(9個) → 「中」(6個) の固定ローテーション
# ─────────────────────────────────────────────────────────────────────
DOMAINS = [
  {
    "name": "半導体・電子材料の国産強靭化",
    "category": "経済安全保障",
    "policy_tailwind": "強",
    "macro_driver": "経済安全保障法×AI需要爆発で補助金10兆円規模、内製化加速",
    "angles": [
      "TSMC熊本・ラピダス千歳で日本回帰した半導体工場の地元サプライヤーは誰が勝つか",
      "EUV High-NA量産移行でレジスト・ペリクル・光学系部品の供給制約と日本寡占企業",
      "HBM・CoWoS拡大で需要急増するTSV工程の日本化学・素材メーカーの寡占ポジション",
      "ラピダスが2026年量産を目指す2nmプロセスでのボトルネックとサプライヤー"
    ]
  },
  {
    "name": "重要鉱物・サプライチェーン脱中国",
    "category": "経済安全保障",
    "policy_tailwind": "強",
    "macro_driver": "中国レアアース輸出規制×経済安保法の重要物資指定で代替投資急増",
    "angles": [
      "中国がレアアース輸出規制を強化した場合に日本の素材・部品メーカーで代替できる企業",
      "海外鉱山権益確保・リサイクル技術で日本が優位に立てる重要鉱物分野",
      "EVバッテリーCOBC（中国外製造）要件対応で恩恵を受ける日本の素材・加工企業",
      "半導体製造に必要な希ガス（ネオン・クリプトン）の日本内製化と受益企業"
    ]
  },
  {
    "name": "原発再稼働・次世代原子力",
    "category": "エネルギー安全保障",
    "policy_tailwind": "強",
    "macro_driver": "高市政権が原発最大限活用を明言、SMR開発予算・再稼働加速で関連投資急増",
    "angles": [
      "原発再稼働加速で日本の電力会社の収益構造が変わる。隠れた受益者（設備・燃料・廃炉）",
      "小型モジュール炉（SMR）の2030年代量産化で日本の重工・素材メーカーが担う部分",
      "核融合（民間：TAE/Helion）への部品・素材供給で日本企業が取れるニッチポジション",
      "廃炉ビジネス拡大（国内40基以上）で受益する解体・廃棄物処理・ロボット企業"
    ]
  },
  {
    "name": "GX・電力インフラ刷新",
    "category": "エネルギー安全保障",
    "policy_tailwind": "強",
    "macro_driver": "GX国債20兆円×電力需要急増で送配電投資ブーム、老朽設備更新が急務",
    "angles": [
      "AI・DC電力需要爆発で送配電設備更新が急務。受益する重電・ケーブル企業",
      "洋上風力の国産化要件で恩恵を受ける日本の部品・素材・建設船メーカー",
      "全固体電池の量産ロードマップ（2027-2030）と日本の素材メーカーの立ち位置",
      "系統安定化向け大型蓄電池（LFP/NMC）の急拡大で受益する日本企業"
    ]
  },
  {
    "name": "防衛産業強化・デュアルユース",
    "category": "防衛安全保障",
    "policy_tailwind": "強",
    "macro_driver": "防衛費GDP2%（年11兆円）前倒し×防衛装備移転解禁で日本防衛産業が急拡大",
    "angles": [
      "防衛費倍増で最も受益する防衛専業・デュアルユース企業（電子戦・ドローン・ミサイル）",
      "宇宙防衛（衛星・宇宙状況監視）の予算急増で受益する日本の宇宙・通信企業",
      "サイバーセキュリティの国防投資拡大で恩恵を受けるセキュリティ企業",
      "防衛装備品の輸出解禁（次期戦闘機GCAP）で受益するサプライヤー"
    ]
  },
  {
    "name": "宇宙・衛星インフラ",
    "category": "防衛安全保障",
    "policy_tailwind": "強",
    "macro_driver": "宇宙安全保障予算拡大×H3量産×民間宇宙商業化で日本の宇宙産業が転換点",
    "angles": [
      "H3ロケット量産化で打ち上げコスト低下、恩恵を受ける日本の衛星・部品メーカー",
      "低軌道衛星コンステレーションで受益する日本の部品・素材・地上局企業",
      "衛星SAR・光学データの政府調達拡大で恩恵を受ける日本の衛星データ企業",
      "宇宙デブリ除去（アストロスケール等）の商業化と日本企業の競争優位"
    ]
  },
  {
    "name": "国土強靭化・インフラ更新",
    "category": "防衛安全保障",
    "policy_tailwind": "強",
    "macro_driver": "国土強靭化計画5年15兆円×老朽インフラ更新ラッシュ×南海トラフ対策",
    "angles": [
      "老朽インフラ（橋梁・トンネル・上下水道）の更新需要で受益する建設・素材企業",
      "南海トラフ地震対策の防災インフラ投資で受益する企業",
      "建設DX（BIM/CIM・自動施工）の普及で受益するテクノロジー企業",
      "水インフラ（上下水道）の老朽化対策で受益する管材・ポンプ・膜メーカー"
    ]
  },
]