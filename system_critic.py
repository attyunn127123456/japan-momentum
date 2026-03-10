"""
System Critic Agent — Deep Alpha Engine 自己改善エージェント

トップティアHFマネージャーの目線でシステム全体を批判的にレビューし、
改善提案を improvements/YYYY-MM-DD.md に蓄積し続ける。

コスト: gpt-4.1-mini × 1日2回 ≈ ¥2/日
実行: cron または手動
"""

import json, httpx, re
from pathlib import Path
from datetime import datetime

BASE     = Path(__file__).parent
SECRETS  = BASE / ".secrets/accounts.json"
OR_URL   = "https://openrouter.ai/api/v1/chat/completions"
CRITIC_MODEL = "openai/gpt-4.1-mini"   # 安くて十分
NOTIFY_MODEL = "anthropic/claude-opus-4-6"  # 重要提案の要約のみ

IMPROVEMENTS_DIR = BASE / "improvements"
IMPROVEMENTS_DIR.mkdir(exist_ok=True)

DISCORD_WEBHOOK_ENV = "DISCORD_USER_ID"
DISCORD_USER_ID = "717228195161571459"

# ── ヘルパー ──────────────────────────────────────────────────
def get_key():
    if SECRETS.exists():
        return json.loads(SECRETS.read_text())["openrouter"]["api_key"]
    return ""

def call_llm(model, messages, temperature=0.7, timeout=120):
    r = httpx.post(OR_URL,
        headers={"Authorization": f"Bearer {get_key()}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def load_context() -> dict:
    """システムのコードと最新出力を読み込む"""
    ctx = {}

    # コードファイル（最初の200行だけ読む → コスト節約）
    for fname in ["deep_alpha_engine.py", "evaluate_hypothesis.py"]:
        p = BASE / fname
        if p.exists():
            lines = p.read_text().splitlines()[:200]
            ctx[fname] = "\n".join(lines)

    # 最新評価済み仮説（直近5件）
    eval_path = BASE / "backtest/evaluated_hypotheses.json"
    if eval_path.exists():
        try:
            data = json.loads(eval_path.read_text())
            hyps = (data.get("ranked_hypotheses") or [])[:5]
            ctx["latest_evaluated"] = json.dumps(hyps, ensure_ascii=False, indent=2)
        except:
            ctx["latest_evaluated"] = "読み込みエラー"

    # 最新生成仮説（直近3件）
    hyp_path = BASE / "backtest/macro_hypotheses.json"
    if hyp_path.exists():
        try:
            data = json.loads(hyp_path.read_text())
            hyps = (data.get("hypotheses") or [])[:3]
            # コードとinsightのみ（サイズ削減）
            slim = [{
                "theme": h.get("theme"),
                "insight": h.get("insight"),
                "moat_signal": h.get("moat_signal"),
                "confidence": h.get("confidence"),
                "domain": h.get("domain"),
            } for h in hyps]
            ctx["latest_hypotheses"] = json.dumps(slim, ensure_ascii=False, indent=2)
        except:
            ctx["latest_hypotheses"] = "読み込みエラー"

    # 過去の改善提案（直近2ファイル）
    past = sorted(IMPROVEMENTS_DIR.glob("*.md"), reverse=True)[:2]
    ctx["past_improvements"] = ""
    for p in past:
        text = p.read_text()
        ctx["past_improvements"] += f"\n### {p.name}\n{text[:800]}\n"

    return ctx

def generate_critique(ctx: dict) -> str:
    """HFマネージャー目線でシステムを批判的レビュー"""

    past_section = f"""
## 過去の改善提案（既出のものは出さない）
{ctx.get('past_improvements', 'なし')}
""" if ctx.get("past_improvements") else ""

    prompt = f"""あなたはQuilon Capital / 丸の内フロント線でトレードを経験した、
10年以上日本株ロング/ショートを運用してきたシニアポートフォリオマネージャーです。

以下のシステムコードと出力を見て、「真のアルファを見つけ出すツール」として
このシステムが持つ**構造的な欠陥・改善余地**を批判的に指摘してください。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## deep_alpha_engine.py（抜粋）
```python
{ctx.get('deep_alpha_engine.py', 'N/A')[:3000]}
```

## evaluate_hypothesis.py（抜粋）
```python
{ctx.get('evaluate_hypothesis.py', 'N/A')[:3000]}
```

## 直近生成仮説（例）
```json
{ctx.get('latest_hypotheses', 'N/A')}
```

## 直近評価結果（例）
```json
{ctx.get('latest_evaluated', 'N/A')[:2000]}
```
{past_section}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

以下のカテゴリで改善提案を出してください（各カテゴリ1〜3件、重複なし）:

1. **Alpha Quality**（仮説の鋭さ・市場との情報非対称性・差別化）
2. **Evaluation Rigor**（Market Sizing精度・Moat評価の深さ・Valuation手法の穴）
3. **Data Edge**（使えていないデータ・J-Quantsの使い方・代替データ源）
4. **Process**（コスト効率・精度向上・フィードバックループ・エラー処理）
5. **Risk Management**（仮説の破綻条件・ポジションサイズ・相関リスク）

各提案は以下の形式で:
- **重要度**: 🔴高 / 🟡中 / 🟢低
- **問題**: 何が足りないか（具体的に）
- **解決策**: どう直すか（実装レベルで）
- **期待効果**: なぜアルファ発見につながるか

HFの視点から遠慮なく厳しく指摘してください。「良い点」は不要です。"""

    return call_llm(CRITIC_MODEL, [{"role": "user", "content": prompt}])


def save_to_file(critique: str) -> Path:
    """improvements/YYYY-MM-DD.md に追記"""
    today = datetime.now().strftime("%Y-%m-%d")
    now   = datetime.now().strftime("%H:%M")
    path  = IMPROVEMENTS_DIR / f"{today}.md"

    header = f"\n\n---\n\n## {now} JST — システム改善提案\n\n"
    with open(path, "a") as f:
        f.write(header + critique + "\n")

    return path


def extract_high_priority(critique: str) -> list:
    """🔴高 の提案だけ抽出"""
    lines = critique.split("\n")
    high = []
    current = []
    in_high = False
    for line in lines:
        if "🔴" in line or "🔴高" in line:
            in_high = True
            current = [line]
        elif in_high and (line.startswith("- **重要度**") or line.startswith("---") or (line.startswith("## ") and "問題" not in line)):
            if current:
                high.append("\n".join(current))
            current = []
            in_high = False
        elif in_high:
            current.append(line)
    if current and in_high:
        high.append("\n".join(current))
    return high


def notify_discord(high_priority_items: list, saved_path: Path):
    """高優先度の提案をDiscordでパンダに通知"""
    if not high_priority_items:
        print("  高優先度提案なし → Discordスキップ")
        return

    items_text = "\n\n".join(high_priority_items[:3])  # 最大3件
    date_str = datetime.now().strftime("%m/%d %H:%M")

    msg = f"""🔴 **システム改善提案** ({date_str})

HFクリティックから高優先度の改善点が出ました。

{items_text}

📄 全提案: `improvements/{saved_path.name}`
→ 実装するものがあれば指示をください"""

    # OpenClaw経由でDiscord送信（スクリプトからはcurl不可なのでファイルに保存）
    notify_path = BASE / "improvements/.pending_notify.json"
    notify_path.write_text(json.dumps({
        "message": msg,
        "user_id": DISCORD_USER_ID,
        "timestamp": datetime.now().isoformat(),
    }, ensure_ascii=False))
    print(f"  通知ファイル保存: {notify_path}")


def run():
    print("=" * 55)
    print("🔍 System Critic Agent 起動")
    print(f"   モデル: {CRITIC_MODEL}")
    print("=" * 55)

    print("\n📂 コンテキスト読み込み中...")
    ctx = load_context()
    print(f"  読み込み完了 ({len(ctx)}ファイル)")

    print("\n🧠 HFクリティック分析中...")
    critique = generate_critique(ctx)

    print("\n💾 保存中...")
    saved = save_to_file(critique)
    print(f"  → {saved}")

    high = extract_high_priority(critique)
    print(f"\n🔴 高優先度提案: {len(high)}件")

    notify_discord(high, saved)

    print("\n" + "=" * 55)
    print("✅ 完了")
    print("=" * 55)
    print("\n--- 改善提案（先頭500字）---")
    print(critique[:500])


if __name__ == "__main__":
    run()
