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
CRITIC_MODEL = "anthropic/claude-sonnet-4-6"  # 批評・判断にはSonnet以上
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


# ─────────────────────────────────────────────────────────────
# PANDA JUDGE — パンダが提案を読んで実装判断するレイヤー
# ─────────────────────────────────────────────────────────────

JUDGE_RULES = """
判断基準:
- 実装コスト（コード変更量）が小さい → 優先
- APIコスト増加が少ない → 優先
- 仮説の質・評価精度に直結する → 優先
- 「いつか」「将来的に」な提案 → 今回はスキップ
- 既に実装済みのもの → スキップ
- 外部データソース追加（課金必要）→ スキップ（あつしに確認してから）
"""

def judge_and_plan(critique: str, past_implementations: str) -> list:
    """今週の改善提案から実装するものを選んで具体的な実装計画を立てる"""
    prompt = f"""あなたはDeep Alpha Engineの開発者 兼 CEOです。
以下のHFクリティックの改善提案を読んで、今すぐ自分で実装すべきものを選んでください。

## 改善提案
{critique[:4000]}

## 判断基準
{JUDGE_RULES}

## 過去の実装（重複スキップ）
{past_implementations[:1000] if past_implementations else 'なし'}

## プロジェクト構成
- deep_alpha_engine.py (仮説生成)
- evaluate_hypothesis.py (HF評価・3段階)
- dashboard/app.py (FastAPI)
- dashboard/static/index.html (UI)

## 出力（JSON）
以下の形式で「今回実装するもの」を1〜2件選んでください:
```json
[{{
  "title": "実装タイトル",
  "category": "alpha_quality|evaluation|data|process|risk",
  "why_now": "なぜ今実装すべきか（1行）",
  "file_to_change": "変更するファイル名",
  "implementation": "具体的に何を変更するか（コードレベルで詳細に）",
  "estimated_impact": "high|medium|low",
  "api_cost_delta": "none|minimal|moderate"
}}]
```
JSONのみ出力。実装困難・コスト増大のものは選ばないこと。"""

    res = call_llm(CRITIC_MODEL, [{"role": "user", "content": prompt}], temperature=0.3)
    m = re.search(r'\[.*\]', res, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass
    return []


def implement_improvement(plan: dict) -> bool:
    """計画に基づいて実際にコードを変更する（Opusが実装）"""
    file_path = BASE / plan["file_to_change"]
    if not file_path.exists():
        print(f"    ファイル不存在: {plan['file_to_change']}", flush=True)
        return False

    current_code = file_path.read_text()

    prompt = f"""以下の改善計画に基づいて、Pythonファイルを修正してください。

## 改善タイトル
{plan['title']}

## 実装内容
{plan['implementation']}

## 現在のコード
```python
{current_code[:6000]}
```

## 要求
1. 修正後のコードを **完全に** 出力してください（省略なし）
2. 変更点以外は一切触らない
3. 既存の機能を壊さない
4. コードのみ出力（説明不要）

```python
（修正後のコード全体をここに）
```"""

    res = call_llm("anthropic/claude-opus-4-6", [{"role": "user", "content": prompt}], temperature=0.2, timeout=300)

    # コードブロックを抽出
    m = re.search(r'```python\s*(.*?)```', res, re.DOTALL)
    if not m:
        m = re.search(r'```\s*(.*?)```', res, re.DOTALL)
    if not m:
        print(f"    コード抽出失敗", flush=True)
        return False

    new_code = m.group(1).strip()
    if len(new_code) < 100:
        print(f"    コードが短すぎる: {len(new_code)}文字", flush=True)
        return False

    # バックアップ
    backup = file_path.with_suffix(f".bak_{datetime.now().strftime('%H%M')}")
    backup.write_text(current_code)

    # 構文チェック
    try:
        import ast
        ast.parse(new_code)
    except SyntaxError as e:
        print(f"    構文エラー: {e}", flush=True)
        return False

    file_path.write_text(new_code)
    print(f"    ✅ {plan['file_to_change']} を更新（バックアップ: {backup.name}）", flush=True)
    return True


def load_past_implementations() -> str:
    """過去の実装ログを読む"""
    log_path = BASE / "improvements/implementation_log.md"
    if log_path.exists():
        return log_path.read_text()[-2000:]
    return ""


def save_implementation_log(plans: list, results: list):
    """実装ログを保存"""
    log_path = BASE / "improvements/implementation_log.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(log_path, "a") as f:
        f.write(f"\n\n## {now}\n")
        for plan, success in zip(plans, results):
            status = "✅ 実装済み" if success else "❌ スキップ"
            f.write(f"- {status}: {plan['title']} ({plan['file_to_change']})\n")
            f.write(f"  理由: {plan['why_now']}\n")


def run_with_auto_implement():
    """批評 → 判断 → 実装 → 報告 の完全自律ループ"""
    print("=" * 55)
    print("🤖 System Critic + Auto-Implement 起動")
    print("=" * 55)

    # 1. コンテキスト読み込み
    print("\n📂 コンテキスト読み込み中...")
    ctx = load_context()

    # 2. HFクリティック分析
    print("\n🧠 HFクリティック分析中...")
    critique = generate_critique(ctx)

    # 3. ファイル保存
    print("\n💾 提案保存中...")
    saved = save_to_file(critique)
    print(f"  → {saved}")

    # 4. パンダが判断（実装するものを選ぶ）
    print("\n🐼 パンダが実装判断中...")
    past = load_past_implementations()
    plans = judge_and_plan(critique, past)
    print(f"  実装対象: {len(plans)}件")
    for p in plans:
        print(f"  - {p.get('title')} ({p.get('file_to_change')})", flush=True)

    # 5. 実装
    results = []
    for plan in plans:
        print(f"\n🔧 実装中: {plan.get('title')}", flush=True)
        success = implement_improvement(plan)
        results.append(success)

    # 6. 実装ログ保存
    if plans:
        save_implementation_log(plans, results)

    # 7. git commit（成功したものがあれば）
    successful = [p for p, r in zip(plans, results) if r]
    if successful:
        import subprocess
        titles = " / ".join(p["title"] for p in successful)
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(BASE), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", f"auto: System Critic自動改善 — {titles}"],
            cwd=str(BASE), capture_output=True
        )
        subprocess.run(
            ["git", "push"],
            cwd=str(BASE), capture_output=True
        )
        print(f"\n✅ git commit + push 完了", flush=True)

    # 8. Discordに結果報告（完了サマリーのみ）
    now_str = datetime.now().strftime("%m/%d %H:%M")
    if successful:
        impl_list = "\n".join(f"• {p['title']}" for p in successful)
        msg = f"🔧 **自動改善完了** ({now_str})\n\n{impl_list}\n\n詳細: `improvements/implementation_log.md`"
    else:
        msg = f"🔍 **HFレビュー完了** ({now_str}) — 今回実装する改善なし（または既実装済み）"

    notify_path = BASE / "improvements/.pending_notify.json"
    notify_path.write_text(json.dumps({
        "message": msg,
        "user_id": DISCORD_USER_ID,
        "timestamp": datetime.now().isoformat(),
    }, ensure_ascii=False))

    print("\n" + "=" * 55)
    print(f"✅ 完了 / 実装: {sum(results)}/{len(results)}")
    print("=" * 55)


if __name__ == "__main__":
    import sys
    if "--auto" in sys.argv:
        run_with_auto_implement()
    else:
        run()
