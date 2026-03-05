"""Daily orchestrator: screener + signal + discord + git push"""
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

def git_push():
    """結果をGitHubにpush（Vercelが自動でデプロイ）"""
    try:
        subprocess.run(["git", "add", "results/", "data/signal_history.json", "backtest/"], check=False)
        subprocess.run(["git", "commit", "-m", f"data: {datetime.now().strftime('%Y-%m-%d')} daily update"], check=False)
        subprocess.run(["git", "push"], check=False)
        print("GitHub push: ✅")
    except Exception as e:
        print(f"GitHub push: ❌ {e}")

def update_index():
    """results/index.json を更新（Vercel用日付一覧）"""
    results_dir = Path("results")
    dates = sorted([f.stem for f in results_dir.glob("????-??-??.json")], reverse=True)
    (results_dir / "index.json").write_text(json.dumps(dates))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-discord", action="store_true")
    parser.add_argument("--skip-screener", action="store_true")
    parser.add_argument("--skip-push", action="store_true")
    args = parser.parse_args()

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Daily Run ===")

    # 1. スクリーニング
    if not args.skip_screener:
        from screener import run_screener
        run_screener()
        update_index()
    
    # 2. シグナルチェック（前日比較）
    from daily_signal import run_signal_check
    run_signal_check()

    # 3. Discord通知
    if not args.skip_discord:
        import os
        token = os.environ.get("DISCORD_BOT_TOKEN")
        user_id = os.environ.get("DISCORD_USER_ID")
        if token and user_id:
            from discord_notify import build_message, send_discord_dm
            send_discord_dm(token, user_id, build_message())
            print("Discord: ✅")

    # 4. GitHubにpush → Vercelが自動デプロイ
    if not args.skip_push:
        git_push()

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Done.")
