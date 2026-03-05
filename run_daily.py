"""Daily orchestrator: screener + discord notify"""
import argparse
import sys
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-discord", action="store_true")
    args = parser.parse_args()

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting daily run...")

    from screener import run_screener
    top = run_screener()

    if not args.skip_discord:
        from discord_notify import build_message, send_discord_dm
        import os
        token = os.environ.get("DISCORD_BOT_TOKEN")
        user_id = os.environ.get("DISCORD_USER_ID")
        if token and user_id:
            msg = build_message()
            ok = send_discord_dm(token, user_id, msg)
            print("Discord: ✅" if ok else "Discord: ❌")
        else:
            print("Discord: スキップ（環境変数未設定）")
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Done.")
