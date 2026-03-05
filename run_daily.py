"""Daily orchestrator: screener + signal check + discord notify"""
import argparse
import json
from datetime import datetime
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-discord", action="store_true")
    parser.add_argument("--skip-screener", action="store_true")
    args = parser.parse_args()

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Daily Run ===")

    # 1. スクリーニング
    if not args.skip_screener:
        from screener import run_screener
        top = run_screener()
    else:
        print("スクリーニング: スキップ")

    # 2. 保有銘柄シグナルチェック
    portfolio_path = Path("portfolio.json")
    signal_results = []
    if portfolio_path.exists():
        holdings = json.loads(portfolio_path.read_text()).get("holdings", [])
        if holdings:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] シグナルチェック: {holdings}")
            from daily_signal import run_signal_check, format_report
            signal_results = run_signal_check(holdings)
        else:
            print("保有銘柄なし（portfolio.json に登録してください）")
    else:
        print("portfolio.json なし（--holdings で作成: python3 daily_signal.py --holdings 5803,7011）")

    # 3. Discord通知（スクリーニング結果 + シグナル）
    if not args.skip_discord:
        import os
        token = os.environ.get("DISCORD_BOT_TOKEN")
        user_id = os.environ.get("DISCORD_USER_ID")
        if token and user_id:
            from discord_notify import build_message, send_discord_dm, format_report as fmt_sig
            # ランキングレポート
            msg = build_message()
            send_discord_dm(token, user_id, msg)
            print("Discord ランキング: ✅")
            # シグナルレポート（保有銘柄あれば）
            if signal_results:
                from daily_signal import format_report
                sig_msg = format_report(signal_results, datetime.now().strftime("%Y-%m-%d"))
                send_discord_dm(token, user_id, sig_msg)
                print("Discord シグナル: ✅")
        else:
            print("Discord: スキップ（環境変数未設定）")

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Done.")
