"""Send daily momentum report to Discord DM"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests


def send_discord_dm(token: str, user_id: str, message: str) -> bool:
    headers = {"Authorization": f"Bot {token}", "Content-Type": "application/json"}
    # Open DM channel
    r = requests.post(
        "https://discord.com/api/v10/users/@me/channels",
        headers=headers,
        json={"recipient_id": user_id},
    )
    if r.status_code not in (200, 201):
        print(f"Failed to open DM: {r.status_code} {r.text}")
        return False
    channel_id = r.json()["id"]
    # Send message
    r2 = requests.post(
        f"https://discord.com/api/v10/channels/{channel_id}/messages",
        headers=headers,
        json={"content": message},
    )
    return r2.status_code in (200, 201)


def format_vol(v: float) -> str:
    if v >= 0.9: return "↑↑↑"
    if v >= 0.6: return "↑↑"
    if v >= 0.33: return "↑"
    return "→"


def build_message(results_path: Path = None) -> str:
    if results_path is None:
        results_path = Path("results/latest.json")
    if not results_path.exists():
        return "❌ 結果ファイルが見つかりません"

    with open(results_path) as f:
        data = json.load(f)

    date = data.get("date", datetime.now().strftime("%Y-%m-%d"))
    top = data.get("top", data.get("results", []))[:10]

    medals = ["🥇", "🥈", "🥉"] + ["　"] * 10
    lines = [f"📊 **今日のモメンタム銘柄 {date}**\n"]

    for i, r in enumerate(top):
        ticker = r["ticker"].replace(".T", "")
        medal = medals[i] if i < 3 else f"{i+1}."
        lines.append(
            f"{medal} **{ticker}** | スコア: {r['score']:.1f}\n"
            f"　中期リターン: {r['return_5_25d']:+.1f}% | 出来高: {format_vol(r['volume_acceleration'])} | 陽線率: {r['green_day_ratio']*100:.0f}%"
        )

    lines.append("\n🔄 次回更新: 明朝9:30 JST")
    return "\n".join(lines)


if __name__ == "__main__":
    token = os.environ.get("DISCORD_BOT_TOKEN")
    user_id = os.environ.get("DISCORD_USER_ID")

    if not token or not user_id:
        print("ERROR: DISCORD_BOT_TOKEN and DISCORD_USER_ID required")
        sys.exit(1)

    msg = build_message()
    print(msg)
    print("\nSending to Discord...")
    ok = send_discord_dm(token, user_id, msg)
    print("✅ Sent!" if ok else "❌ Failed")
