# Japan Momentum Screener 🐼

日本株テーマモメンタム投資スクリーナー

## アーキテクチャ

```
Mac mini (常時稼働)
├── screener.py     毎朝0:30 UTC (9:30 JST) cron実行
├── daily_signal.py 前日比シグナル生成
├── dashboard/      FastAPI ポート8080
└── cloudflared     Cloudflare Tunnel → 外部公開
```

## セットアップ

### 1. 依存パッケージ
```bash
pip3 install -r requirements.txt
```

### 2. Cloudflare Tunnelセットアップ（初回のみ）
```bash
# ログイン（ブラウザが開く）
cloudflared tunnel login

# トンネル作成
cloudflared tunnel create japan-momentum

# 作成されたトンネルIDを infra/tunnel.yml に記入
# credentials-file のパスも確認
```

### 3. cron設定（毎朝9:30 JST自動実行）
```bash
bash infra/setup_cron.sh
```

### 4. ダッシュボード常時起動
```bash
# 手動起動（テスト用）
bash infra/start.sh

# LaunchAgent登録（Mac再起動後も自動起動）
launchctl load ~/Library/LaunchAgents/io.akplabo.japan-momentum.plist
```

## 使い方

### 手動実行
```bash
python3 screener.py              # スクリーニング
python3 daily_signal.py          # シグナル生成
python3 run_daily.py             # 全部まとめて
python3 run_daily.py --skip-push # git pushなし
```

### バックテスト
```bash
python3 backtest.py
python3 backtest.py --start 2022-01-01 --top-n 5 --rebalance weekly
```

### ログ確認
```bash
tail -f logs/daily.log
tail -f logs/dashboard.log
```

## スコアロジック

```
momentum_score =
  return_5_25d * 0.40    # 5〜25日前の累積リターン（直近3日除外）
  + volume_acceleration * 0.30  # 週次出来高トレンド
  + green_day_ratio * 0.20      # 直近25日の陽線比率
  + rs_acceleration * 0.10      # RSスコアの1ヶ月前比改善度
```

## 環境変数（オプション）
```bash
export JQUANTS_API_KEY=your_key      # デフォルト値あり
export DISCORD_BOT_TOKEN=your_token  # Discord通知用
export DISCORD_USER_ID=717228195161571459
```
