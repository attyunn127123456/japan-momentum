# Japan Momentum Screener 🐼

日本株テーマモメンタム投資スクリーナー

## セットアップ

```bash
pip install -r requirements.txt
```

## 環境変数（Discord通知用）

```bash
export DISCORD_BOT_TOKEN=your_bot_token
export DISCORD_USER_ID=717228195161571459
```

## 使い方

### 毎日スクリーニング実行
```bash
python run_daily.py           # スクリーニング + Discord通知
python run_daily.py --skip-discord  # 通知なし（テスト用）
```

### スクリーニングのみ
```bash
python screener.py
```

### バックテスト（過去3年）
```bash
python backtest.py
python backtest.py --start 2022-01-01 --top-n 5 --rebalance weekly
python backtest.py --start 2023-01-01 --top-n 10 --rebalance monthly
```

### ダッシュボード起動
```bash
python dashboard/app.py
# → http://localhost:8080
```

## Cron設定（毎朝9:30 JST = 0:30 UTC）

```cron
30 0 * * 1-5 cd /Users/panda/Projects/japan-momentum && python run_daily.py >> logs/daily.log 2>&1
```

## スコアロジック

```
momentum_score =
  return_5_25d * 0.40    # 5〜25日前の累積リターン（直近3日除外でポンプ銘柄を排除）
  + volume_acceleration * 0.30  # 週次出来高トレンド（W1<W2<W3<W4）
  + green_day_ratio * 0.20      # 直近25日の陽線比率
  + rs_acceleration * 0.10      # RSスコアの1ヶ月前比改善度
```

RS（相対強度）= 日経225対比の加重平均リターン
- 63日: 40%、126日: 30%、252日: 20%、21日: 10%
