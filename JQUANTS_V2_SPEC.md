# J-Quants API V2 仕様メモ（絶対忘れるな）

## 認証方式（V2）
- **APIキー方式**（V1のトークン方式ではない）
- ヘッダー: `x-api-key: <APIキー>`
- APIキーに有効期限なし（V1のID Token/Refresh Tokenは廃止）
- ベースURL: `https://api.jquants.com/v2`

## 現在のAPIキー
```
cph3PdiF8zxH9GxClcFfShcJdSUzuNpV9ho_zMPm4a8
```
※ プラン: **Premium**（最高プラン契約済み）
※ レートリミット: **500リクエスト/分**

---

## 正しいエンドポイント一覧（V2）

### 株価・銘柄
| データ | V2エンドポイント | 備考 |
|---|---|---|
| 株価四本値 | `/equities/bars/daily` | params: code, date |
| 前場四本値 | `/equities/bars/daily/am` | |
| 上場銘柄一覧 | `/equities/master` | |
| 決算発表予定日 | `/equities/earnings-calendar` | ※V1は `/fins/announcement` だった |
| 投資部門別情報 | `/equities/investor-types` | ※V1は `/markets/trades_spec` だった |

### 財務情報
| データ | V2エンドポイント | 備考 |
|---|---|---|
| 財務情報（EPS・売上等） | `/fins/summary` | ※V1は `/fins/statements` だった → **これが403の原因** |
| 財務諸表（BS/PL/CF） | `/fins/details` | |
| 配当金情報 | `/fins/dividend` | |

### 市場データ
| データ | V2エンドポイント | 備考 |
|---|---|---|
| 売買内訳データ | `/markets/breakdown` | |
| 取引カレンダー | `/markets/calendar` | |
| 信用取引週末残高 | `/markets/margin-interest` | ※V1は `/markets/weekly_margin_interest` |
| 日々公表信用取引残高 | `/markets/margin-alert` | |
| 業種別空売り比率 | `/markets/short-ratio` | ※V1は `/markets/short_selling` |
| 空売り残高報告 | `/markets/short-sale-report` | |

### 指数・デリバティブ
| データ | V2エンドポイント | 備考 |
|---|---|---|
| 指数四本値 | `/indices/bars/daily` | params: code |
| TOPIX四本値 | `/indices/bars/daily/topix` | ※V1は `/indices/topix` だった |
| 先物四本値 | `/derivatives/bars/daily/futures` | |
| オプション四本値 | `/derivatives/bars/daily/options` | |
| 日経225オプション | `/derivatives/bars/daily/options/225` | |

---

## レスポンス形式
```json
{
  "data": [ {...}, {...} ],
  "pagination_key": "..."
}
```
- 全エンドポイントが `data` キーの配列を返す
- ページネーションは `pagination_key` を次のリクエストのパラメータに渡す

## 株価カラム名（V2はV1と違う！）
| 項目 | V1 | V2 |
|---|---|---|
| 始値 | Open | O |
| 高値 | High | H |
| 安値 | Low | L |
| 終値 | Close | C |
| 出来高 | Volume | Vo |
| 売買代金 | TurnoverValue | Va |
| 調整後終値 | AdjustmentClose | AdjC |
| 調整後出来高 | AdjustmentVolume | AdjVo |

## Premiumプラン提供範囲
- 全データセット利用可能
- 過去20年分（V2の制限）
- レートリミット: 500リクエスト/分

---

## よくある間違い（絶対に繰り返すな）

❌ `/fins/statements` → ✅ `/fins/summary`
❌ `/fins/announcement` → ✅ `/equities/earnings-calendar`
❌ `/markets/trades_spec` → ✅ `/equities/investor-types`
❌ `/indices/topix` → ✅ `/indices/bars/daily/topix`
❌ V1のトークン認証 → ✅ x-api-key ヘッダー
