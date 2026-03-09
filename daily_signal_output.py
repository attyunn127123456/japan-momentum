#!/usr/bin/env python3
"""
日次シグナルを計算してJSONファイルに出力する。
OpenClawのcron/heartbeatが呼び出し、結果を読んでDiscord通知する。

build_score_nan_safe は存在しないため、optimize.build_score_df と
eval_params のスコア計算ロジックを再現する。
"""
import sys, json
sys.path.insert(0, '.')
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def build_full_score_df(params, factor_dfs):
    """eval_params のスコア計算部分を再現（バックテスト不要版）"""
    from optimize import build_score_df

    lb = params.get('lookback', 40)
    if lb not in factor_dfs:
        return None

    weights = {k: params.get(k + "_w", 0.0) for k in ["ret", "rs", "green", "smooth", "resilience"]}
    score_df = build_score_df(factor_dfs, lb, weights)

    def _add(ranked_df):
        nonlocal score_df
        if ranked_df is None:
            return
        if score_df is None:
            score_df = ranked_df
        else:
            score_df = score_df.add(ranked_df, fill_value=0)

    facs = factor_dfs[lb]

    extra_factors = [
        'short_momentum', 'high52', 'omega', 'close_location', 'range_expand',
        'win_streak', 'sector_momentum', 'overnight_return', 'volume_acceleration',
        'higher_lows', 'body_strength', 'vol_return_corr', 'accumulation',
        'momentum_consistency', 'upside_capture', 'gap_momentum', 'volume_confirm',
        'ret_skip', 'cluster_boost', 'return_autocorr', 'volume_slope', 'clean_momentum',
    ]

    # short_momentum は ret5/ret10 の平均
    short_momentum_w = params.get('short_momentum_w', 0.0)
    if short_momentum_w > 0 and 'ret5' in facs and 'ret10' in facs:
        sm_df = (facs['ret5'] + facs['ret10']) / 2
        sm_ranked = sm_df.rank(axis=1, pct=True) * short_momentum_w
        _add(sm_ranked)

    for fac in extra_factors:
        if fac == 'short_momentum':
            continue  # already handled above
        w = params.get(fac + '_w', 0.0)
        if w > 0 and fac in facs:
            ranked = facs[fac].rank(axis=1, pct=True) * w
            _add(ranked)

    return score_df


def run():
    output_path = Path('backtest/daily_signal_output.json')

    # ベストパラメータ取得
    q_path = Path('backtest/hypothesis_queue.json')
    if not q_path.exists():
        print("hypothesis_queue.json が見つかりません")
        return None

    q = json.loads(q_path.read_text())
    baseline = q.get('baseline', {})
    params = baseline.get('params', {})
    if not params:
        print("baselineパラメータが空です")
        return None

    # 直近データで現在のスコアを計算
    from fetch_cache import read_ohlcv
    from universe import get_top_liquid_tickers
    from optimize import precompute
    from backtest import get_nikkei_history

    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    warmup = (datetime.now() - timedelta(days=600)).strftime('%Y-%m-%d')

    print(f"銘柄リスト取得中...")
    codes = get_top_liquid_tickers(2000)
    prices_dict = {}
    for c in codes:
        df = read_ohlcv(c, warmup, end)
        if df is not None and not df.empty and 'AdjC' in df.columns:
            prices_dict[c] = df

    if len(prices_dict) < 100:
        print(f"データ不足: {len(prices_dict)}銘柄")
        return None

    print(f"{len(prices_dict)}銘柄のデータ取得完了。ファクター計算中...")
    nikkei = get_nikkei_history(start, end)
    lb = params.get('lookback', 40)
    factor_dfs = precompute(prices_dict, nikkei, [lb])

    # スコア計算（全日付 x 全銘柄）
    score_df = build_full_score_df(params, factor_dfs)
    if score_df is None or score_df.empty:
        print("スコア計算失敗")
        return None

    # 最新日のスコアを取得
    latest = score_df.iloc[-1].dropna().sort_values(ascending=False)

    # 銘柄名マッピング
    master_path = Path("data/fundamentals/equities_master.parquet")
    name_map = {}
    if master_path.exists():
        master = pd.read_parquet(master_path)
        name_col = next((c for c in ["CoName", "CompanyName", "Name"] if c in master.columns), None)
        if name_col:
            name_map = dict(zip(master["Code"].astype(str), master[name_col].astype(str)))

    top_n = params.get('top_n', 2)

    # 現在の推奨銘柄（top_n）
    recommended = []
    for code in latest.index[:top_n]:
        recommended.append({
            'code': str(code),
            'name': name_map.get(str(code), str(code)),
            'score': round(float(latest[code]), 4),
        })

    # 前日の推奨と比較してBUY/SELL/HOLD
    prev_path = Path('backtest/daily_signal_prev.json')
    prev_codes = []
    if prev_path.exists():
        prev_data = json.loads(prev_path.read_text())
        prev_codes = [s['code'] for s in prev_data.get('recommended', [])]

    current_codes = [s['code'] for s in recommended]
    buy = [s for s in recommended if s['code'] not in prev_codes]
    sell = [{'code': c, 'name': name_map.get(c, c)} for c in prev_codes if c not in current_codes]
    hold = [s for s in recommended if s['code'] in prev_codes]

    # 全銘柄スコアを保存（検索用）
    all_scores = []
    for code in latest.index:  # スコアがあるもの全て
        all_scores.append({
            'code': str(code),
            'name': name_map.get(str(code), str(code)),
            'score': round(float(latest[code]), 4),
        })

    # top20は引き続き先頭20件
    top20 = all_scores[:20]

    # 市場レジームフィルター状態を計算
    from optimize import check_market_regime_filter
    regime_filter_type = params.get('market_regime_filter', 'none')
    regime_filter_status = {}
    try:
        latest_date = nikkei.index[-1]
        nk_val = float(nikkei.iloc[-1])
        # 各フィルターの状態を個別に確認
        ma100_active = check_market_regime_filter(nikkei, latest_date, 'ma100')
        dd15_active = check_market_regime_filter(nikkei, latest_date, 'dd15')
        combined_active = check_market_regime_filter(nikkei, latest_date, 'combined')
        # MA100の値も計算
        ma100_val = float(nikkei.iloc[-100:].mean()) if len(nikkei) >= 100 else None
        # 250日高値
        high250_val = float(nikkei.iloc[-min(250, len(nikkei)):].max())
        dd_from_high = round((nk_val - high250_val) / high250_val * 100, 2) if high250_val > 0 else None
        regime_filter_status = {
            'configured_filter': regime_filter_type,
            'nikkei_current': round(nk_val, 2),
            'ma100': round(ma100_val, 2) if ma100_val else None,
            'ma100_active': ma100_active,
            'high250': round(high250_val, 2),
            'dd_from_high_pct': dd_from_high,
            'dd15_active': dd15_active,
            'combined_active': combined_active,
            'should_be_cash': check_market_regime_filter(nikkei, latest_date, regime_filter_type),
        }
    except Exception as e:
        regime_filter_status = {'error': str(e)}

    output = {
        'as_of': end,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M JST'),
        'params': {'lookback': lb, 'top_n': top_n},
        'market_regime_filter': regime_filter_status,
        'recommended': recommended,
        'changes': {'buy': buy, 'sell': sell, 'hold': hold},
        'top20': top20,
        'all_scores': all_scores,  # 全銘柄スコア追加
        'total_return_pct': baseline.get('total_pct', 0),
    }

    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    # 前日データを更新
    prev_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str))

    print(f"出力完了: {output_path}")
    print(f"推奨: {[s['name'] for s in recommended]}")
    print(f"BUY: {[s['name'] for s in buy]}, SELL: {[s['name'] for s in sell]}, HOLD: {[s['name'] for s in hold]}")

    # generate_dashboard_cache.py を呼んで全キャッシュを最新化
    import subprocess
    print("ダッシュボードキャッシュ更新中...", flush=True)
    result = subprocess.run(
        ['python3', 'generate_dashboard_cache.py'],
        capture_output=True, text=True, cwd=str(Path('.').resolve())
    )
    if result.returncode == 0:
        print("ダッシュボードキャッシュ更新完了", flush=True)
    else:
        print(f"キャッシュ更新エラー: {result.stderr[-200:]}", flush=True)

    return output


if __name__ == '__main__':
    run()
