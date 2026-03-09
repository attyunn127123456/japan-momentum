#!/usr/bin/env python3
"""
日次シグナルを計算してJSONファイルに出力する。
チャンピオン構成（lb=60, top_n=5, risk_parity, ma100）でスコアリング。

出力: backtest/daily_signal_output.json
形式: { as_of, top5, signals, regime, ... }
"""
import sys, json, time
sys.path.insert(0, '.')
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def build_full_score_df(params, factor_dfs):
    """eval_params のスコア計算部分を再現（バックテスト不要版）"""
    from optimize import build_score_df

    lb = params.get('lookback', 60)
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
            continue
        w = params.get(fac + '_w', 0.0)
        if w > 0 and fac in facs:
            ranked = facs[fac].rank(axis=1, pct=True) * w
            _add(ranked)

    return score_df


def get_latest_prices(prices_dict, codes):
    """各銘柄の最新終値を取得"""
    prices = {}
    for code in codes:
        if code in prices_dict:
            df = prices_dict[code]
            if 'AdjC' in df.columns and not df.empty:
                prices[code] = float(df['AdjC'].iloc[-1])
    return prices


def run():
    t_start = time.time()
    output_path = Path('backtest/daily_signal_output.json')

    # ── チャンピオン構成パラメータ取得 ──
    q_path = Path('backtest/hypothesis_queue.json')
    if not q_path.exists():
        print("ERROR: hypothesis_queue.json が見つかりません")
        return None

    q = json.loads(q_path.read_text())
    baseline = q.get('baseline', {})
    params = baseline.get('params', {})
    if not params:
        print("ERROR: baselineパラメータが空です")
        return None

    lb = params.get('lookback', 60)
    top_n = params.get('top_n', 5)
    regime_filter_type = params.get('market_regime_filter', 'ma100')

    print(f"チャンピオン構成: lb={lb}, top_n={top_n}, filter={regime_filter_type}")

    # ── データ読み込み ──
    from fetch_cache import read_ohlcv
    from universe import get_top_liquid_tickers
    from optimize import precompute, check_market_regime_filter
    from backtest import get_nikkei_history, detect_regime

    end = datetime.now().strftime('%Y-%m-%d')
    # warmup期間: lookback + 余裕（計算に必要な最低限）
    start = (datetime.now() - timedelta(days=max(lb * 3, 365))).strftime('%Y-%m-%d')

    print(f"銘柄リスト取得中...")
    codes = get_top_liquid_tickers(2000)

    prices_dict = {}
    for c in codes:
        df = read_ohlcv(c, start, end)
        if df is not None and not df.empty and 'AdjC' in df.columns:
            prices_dict[c] = df

    if len(prices_dict) < 100:
        print(f"ERROR: データ不足: {len(prices_dict)}銘柄")
        return None

    print(f"{len(prices_dict)}銘柄のデータ取得完了。")

    # ── 日経/TOPIX取得 ──
    nikkei = get_nikkei_history(start, end)
    print(f"市場指数: {len(nikkei)}日分")

    # ── ファクター計算 ──
    print(f"ファクター計算中 (lb={lb})...")
    factor_dfs = precompute(prices_dict, nikkei, [lb])

    # ── スコア計算 ──
    score_df = build_full_score_df(params, factor_dfs)
    if score_df is None or score_df.empty:
        print("ERROR: スコア計算失敗")
        return None

    print(f"スコア計算完了: {score_df.shape}")

    # ── 最新日のスコア ──
    latest = score_df.iloc[-1].dropna().sort_values(ascending=False)
    latest_date = score_df.index[-1]

    # ── 市場レジーム判定 ──
    regime = detect_regime(nikkei, latest_date)
    ma100_cash = check_market_regime_filter(nikkei, latest_date, 'ma100')

    nk_val = float(nikkei.iloc[-1])
    ma100_val = float(nikkei.iloc[-100:].mean()) if len(nikkei) >= 100 else None

    print(f"レジーム: {regime.upper()}, MA100フィルター(cash?): {ma100_cash}")

    # ── 銘柄名マッピング ──
    master_path = Path("data/fundamentals/equities_master.parquet")
    name_map = {}
    if master_path.exists():
        master = pd.read_parquet(master_path)
        name_col = next((c for c in ["CoName", "CompanyName", "Name"] if c in master.columns), None)
        if name_col:
            name_map = dict(zip(master["Code"].astype(str), master[name_col].astype(str)))

    # ── 最新価格取得 ──
    current_prices = get_latest_prices(prices_dict, [str(c) for c in latest.index])

    # ── Top N 推奨銘柄 ──
    top5 = []
    for code in latest.index[:top_n]:
        code_str = str(code)
        top5.append({
            'code': code_str,
            'name': name_map.get(code_str, code_str),
            'score': round(float(latest[code]), 4),
            'price': current_prices.get(code_str, None),
        })

    # ── 前回シグナルとの比較でBUY/SELL/HOLD ──
    prev_path = Path('backtest/daily_signal_prev.json')
    prev_codes = []
    if prev_path.exists():
        try:
            prev_data = json.loads(prev_path.read_text())
            prev_codes = [s['code'] for s in prev_data.get('top5', prev_data.get('recommended', []))]
        except Exception:
            pass

    current_codes = [s['code'] for s in top5]

    signals = []
    # BUY: 新たにtop_nに入った銘柄
    for s in top5:
        if s['code'] not in prev_codes:
            signals.append({**s, 'action': 'BUY'})
        else:
            signals.append({**s, 'action': 'HOLD'})
    # SELL: top_nから外れた銘柄
    for c in prev_codes:
        if c not in current_codes:
            signals.append({
                'code': c,
                'name': name_map.get(c, c),
                'action': 'SELL',
                'score': 0,
                'price': current_prices.get(c, None),
            })

    # ── MA100フィルターが「キャッシュ」なら全てSELL ──
    if ma100_cash:
        print("⚠ MA100フィルター発動: 全ポジションSELLシグナル")
        signals = [{'code': s['code'], 'name': s.get('name', ''), 'action': 'SELL',
                     'score': s.get('score', 0), 'price': s.get('price')} for s in top5]
        for c in prev_codes:
            if c not in current_codes:
                signals.append({
                    'code': c, 'name': name_map.get(c, c),
                    'action': 'SELL', 'score': 0, 'price': current_prices.get(c),
                })

    # ── Top20（参考用） ──
    top20 = []
    for code in latest.index[:20]:
        code_str = str(code)
        top20.append({
            'code': code_str,
            'name': name_map.get(code_str, code_str),
            'score': round(float(latest[code]), 4),
            'price': current_prices.get(code_str, None),
        })

    # ── 出力 ──
    output = {
        'as_of': end,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M JST'),
        'params': {
            'lookback': lb,
            'top_n': top_n,
            'position_sizing': 'risk_parity',
            'market_regime_filter': regime_filter_type,
            'rebalance': 'weekly',
        },
        'regime': regime.upper(),
        'market_filter': {
            'type': regime_filter_type,
            'should_be_cash': ma100_cash,
            'index_value': round(nk_val, 2),
            'ma100': round(ma100_val, 2) if ma100_val else None,
        },
        'top5': top5,
        'signals': signals,
        'top20': top20,
        'total_stocks_scored': len(latest),
    }

    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    # 前日データを更新
    prev_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str))

    elapsed = time.time() - t_start
    print(f"\n✅ 出力完了: {output_path} ({elapsed:.1f}秒)")
    print(f"Top {top_n}: {[s['name'] + '(' + s['code'] + ')' for s in top5]}")
    buy_list = [s for s in signals if s['action'] == 'BUY']
    sell_list = [s for s in signals if s['action'] == 'SELL']
    hold_list = [s for s in signals if s['action'] == 'HOLD']
    print(f"BUY: {len(buy_list)}, SELL: {len(sell_list)}, HOLD: {len(hold_list)}")

    return output


if __name__ == '__main__':
    run()
