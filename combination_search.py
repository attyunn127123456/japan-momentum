"""
有望仮説の組み合わせテスト。
evolution_log.jsonから有望仮説を抽出し、2〜3つの組み合わせをバックテストする。
"""
import json, time, itertools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from fetch_cache import read_ohlcv
from universe import get_top_liquid_tickers
from backtest import get_rebalance_dates, get_nikkei_history
from optimize import precompute, eval_params

EVO_LOG      = Path('backtest/evolution_log.json')
COMBO_LOG    = Path('backtest/combination_log.json')
QUEUE_FILE   = Path('backtest/hypothesis_queue.json')
START_TRAIN  = '2023-01-01'
START_VAL    = '2025-01-01'
END          = datetime.now().strftime('%Y-%m-%d')
N_CODES      = 200

# パラメータのデフォルト値（合成時に不足分を補完）
PARAM_DEFAULTS = {
    'lookback':     60,
    'top_n':        10,
    'rebalance':    'weekly',
    'ret_w':        0.3,
    'rs_w':         0.3,
    'green_w':      0.2,
    'smooth_w':     0.2,
    'resilience_w': 0.0,
}

NUMERIC_PARAM_BOUNDS = {
    'lookback':     (40, 100),
    'top_n':        (3, 30),
    'ret_w':        (0.0, 1.0),
    'rs_w':         (0.0, 0.5),
    'green_w':      (0.0, 0.4),
    'smooth_w':     (0.0, 0.4),
    'resilience_w': (0.0, 0.3),
}


def load_data():
    print(f'データロード ({N_CODES}銘柄)...', flush=True)
    codes = get_top_liquid_tickers(N_CODES)
    prices_dict = {}
    warmup = (datetime.strptime(START_TRAIN, '%Y-%m-%d') - timedelta(days=200)).strftime('%Y-%m-%d')
    for c in codes:
        df = read_ohlcv(c, warmup, END)
        if df is None or df.empty or 'AdjC' not in df.columns:
            continue
        avg_price = df['AdjC'].tail(20).mean()
        if 200 <= avg_price <= 10000:
            prices_dict[c] = df
    print(f'  {len(prices_dict)}銘柄（中型株フィルタ後）', flush=True)
    nikkei = get_nikkei_history(warmup, END)
    return prices_dict, nikkei, warmup


def extract_candidates(evo_log):
    """evolution_logから有望仮説を候補として抽出"""
    candidates = []
    all_entries = evo_log.get('all', []) + evo_log.get('best10', [])
    seen_ids = set()
    for entry in all_entries:
        eid = entry.get('id', '')
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        delta = entry.get('delta_sharpe')
        win   = entry.get('win', False)
        params = entry.get('params', {})
        if not params:
            continue
        if (delta is not None and delta > -0.1) or win:
            candidates.append({'id': eid, 'params': params, 'sharpe': entry.get('sharpe'), 'delta_sharpe': delta})
    print(f'候補仮説: {len(candidates)}件', flush=True)
    return candidates


def merge_params(p1, p2):
    """2つのパラメータセットを合成する（平均値で合成）"""
    merged = dict(PARAM_DEFAULTS)
    merged.update(p1)

    for key, v2 in p2.items():
        v1 = merged.get(key)
        if key == 'rebalance':
            merged[key] = v1 if v1 else v2
        elif isinstance(v2, (int, float)) and isinstance(v1, (int, float)):
            avg = (v1 + v2) / 2
            if key in NUMERIC_PARAM_BOUNDS:
                lo, hi = NUMERIC_PARAM_BOUNDS[key]
                avg = max(lo, min(hi, avg))
            if key == 'lookback':
                avg = int(round(avg / 10) * 10)
            elif key == 'top_n':
                avg = int(round(avg))
            merged[key] = avg
        else:
            merged[key] = v2

    return merged


def run_combination_search():
    """有望仮説の組み合わせをテストしてcombo_logに保存"""
    print('\n--- 組み合わせテスト 開始 ---', flush=True)

    if not EVO_LOG.exists():
        print('evolution_log.json が存在しません。スキップ。', flush=True)
        return
    evo_log = json.loads(EVO_LOG.read_text())

    candidates = extract_candidates(evo_log)
    if len(candidates) < 2:
        print('候補が2件未満のためスキップ。', flush=True)
        return

    prices_dict, nikkei, warmup = load_data()
    factor_dfs = precompute(prices_dict, nikkei, [40, 60, 80, 100])
    return_df  = pd.DataFrame({c: prices_dict[c]['AdjC'] for c in prices_dict}).pct_change()
    daily_d    = get_rebalance_dates(warmup, END, 'daily')
    weekly_d   = get_rebalance_dates(warmup, END, 'weekly')
    date_map   = {'daily': daily_d, 'weekly': weekly_d}

    queue    = json.loads(QUEUE_FILE.read_text())
    baseline = queue['baseline']

    top_candidates = sorted(candidates, key=lambda x: x.get('sharpe') or 0, reverse=True)[:10]
    combos = list(itertools.combinations(top_candidates, 2))
    print(f'組み合わせ数: {len(combos)}', flush=True)

    combo_results = []

    for cA, cB in combos:
        merged = merge_params(cA['params'], cB['params'])
        combo_id = f"combo_{cA['id']}+{cB['id']}"

        try:
            r_train = eval_params(
                merged, factor_dfs, prices_dict,
                date_map[merged['rebalance']], nikkei, START_TRAIN, return_df
            )
            r_val = eval_params(
                merged, factor_dfs, prices_dict,
                date_map[merged['rebalance']], nikkei, START_VAL, return_df
            )
        except Exception as e:
            print(f'  {combo_id}: エラー {e}', flush=True)
            continue

        if not r_train:
            continue

        delta_sharpe = r_train['sharpe'] - baseline['sharpe']

        from evolution_engine import evaluate
        adopted, score, reasons = evaluate(r_train, r_val, baseline)

        print(
            f'  {combo_id}: sharpe={r_train["sharpe"]:.3f} delta={delta_sharpe:+.3f} '
            f'score={score:+.3f} {"採用" if adopted else "不採用"}',
            flush=True
        )

        entry = {
            'at':               datetime.now().isoformat(),
            'id':               combo_id,
            'hypotheses':       [cA['id'], cB['id']],
            'merged_params':    merged,
            'sharpe':           r_train['sharpe'],
            'total_return_pct': r_train.get('total_return_pct'),
            'alpha_pct':        r_train.get('alpha_pct'),
            'max_dd_pct':       r_train.get('max_dd_pct'),
            'delta_sharpe':     delta_sharpe,
            'score':            score,
            'adopted':          adopted,
            'reasons':          reasons,
        }
        combo_results.append(entry)

        if adopted:
            print(f'\n組み合わせで改善: sharpe {baseline["sharpe"]} -> {r_train["sharpe"]}', flush=True)
            queue['baseline'] = {
                'sharpe':     r_train['sharpe'],
                'total_pct':  r_train['total_return_pct'],
                'alpha_pct':  r_train['alpha_pct'],
                'max_dd_pct': r_train['max_dd_pct'],
                'params':     merged,
                'date':       datetime.now().strftime('%Y-%m-%d'),
                'hypothesis': combo_id,
            }
            QUEUE_FILE.write_text(json.dumps(queue, ensure_ascii=False, indent=2))
            baseline = queue['baseline']

    existing = json.loads(COMBO_LOG.read_text()) if COMBO_LOG.exists() else []
    existing.extend(combo_results)
    COMBO_LOG.write_text(json.dumps(existing[-300:], ensure_ascii=False, indent=2))
    print(f'組み合わせテスト完了: {len(combo_results)}件 / combo_log保存', flush=True)


if __name__ == '__main__':
    run_combination_search()
