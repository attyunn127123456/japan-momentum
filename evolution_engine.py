"""
自律改善エンジン。
1. 現ベストパラメータ周辺を局所探索
2. 新ファクターを順番に追加試験
3. 複合評価ルールで採用/棄却
4. 完了後にopusサブエージェントで仮説生成
5. ノンストップで回り続ける
"""
import itertools, json, time, traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from fetch_cache import read_ohlcv
from universe import get_top_liquid_tickers
from backtest import get_rebalance_dates, get_nikkei_history
from optimize import precompute, eval_params

QUEUE_FILE  = Path('backtest/hypothesis_queue.json')
DONE_FILE   = Path('backtest/hypothesis_done.json')
EVO_LOG     = Path('backtest/evolution_log.json')
START_TRAIN = '2023-01-01'   # 訓練期間
START_VAL   = '2025-01-01'   # 検証期間（汎化性チェック用）
END         = datetime.now().strftime('%Y-%m-%d')
N_CODES     = 2000

# ========== 複合評価ルール ==========
def evaluate(result_train, result_val, baseline):
    """
    total_return最大化スコアで採用/棄却を判定。
    目標: 日経+98%の期間で+300%（年率約60%）を取ること。
    Returns: (adopted: bool, score: float, reasons: list)
    """
    reasons = []

    if not result_train:
        return False, -99, ['訓練期間: 結果なし']

    r = result_train

    # スコア: total_return最大化（max_dd >= 40%なら大幅ペナルティ）
    score = r['total_return_pct'] - max(0, r['max_dd_pct'] - 40) * 5
    reasons.append(f'total_return={r["total_return_pct"]:+.1f}%')
    reasons.append(f'max_dd={r["max_dd_pct"]:.1f}%')

    if r['max_dd_pct'] >= 40:
        reasons.append(f'❌ max_dd >= 40% ペナルティ適用')

    # 取引数チェック
    if r['n_trades'] < 20:
        reasons.append(f'❌ 取引数不足 n={r["n_trades"]}')
        score -= 50
    else:
        reasons.append(f'n_trades={r["n_trades"]}')

    # 採用条件: total_returnがベースラインより+5%以上改善、かつmax_dd < 40%
    delta_return = r['total_return_pct'] - baseline.get('total_pct', 0)
    adopted = delta_return > 5 and r['max_dd_pct'] < 40
    reasons.append(f'delta_return={delta_return:+.1f}%')

    return adopted, round(score, 3), reasons


# ========== データロード ==========
def load_data():
    print(f'データロード ({N_CODES}銘柄)...', flush=True)
    codes = get_top_liquid_tickers(N_CODES)
    # 中型株フィルタ（universe_midcapが有効だったため）
    prices_dict = {}
    warmup = (datetime.strptime(START_TRAIN, '%Y-%m-%d') - timedelta(days=200)).strftime('%Y-%m-%d')
    for c in codes:
        df = read_ohlcv(c, warmup, END)
        if df is None or df.empty or 'AdjC' not in df.columns:
            continue
        avg_price = df['AdjC'].tail(20).mean()
        if 200 <= avg_price <= 10000:  # 中型株フィルタ
            prices_dict[c] = df
    print(f'  {len(prices_dict)}銘柄（中型株フィルタ後）', flush=True)
    nikkei = get_nikkei_history(warmup, END)
    return prices_dict, nikkei, warmup


def load_fundamental_factors():
    """ファンダメンタルデータをファクターDFとして返す"""
    factors = {}
    
    # 財務サマリー → EPS成長・売上成長
    fs_path = Path('data/fundamentals/fins_summary.parquet')
    if fs_path.exists():
        try:
            fs = pd.read_parquet(fs_path)
            if 'Code' in fs.columns:
                fs['DisclosedDate'] = pd.to_datetime(fs.get('DisclosedDate', fs.get('Date','')), errors='coerce')
                for code, grp in fs.groupby('Code'):
                    grp = grp.sort_values('DisclosedDate').dropna(subset=['DisclosedDate'])
                    if len(grp) < 2:
                        continue
                    grp = grp.set_index('DisclosedDate')
                    eps = pd.to_numeric(grp.get('EarningsPerShare', pd.Series(dtype=float)), errors='coerce')
                    rev = pd.to_numeric(grp.get('NetSales', pd.Series(dtype=float)), errors='coerce')
                    eps_growth = eps.pct_change().fillna(0)
                    rev_growth = rev.pct_change().fillna(0)
                    factors[code] = {'eps_growth': eps_growth, 'rev_growth': rev_growth}
        except Exception as e:
            print(f'財務ファクター読み込みエラー: {e}')
    
    # 信用倍率
    mi_path = Path('data/fundamentals/margin_interest.parquet')
    if mi_path.exists():
        try:
            mi = pd.read_parquet(mi_path)
            mi['Date'] = pd.to_datetime(mi.get('Date',''), errors='coerce')
            for code, grp in mi.groupby('Code'):
                grp = grp.sort_values('Date').set_index('Date')
                buy = pd.to_numeric(grp.get('LongMarginTradeVolume', pd.Series()), errors='coerce').fillna(0)
                sell = pd.to_numeric(grp.get('ShortMarginTradeVolume', pd.Series()), errors='coerce').fillna(1).replace(0,1)
                credit = buy / sell
                if code not in factors:
                    factors[code] = {}
                factors[code]['credit_ratio'] = credit
        except Exception as e:
            print(f'信用ファクター読み込みエラー: {e}')
    
    print(f'ファンダメンタルファクター: {len(factors)}銘柄')
    return factors


# ========== 局所探索 (GA版) ==========
def local_search(baseline_params, factor_dfs, prices_dict, nikkei, date_map, fund_factors=None):
    """遺伝的アルゴリズムで局所探索（全網羅の代わりに進化的探索）"""
    import random
    random.seed(42)

    bp = baseline_params
    best_params = dict(bp)
    best_sharpe = bp.get('sharpe', 0)

    # return_dfを1回だけ計算（パフォーマンス改善）
    return_df = pd.DataFrame({c: prices_dict[c]['AdjC'] for c in prices_dict}).pct_change()

    # パラメータ範囲定義（factor_dfsで利用可能なlookbackのみ）
    RANGES = {
        'lookback':     [lb for lb in [40, 60, 80, 100, 120] if lb in factor_dfs],
        'top_n':        [3, 5, 7, 10],  # 集中投資方向に
        'ret_w':        [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
        'rs_w':         [0.0, 0.1, 0.15, 0.2, 0.25, 0.3],
        'green_w':      [0.0, 0.05, 0.1, 0.15, 0.2],
        'smooth_w':     [0.0, 0.05, 0.1, 0.15, 0.2],
        'resilience_w':     [0.0, 0.05, 0.1],
        'high52_w':         [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4],
        'omega_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'short_momentum_w': [0.0, 0.05, 0.1, 0.15, 0.2],
        'cluster_boost_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
    }

    def mutate(params):
        """1〜3個のパラメータをランダムに変異"""
        p = dict(params)
        keys = random.sample(list(RANGES.keys()), random.randint(1, 3))
        for k in keys:
            p[k] = random.choice(RANGES[k])
        # 重みを正規化
        weight_keys = ['ret_w', 'rs_w', 'green_w', 'smooth_w', 'resilience_w', 'high52_w', 'omega_w', 'short_momentum_w']
        total = sum(p.get(k, 0) for k in weight_keys)
        if total > 0:
            for k in weight_keys:
                p[k] = round(p.get(k, 0) / total, 4)
        p['rebalance'] = bp.get('rebalance', 'weekly')
        return p

    def eval_params_local(p):
        try:
            return eval_params(p, factor_dfs, prices_dict, date_map[p['rebalance']],
                               nikkei, START_TRAIN, return_df)
        except Exception:
            return None

    # 初期母集団（ベスト + ランダム変異19個）
    population = [dict(bp)]
    for _ in range(19):
        population.append(mutate(bp))

    results = []
    POP_SIZE = 20
    N_GENERATIONS = 10

    print(f'局所探索(GA): {POP_SIZE}個×{N_GENERATIONS}世代 = 最大{POP_SIZE*N_GENERATIONS}パターン', flush=True)

    for gen in range(N_GENERATIONS):
        gen_results = []
        for p in population:
            r = eval_params_local(p)
            if r:
                gen_results.append((r['sharpe'], p, r))

        gen_results.sort(key=lambda x: -x[0])
        results.extend(gen_results)

        if gen_results:
            best_gen_sharpe = gen_results[0][0]
            print(f'  世代{gen+1}: best_sharpe={best_gen_sharpe:.3f}', flush=True)
            if best_gen_sharpe > best_sharpe:
                best_sharpe = best_gen_sharpe
                best_params = gen_results[0][1]

        # 上位5を残して15個を変異
        survivors = [p for _, p, _ in gen_results[:5]]
        if not survivors:
            break
        population = list(survivors)
        for _ in range(POP_SIZE - len(survivors)):
            parent = random.choice(survivors)
            population.append(mutate(parent))

    results.sort(key=lambda x: -x[0])
    # 既存の消費コードに合わせて (params, result) 形式で返す
    return [(p, r) for _, p, r in results]


# ========== 新ファクター試験 ==========
NEW_FACTORS = [
    'eps_growth',       # EPS成長率
    'rev_growth',       # 売上成長率
    'credit_ratio',     # 信用倍率（買い残/売り残）
    'rsi14',            # RSI(14)
    'ma_deviation',     # 移動平均乖離率
    'momentum_12_1',    # 12ヶ月-1ヶ月モメンタム（古典的ファクター）
]

def add_factor_to_precomputed(factor_name, factor_dfs, prices_dict, nikkei, fund_factors):
    """新ファクターをfactor_dfsに追加"""
    nk_rets = nikkei.pct_change()
    
    for code, df in prices_dict.items():
        p = df['AdjC'].dropna()
        dr = p.pct_change()
        
        for lb in [40, 60, 80]:
            key = (code, lb)
            if key not in factor_dfs:
                continue
            
            if factor_name == 'rsi14':
                delta = dr.copy()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = (-delta.clip(upper=0)).rolling(14).mean()
                rs = gain / loss.replace(0, 1e-9)
                rsi = 100 - 100 / (1 + rs)
                factor_dfs[key]['rsi14'] = (rsi / 100).astype(float)  # 0-1に正規化
            
            elif factor_name == 'ma_deviation':
                ma50 = p.rolling(50).mean()
                dev = (p / ma50 - 1).clip(-0.3, 0.3)
                factor_dfs[key]['ma_deviation'] = dev.astype(float)
            
            elif factor_name == 'momentum_12_1':
                # 12ヶ月前から1ヶ月前のリターン（直近1ヶ月除外）
                ret_12 = p / p.shift(252) - 1
                ret_1  = p / p.shift(21) - 1
                mom = ret_12 - ret_1
                factor_dfs[key]['momentum_12_1'] = mom.astype(float)
            
            elif factor_name in ('eps_growth', 'rev_growth', 'credit_ratio'):
                if code in fund_factors and factor_name in fund_factors[code]:
                    series = fund_factors[code][factor_name]
                    # リインデックスして日次に合わせる（前値埋め）
                    reindexed = series.reindex(p.index, method='ffill').fillna(0)
                    factor_dfs[key][factor_name] = reindexed.astype(float)
    
    return factor_dfs


# ========== メイン改善ループ ==========
def run_evolution():
    queue = json.loads(QUEUE_FILE.read_text())
    baseline = queue['baseline']
    
    print(f'\n=== Evolution Engine 起動 ===', flush=True)
    print(f'ベースライン: sharpe={baseline["sharpe"]}, total={baseline.get("total_pct")}%', flush=True)
    
    prices_dict, nikkei, warmup = load_data()
    fund_factors = load_fundamental_factors()
    
    print('ファクター事前計算...', flush=True)
    factor_dfs = precompute(prices_dict, nikkei, [40, 60, 80, 100])
    
    return_df = pd.DataFrame({c: prices_dict[c]['AdjC'] for c in prices_dict}).pct_change()
    daily_d  = get_rebalance_dates(warmup, END, 'daily')
    weekly_d = get_rebalance_dates(warmup, END, 'weekly')
    date_map = {'daily': daily_d, 'weekly': weekly_d}
    
    # ---- Step1: queueの手動仮説を先に処理 ----
    for hypo in queue['queue']:
        if hypo['status'] == 'pending':
            print(f'\n--- 手動仮説: {hypo["id"]} ---', flush=True)
            # run_hypothesis.pyに任せる（既存フロー）
            import subprocess, sys
            subprocess.Popen([sys.executable, 'run_hypothesis.py'],
                           cwd=str(Path(__file__).parent),
                           stdout=open('logs/hypothesis.log','a'),
                           stderr=subprocess.STDOUT)
            return  # run_hypothesis.pyがauto_nextで連鎖する
    
    # ---- Step2: 局所探索 ----
    print('\n--- 局所探索 開始 ---', flush=True)
    bp = baseline.get('params', {
        'lookback':60,'top_n':10,'rebalance':'weekly',
        'ret_w':0.3,'rs_w':0.3,'green_w':0.2,'smooth_w':0.2,'resilience_w':0.0
    })
    local_results = local_search(bp, factor_dfs, prices_dict, nikkei, date_map)
    
    # 複合評価
    best_local = None
    best_score = -99
    for p, r_train in local_results:
        r_val = eval_params(p, factor_dfs, prices_dict, date_map[p['rebalance']], nikkei, START_VAL, return_df)
        adopted, score, reasons = evaluate(r_train, r_val, baseline)
        print(f'  sharpe={r_train["sharpe"]:.3f} score={score:+.3f} {"✅" if adopted else "❌"} lb={p["lookback"]} tn={p["top_n"]}', flush=True)
        if score > best_score:
            best_score = score
            best_local = (p, r_train, r_val, adopted, reasons)
    
    if best_local:
        p, r_train, r_val, adopted, reasons = best_local
        append_log('local_search', f'局所探索ベスト lb={p["lookback"]} tn={p["top_n"]}',
                   r_train, adopted, r_train['sharpe'] - baseline['sharpe'])
        if adopted:
            print(f'\n✅ 局所探索で改善: sharpe {baseline["sharpe"]} → {r_train["sharpe"]}', flush=True)
            queue['baseline'] = {
                'sharpe': r_train['sharpe'],
                'total_pct': r_train['total_return_pct'],
                'alpha_pct': r_train['alpha_pct'],
                'max_dd_pct': r_train['max_dd_pct'],
                'params': p,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'hypothesis': 'local_search',
            }
            QUEUE_FILE.write_text(json.dumps(queue, ensure_ascii=False, indent=2))
            baseline = queue['baseline']

        # GA local_search 完了後にファクター整理を実行（best result のパラメータで）
        print('\n--- ファクター整理 (cleanup_weak_signals) ---', flush=True)
        cleanup_weak_signals(p)
    
    # ---- Step3: 新ファクター試験 ----
    print('\n--- 新ファクター試験 ---', flush=True)
    tested_factors = set(queue.get('tested_factors', []))
    
    for factor_name in NEW_FACTORS:
        if factor_name in tested_factors:
            continue
        
        print(f'  ファクター追加: {factor_name}', flush=True)
        # ファンダメンタル系はデータが揃ってから
        if factor_name in ('eps_growth', 'rev_growth', 'credit_ratio'):
            if not Path('data/fundamentals/fins_summary.parquet').exists():
                print(f'  → データ未取得、スキップ')
                continue
        
        try:
            fds_copy = {k: v.copy() for k, v in factor_dfs.items()}
            add_factor_to_precomputed(factor_name, fds_copy, prices_dict, nikkei, fund_factors)
        except Exception as e:
            print(f'  ファクター追加エラー: {e}')
            continue
        
        # テスト: ベストパラメータに新ファクターを追加
        test_p = {**bp, factor_name+'_w': 0.1}
        # 既存重みを少し下げる
        test_p['ret_w'] = max(0.1, bp.get('ret_w', 0.3) - 0.05)
        
        r_train = eval_params(test_p, fds_copy, prices_dict, date_map[test_p['rebalance']], nikkei, START_TRAIN, return_df)
        r_val   = eval_params(test_p, fds_copy, prices_dict, date_map[test_p['rebalance']], nikkei, START_VAL, return_df)
        adopted, score, reasons = evaluate(r_train, r_val, baseline)
        
        print(f'  {factor_name}: sharpe={r_train["sharpe"] if r_train else "N/A":.3f} {"✅採用" if adopted else "❌不採用"}', flush=True)
        
        tested_factors.add(factor_name)
        queue['tested_factors'] = list(tested_factors)
        append_log(f'factor_{factor_name}', f'新ファクター: {factor_name}',
                   r_train, adopted, (r_train['sharpe'] - baseline['sharpe']) if r_train else None)
        
        if adopted and r_train:
            # ファクターを本採用してfactor_dfsを更新
            factor_dfs = fds_copy
            bp = test_p
            baseline['sharpe'] = r_train['sharpe']
            baseline['params'] = test_p
            queue['baseline'] = {**baseline, 'date': datetime.now().strftime('%Y-%m-%d'), 'hypothesis': f'factor_{factor_name}'}
            print(f'  → ベースライン更新: sharpe={r_train["sharpe"]}', flush=True)
        
        QUEUE_FILE.write_text(json.dumps(queue, ensure_ascii=False, indent=2))
        time.sleep(1)
    
    # ---- Step4: 組み合わせテスト ----
    from combination_search import run_combination_search
    run_combination_search()

    # ---- Step4b: シグナル積み上げ最適化 ----
    from signal_search import run_signal_search
    run_signal_search()

    # ---- 完了シグナル ----
    DONE_FILE.write_text(json.dumps({
        'status': 'done',
        'id': 'evolution_cycle',
        'win': True,
        'delta_sharpe': baseline['sharpe'] - json.loads(QUEUE_FILE.read_text())['baseline'].get('sharpe', 0),
        'result': baseline,
        'at': datetime.now().isoformat(),
    }, ensure_ascii=False, indent=2))
    
    print('\n=== Evolution cycle 完了 ===', flush=True)
    
    # 即opusをキック
    import subprocess
    subprocess.run(
        ['openclaw', 'system', 'event',
         '--mode', 'now',
         '--text', 'evolution_done: local search and factor tests completed. Please run opus analysis and generate 3 new hypotheses now.'],
        capture_output=True
    )
    print('openclaw system event 送信完了', flush=True)


def cleanup_weak_signals(baseline_params):
    """weight が 0.02 以下のファクターを rejected に整理する。
    signal_library.json が存在しない場合はスキップ（エラーにしない）。"""
    lib_path = Path('backtest/signal_library.json')
    if not lib_path.exists():
        print('  signal_library.json が存在しないため cleanup_weak_signals をスキップ', flush=True)
        return

    try:
        lib = json.loads(lib_path.read_text())
        changed = 0
        for sig in lib['signals']:
            # param_key が直接設定されていればそれを使い、なければ id から推測
            param_key = sig.get('param_key') or (sig['id'].replace('_factor', '') + '_w')
            w = baseline_params.get(param_key, 0)
            if w < 0.02 and sig.get('status') == 'active':
                sig['status'] = 'rejected'
                changed += 1
                print(f'  🗑️  {sig["id"]}: weight={w:.3f} → rejected', flush=True)
        if changed:
            lib_path.write_text(json.dumps(lib, indent=2, ensure_ascii=False))
            print(f'  cleanup_weak_signals: {changed} シグナルを rejected に変更', flush=True)
        else:
            print('  cleanup_weak_signals: 変更なし', flush=True)
    except Exception as e:
        print(f'  cleanup_weak_signals エラー（スキップ）: {e}', flush=True)


def append_log(hid, desc, result, win, delta):
    log = json.loads(EVO_LOG.read_text()) if EVO_LOG.exists() else {'best10':[], 'all':[], 'total':0}
    entry = {
        'at': datetime.now().isoformat(), 'id': hid, 'desc': desc, 'win': win,
        'delta_sharpe': delta,
        'sharpe': result.get('sharpe') if result else None,
        'total_return_pct': result.get('total_return_pct') if result else None,
        'alpha_pct': result.get('alpha_pct') if result else None,
        'max_dd_pct': result.get('max_dd_pct') if result else None,
        'params': {k: result[k] for k in result if '_w' in k or k in ['lookback','top_n','rebalance']} if result else {},
    }
    all_entries = log.get('all', []) + [entry]
    valid = sorted([x for x in all_entries if x.get('sharpe') is not None],
                   key=lambda x: x['sharpe'], reverse=True)
    EVO_LOG.write_text(json.dumps({'best10': valid[:10], 'all': valid[:300], 'total': len(all_entries)},
                                   ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        run_evolution()
    except Exception as e:
        print(f'エラー: {e}')
        traceback.print_exc()
        DONE_FILE.write_text(json.dumps({'status':'error','id':'evolution','error':str(e)}, ensure_ascii=False))
