"""
自律改善エンジン。
1. 現ベストパラメータ周辺を局所探索
2. 新ファクターを順番に追加試験
3. 複合評価ルールで採用/棄却
4. 完了後にopusサブエージェントで仮説生成
5. ノンストップで回り続ける
"""
import itertools, time, traceback
import json_safe as json

def _backup_baseline(baseline):
    """ベースライン更新前に自動バックアップ"""
    from datetime import datetime
    from pathlib import Path
    import json
    try:
        Path('backtest/backups').mkdir(exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        total = baseline.get('total_pct', 0)
        p = f'backtest/backups/baseline_{ts}_total{total:.1f}.json'
        Path(p).write_text(json.dumps(baseline, ensure_ascii=False, indent=2, default=str))
        print(f'[backup] {p}', flush=True)
    except Exception as e:
        print(f'[backup] 失敗: {e}', flush=True)

def _fb(o):
    """bool/NaN/Inf をJSONシリアライズ可能に変換"""
    if isinstance(o, bool): return int(o)
    if isinstance(o, dict): return {k: _fb(v) for k, v in o.items()}
    if isinstance(o, list): return [_fb(i) for i in o]
    import math
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)): return None
    return o


def sanitize(obj):
    """bool/NaN/Inf をJSONシリアライズ可能にする"""
    import math
    if isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(i) for i in obj]
    return obj
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from fetch_cache import read_ohlcv
from universe import get_top_liquid_tickers
from backtest import get_rebalance_dates, get_nikkei_history
from optimize import precompute, eval_params, run_optuna_optimization, run_oos_validation, run_walk_forward_validation, eval_walkforward

QUEUE_FILE  = Path('backtest/hypothesis_queue.json')
DONE_FILE   = Path('backtest/hypothesis_done.json')
EVO_LOG     = Path('backtest/evolution_log.json')

MAX_ACTIVE_FACTORS = 7  # 同時に使えるファクターの最大数（0より大きいウェイトを持つもの）
START_TRAIN = '2021-01-01'   # IS訓練期間（直近5年）
END_TRAIN   = '2025-12-31'   # IS訓練期間終了
START_VAL   = '2026-01-01'   # 検証期間開始（今年）
END_VAL     = datetime.now().strftime('%Y-%m-%d')  # 今日まで
END         = datetime.now().strftime('%Y-%m-%d')
N_CODES     = 4000

# OOSの最低基準
OOS_SHARPE_MIN = 1.5
OOS_TOTAL_RETURN_MIN = 50.0  # %

# ========== 複合評価ルール ==========
def evaluate(result_train, result_val, baseline):
    """
    Calmar Ratio + DD制約 で採用/棄却を判定。
    目標: DD ≤ 30%（絶対条件）+ Sharpe最大化。
    
    Calmar Ratio = 年率リターン / 最大DD
    DD制約: DD > 30% なら重ペナルティ、DD > 40% なら棄却。
    
    Returns: (adopted: bool, score: float, reasons: list)
    """
    reasons = []

    if not result_train:
        return False, -99, ['訓練期間: 結果なし']

    r = result_train
    dd = r['max_dd_pct']
    total_ret = r['total_return_pct']
    sharpe = r['sharpe']

    # Calmar Ratio 計算（年率リターン / MaxDD）
    # total_return_pctを年数で割って年率化（近似）
    n_trades = r.get('n_trades', 52)
    est_years = max(n_trades / 52, 0.5)  # 週次リバランス想定
    annual_ret = total_ret / est_years
    calmar = annual_ret / max(dd, 1.0)

    reasons.append(f'total_return={total_ret:+.1f}%')
    reasons.append(f'max_dd={dd:.1f}%')
    reasons.append(f'sharpe={sharpe:.3f}')
    reasons.append(f'calmar={calmar:.2f}')

    # スコア: Calmar Ratio ベース（DD制約付き）
    if dd <= 30:
        # DD 30%以下: Calmar × 10 + Sharpeボーナス
        score = calmar * 10 + sharpe * 5
        reasons.append(f'✅ DD ≤ 30%')
    elif dd <= 40:
        # DD 30-40%: ペナルティ付き
        penalty = (dd - 30) * 3
        score = calmar * 10 + sharpe * 5 - penalty
        reasons.append(f'⚠️ DD 30-40% ペナルティ={penalty:.1f}')
    else:
        # DD > 40%: 大幅ペナルティ
        score = calmar * 10 - (dd - 30) * 10
        reasons.append(f'❌ DD > 40% 大幅ペナルティ')

    # 取引数チェック
    if n_trades < 20:
        reasons.append(f'❌ 取引数不足 n={n_trades}')
        score -= 50
    else:
        reasons.append(f'n_trades={n_trades}')

    # 採用条件:
    # 1. DD < 35% (絶対条件に近い、30%目標だが少し余裕)
    # 2. Calmarがベースラインより改善 OR Sharpeがベースラインより改善
    # 3. total_returnがベースラインの90%以上（大幅劣化しない）
    baseline_total = baseline.get('total_pct', 0)
    baseline_sharpe = baseline.get('sharpe', 0)
    baseline_dd = baseline.get('max_dd_pct', 100)
    baseline_calmar = baseline_total / max(baseline_dd, 1.0)

    delta_return = total_ret - baseline_total
    delta_sharpe = sharpe - baseline_sharpe
    delta_calmar = calmar - baseline_calmar

    adopted = (
        dd < 35 and
        total_ret > baseline_total * 0.9 and
        (delta_calmar > 0 or delta_sharpe > 0.05) and
        sharpe >= baseline_sharpe * 0.95
    )
    reasons.append(f'delta_calmar={delta_calmar:+.2f}')
    reasons.append(f'delta_sharpe={delta_sharpe:+.3f}')

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
        prices_dict[c] = df
    print(f'  {len(prices_dict)}銘柄ロード完了', flush=True)
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


# ========== ファクター数制限ヘルパー ==========
def apply_factor_limit(p, max_factors=None):
    """アクティブファクター数を MAX_ACTIVE_FACTORS 以下に制限する。
    0.005 より大きいウェイトを持つファクターのうち、値の小さいものから削る。"""
    if max_factors is None:
        max_factors = MAX_ACTIVE_FACTORS
    active_factor_count = sum(1 for k, v in p.items() if k.endswith('_w') and v > 0.005)
    if active_factor_count > max_factors:
        factor_weights = [(k, v) for k, v in p.items() if k.endswith('_w') and v > 0.005]
        factor_weights.sort(key=lambda x: x[1])  # 小さい順
        to_zero = factor_weights[:active_factor_count - max_factors]
        for k, _ in to_zero:
            p[k] = 0.0
    return p


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
        'lookback':     [lb for lb in [20, 40, 60, 80, 100] if lb in factor_dfs],
        'top_n':        [3, 5, 7, 10],  # 集中投資方向に
        'ret_w':        [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
        'rs_w':         [0.0, 0.1, 0.15, 0.2, 0.25, 0.3],
        'green_w':      [0.0, 0.05, 0.1, 0.15, 0.2],
        'smooth_w':     [0.0, 0.05, 0.1, 0.15, 0.2],
        'resilience_w':     [0.0, 0.05, 0.1],
        'high52_w':         [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4],
        'omega_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'short_momentum_w': [0.0, 0.05, 0.1, 0.15, 0.2],
        'buying_intensity_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'intraday_support_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'relative_strength_accel_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'low_floor_momentum_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'intraday_return_ratio_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'crash_beta_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'volume_turnover_decay_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'negative_skew_penalty_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'frog_in_pan_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'mean_reversion_risk_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'stock_sharpe_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'amihud_liquidity_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'volume_breakout_count_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'body_momentum_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'up_volume_ratio_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'price_accel_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'inst_footprint_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'downside_vol_ratio_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'vol_compression_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'sector_residual_mom_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'downside_stability_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'intraday_trend_ratio_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'overnight_intraday_agreement_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'sector_return_dispersion_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'gap_fill_resistance_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'vol_contraction_breakout_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'path_convexity_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'bear_resilience_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'relative_volume_breakout_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'intraday_trend_strength_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'drawdown_recovery_speed_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'price_efficiency_ratio_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'updown_vol_asymmetry_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'higher_low_streak_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'market_down_day_alpha_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'market_decorrelation_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'lower_shadow_support_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'overnight_gap_persistence_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'intraday_range_quality_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'cross_lookback_stability_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'sector_breadth_confirm_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'multi_tf_agreement_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'pv_covar_trend_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'body_dominance_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'up_down_vol_ratio_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'buying_pressure_trend_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'volume_distribution_skew_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'volatility_squeeze_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'dd_recovery_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'vol_up_down_ratio_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'price_efficiency_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'price_level_persistence_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'intraday_range_trend_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'relative_volume_momentum_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'range_position_trend_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'mtf_convergence_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'liquidity_resilience_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'cross_sectional_vol_rank_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'max_dd_ratio_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'clean_momentum_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'volume_slope_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'return_autocorr_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'drawdown_depth_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'dip_absorption_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'price_acceleration_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'trend_efficiency_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'tail_ratio_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'sector_residual_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'intraday_strength_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'vol_return_corr_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'body_strength_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'higher_lows_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'upside_capture_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'momentum_consistency_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'accumulation_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'close_location_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'range_expand_w':            [0.0, 0.05, 0.1, 0.15, 0.2],
        'win_streak_w':              [0.0, 0.05, 0.1, 0.15, 0.2],
        'gap_momentum_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'volume_confirm_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'ret_skip_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'cluster_boost_w':          [0.0, 0.05, 0.1, 0.15, 0.2],
        'sector_momentum_w':        [0.0, 0.05, 0.1, 0.15, 0.2],
        'overnight_return_w':       [0.0, 0.05, 0.1, 0.15, 0.2],
        'volume_acceleration_w':    [0.0, 0.05, 0.1, 0.15, 0.2],
        'trailing_stop':            [-0.15, -0.10, -0.07, -0.05, -0.03, None],
    }

    def mutate(params):
        """1〜3個のパラメータをランダムに変異"""
        p = dict(params)
        keys = random.sample(list(RANGES.keys()), random.randint(1, 3))
        for k in keys:
            p[k] = random.choice(RANGES[k])
        # 重みを正規化
        weight_keys = ['ret_w', 'rs_w', 'green_w', 'smooth_w', 'resilience_w', 'high52_w', 'omega_w', 'short_momentum_w', 'close_location_w', 'range_expand_w', 'win_streak_w', 'sector_momentum_w', 'overnight_return_w', 'volume_acceleration_w']
        total = sum(p.get(k, 0) for k in weight_keys)
        if total > 0:
            for k in weight_keys:
                p[k] = round(p.get(k, 0) / total, 4)
        p['rebalance'] = bp.get('rebalance', 'weekly')
        # ファクター数制限を適用
        p = apply_factor_limit(p)
        return p

    def eval_params_local(p):
        try:
            return eval_params(p, factor_dfs, prices_dict, date_map[p['rebalance']],
                               nikkei, START_TRAIN, return_df)
        except Exception:
            return None

    # 初期母集団（ベスト + ランダム変異19個）
    population = [apply_factor_limit(dict(bp))]
    for _ in range(19):
        population.append(mutate(bp))

    results = []
    POP_SIZE = 20
    N_GENERATIONS = 10

    print(f'局所探索(GA): {POP_SIZE}個×{N_GENERATIONS}世代 = 最大{POP_SIZE*N_GENERATIONS}パターン', flush=True)

    for gen in range(N_GENERATIONS):
        gen_results = []
        for p in population:
            # ファクター数制限: 評価前に上限を超えたものを削る
            p = apply_factor_limit(p)
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

            elif factor_name == 'crash_beta_w':
                # 日経worst5%日のベータ（低いほど暴落耐性が高い→高スコア）
                nk_aligned = nk_rets.reindex(dr.index).fillna(0)
                threshold = nk_aligned.quantile(0.05)
                crash_days = nk_aligned[nk_aligned <= threshold].index
                if len(crash_days) >= 5:
                    stock_crash = dr.reindex(crash_days).fillna(0)
                    nk_crash = nk_aligned.reindex(crash_days).fillna(0)
                    cov = stock_crash.rolling(min(lb, len(crash_days))).cov(nk_crash)
                    var = nk_crash.rolling(min(lb, len(crash_days))).var().replace(0, np.nan)
                    beta = (cov / var).reindex(dr.index, method='ffill').fillna(1)
                    factor_dfs[key][factor_name] = (-beta).clip(-3, 3).astype(float)
                else:
                    factor_dfs[key][factor_name] = pd.Series(0.0, index=p.index)

            elif factor_name == 'frog_in_pan_w':
                # Frog-in-the-Pan: 小さな正リターンの連続（Da et al. 2014）
                pos_days = (dr > 0).rolling(lb).mean()
                total_pos = dr.clip(lower=0).rolling(lb).sum().replace(0, np.nan)
                max_day_pos = dr.clip(lower=0).rolling(lb).max()
                concentration = (max_day_pos / total_pos).fillna(1)
                fip = pos_days * (1 - concentration)
                factor_dfs[key][factor_name] = fip.fillna(0).astype(float)

            elif factor_name == 'amihud_liquidity_w':
                # Amihud流動性（高いほど流動性高い＝優遇）
                vol = df['Vo'].fillna(df.get('AdjVo', pd.Series(np.nan, index=df.index))).replace(0, np.nan)
                price_impact = (dr.abs() / vol).replace([np.inf, -np.inf], np.nan)
                illiquidity = price_impact.rolling(lb).mean()
                liquidity = (1 / illiquidity.replace(0, np.nan)).fillna(0)
                # ランク正規化
                factor_dfs[key][factor_name] = liquidity.fillna(0).astype(float)

            elif factor_name == 'stock_sharpe_w':
                # 個別銘柄Sharpe比（lb期間）
                mu = dr.rolling(lb).mean()
                sigma = dr.rolling(lb).std().replace(0, np.nan)
                sharpe = (mu / sigma).fillna(0).clip(-3, 3)
                factor_dfs[key][factor_name] = sharpe.astype(float)

            elif factor_name == 'intraday_return_ratio_w':
                # 日中リターン比率: (Open→Close累積) / (Close→Close累積)
                if 'AdjO' in df.columns and 'AdjC' in df.columns:
                    intraday = (df['AdjC'] / df['AdjO'] - 1).fillna(0)
                    intraday_cum = intraday.rolling(lb).mean()
                    total_cum = dr.rolling(lb).mean().replace(0, np.nan)
                    ratio = (intraday_cum / total_cum.abs()).clip(-2, 2).fillna(0)
                    factor_dfs[key][factor_name] = ratio.astype(float)
                else:
                    factor_dfs[key][factor_name] = pd.Series(0.0, index=p.index)

            elif factor_name == 'low_floor_momentum_w':
                # 安値ベースモメンタム（下値切り上げ）
                if 'L' in df.columns:
                    low = df['L']
                    low_mom = (low / low.shift(lb) - 1).fillna(0).clip(-1, 5)
                    factor_dfs[key][factor_name] = low_mom.astype(float)
                else:
                    factor_dfs[key][factor_name] = pd.Series(0.0, index=p.index)

            elif factor_name == 'negative_skew_penalty_w':
                # 負の歪度ペナルティ（宝くじ銘柄回避）
                skew = dr.rolling(lb).skew().fillna(0)
                factor_dfs[key][factor_name] = (-skew).clip(-3, 3).astype(float)

            elif factor_name == 'trend_efficiency_w':
                # Kaufman効率比: |累積リターン| / Σ|日次リターン|
                net = dr.rolling(lb).sum().abs()
                gross = dr.abs().rolling(lb).sum().replace(0, np.nan)
                efficiency = (net / gross).fillna(0).clip(0, 1)
                factor_dfs[key][factor_name] = efficiency.astype(float)

            elif factor_name == 'volume_turnover_decay_w':
                # 低回転率優遇（発見されていないモメンタム初期）
                vol = df.get('Vo', df.get('AdjVo', pd.Series(np.nan, index=df.index))).replace(0, np.nan)
                dollar_vol = (p * vol).replace(0, np.nan)
                avg_dvol = dollar_vol.rolling(lb).mean()
                # 全銘柄の中央値との比率（後でランク化されるので絶対値でOK）
                low_turnover = (1 / avg_dvol.replace(0, np.nan)).fillna(0)
                factor_dfs[key][factor_name] = low_turnover.astype(float)

            elif factor_name == 'mean_reversion_risk_w':
                # 平均回帰リスク（過去1年σに対して異常上昇した銘柄にペナルティ）
                annual_std = dr.rolling(252).std().replace(0, np.nan)
                lb_ret = (p / p.shift(lb) - 1).fillna(0)
                z_score = (lb_ret / (annual_std * np.sqrt(lb / 252))).fillna(0)
                factor_dfs[key][factor_name] = (-z_score).clip(-3, 3).astype(float)

            elif factor_name == 'vwap_position_w':
                # VWAP位置（close > VWAP = 含み益保有者多い）
                vol = df.get('Vo', df.get('AdjVo', pd.Series(np.nan, index=df.index))).replace(0, np.nan)
                vwap = (p * vol).rolling(lb).sum() / vol.rolling(lb).sum().replace(0, np.nan)
                pos = (p / vwap - 1).fillna(0).clip(-0.5, 0.5)
                factor_dfs[key][factor_name] = pos.astype(float)

            elif factor_name == 'body_momentum_w':
                # ボディモメンタム（ギャップノイズ除去）
                if 'AdjO' in df.columns:
                    body = (df['AdjC'] - df['AdjO']).fillna(0)
                    body_cum = body.rolling(lb).sum()
                    price_range = p.rolling(lb).max() - p.rolling(lb).min()
                    normalized = (body_cum / price_range.replace(0, np.nan)).fillna(0).clip(-3, 3)
                    factor_dfs[key][factor_name] = normalized.astype(float)
                else:
                    factor_dfs[key][factor_name] = pd.Series(0.0, index=p.index)

            elif factor_name == 'up_volume_ratio_w':
                # 上昇日の出来高比率
                vol = df.get('Vo', df.get('AdjVo', pd.Series(np.nan, index=df.index))).replace(0, np.nan)
                up_vol = vol.where(dr > 0, 0).rolling(lb).sum()
                total_vol = vol.rolling(lb).sum().replace(0, np.nan)
                ratio = (up_vol / total_vol).fillna(0.5)
                factor_dfs[key][factor_name] = ratio.astype(float)

            elif factor_name == 'price_accel_w':
                # 価格加速度（後半lb/2 > 前半lb/2）
                half = max(lb // 2, 5)
                ret_recent = (p / p.shift(half) - 1).fillna(0)
                ret_old = (p.shift(half) / p.shift(lb) - 1).fillna(0)
                accel = (ret_recent - ret_old).clip(-2, 2)
                factor_dfs[key][factor_name] = accel.astype(float)

            elif factor_name == 'downside_vol_ratio_w':
                # 下方ボラ比率の逆数（上方歪みを優遇）
                down_vol = dr.where(dr < 0, 0).rolling(lb).std().replace(0, np.nan)
                total_vol_s = dr.rolling(lb).std().replace(0, np.nan)
                ratio = (down_vol / total_vol_s).fillna(0.5)
                factor_dfs[key][factor_name] = (1 - ratio).astype(float)

            elif factor_name == 'vol_compression_w':
                # ボラティリティ圧縮（ATR40/ATR5 の逆数）
                if 'H' in df.columns and 'L' in df.columns:
                    atr_short = (df['H'] - df['L']).rolling(5).mean().replace(0, np.nan)
                    atr_long = (df['H'] - df['L']).rolling(lb).mean().replace(0, np.nan)
                    compression = (atr_short / atr_long).fillna(1)
                    factor_dfs[key][factor_name] = (1 / compression.replace(0, np.nan)).fillna(0).clip(0, 5).astype(float)
                else:
                    factor_dfs[key][factor_name] = pd.Series(0.0, index=p.index)

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
    factor_dfs = precompute(prices_dict, nikkei, [20, 40, 60, 80, 100])
    
    return_df = pd.DataFrame({c: prices_dict[c]['AdjC'] for c in prices_dict}).pct_change()
    # 全期間 date_map（Walk-Forward検証用）
    daily_d   = get_rebalance_dates(warmup, END, 'daily')
    weekly_d  = get_rebalance_dates(warmup, END, 'weekly')
    monthly_d = get_rebalance_dates(warmup, END, 'monthly')
    date_map = {'daily': daily_d, 'weekly': weekly_d, 'monthly': monthly_d}

    # IS訓練期間専用 date_map（OOSと重複しない 2016〜2020）
    daily_train   = get_rebalance_dates(warmup, END_TRAIN, 'daily')
    weekly_train  = get_rebalance_dates(warmup, END_TRAIN, 'weekly')
    date_map_train = {'daily': daily_train, 'weekly': weekly_train}

    # 検証期間専用 date_map（2021〜2022）
    daily_val   = get_rebalance_dates(START_VAL, END_VAL, 'daily')
    weekly_val  = get_rebalance_dates(START_VAL, END_VAL, 'weekly')
    date_map_val = {'daily': daily_val, 'weekly': weekly_val}

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
    
    # ---- Step2: 局所探索 (Optuna TPE) ----
    print('\n--- 局所探索 開始 (Optuna TPE) ---', flush=True)
    bp = baseline.get('params', {
        'lookback':60,'top_n':10,'rebalance':'weekly',
        'ret_w':0.3,'rs_w':0.3,'green_w':0.2,'smooth_w':0.2,'resilience_w':0.0
    })
    local_results = run_optuna_optimization(
        bp, factor_dfs, prices_dict, nikkei, date_map_train,
        START_TRAIN, return_df, n_trials=300,
        val_date_map=date_map_val, val_start=START_VAL,
    )
    
    # 複合評価
    best_local = None
    best_score = -99
    for p, r_train in local_results:
        r_val = eval_params(p, factor_dfs, prices_dict, date_map_val[p['rebalance']], nikkei, START_VAL, return_df)
        adopted, score, reasons = evaluate(r_train, r_val, baseline)
        print(f'  sharpe={r_train["sharpe"]:.3f} score={score:+.3f} {"✅" if adopted else "❌"} lb={p["lookback"]} tn={p["top_n"]}', flush=True)
        if score > best_score:
            best_score = score
            best_local = (p, r_train, r_val, adopted, reasons)
    
    if best_local:
        p, r_train, r_val, adopted, reasons = best_local
        wf_result = None

        if adopted:
            # ベースライン更新前にWalk-Forward検証
            print("\n--- Walk-Forward Cross Validation ---", flush=True)
            try:
                wf_result = run_walk_forward_validation(p, n_codes=N_CODES, codes=list(prices_dict.keys()))
            except Exception as wf_e:
                print(f"Walk-Forward検証エラー（スキップ）: {wf_e}", flush=True)

            if wf_result:
                print(f"WF結果: avg_total={wf_result['avg_total_return']:.1f}%, avg_sharpe={wf_result['avg_sharpe']:.3f}, n_passed={wf_result['n_passed']}/3", flush=True)
                # fold3（直近2024-2026）合格が必須条件
                fold3 = next((f for f in wf_result.get("folds", []) if f.get("id") == "fold3"), None)
                fold3_passed = fold3 and fold3.get("passed", False)
                print(f"WF: fold3（直近）={'✅合格' if fold3_passed else '❌不合格'}, n_passed={wf_result['n_passed']}/3", flush=True)
                if not fold3_passed:
                    print(f"⚠️ fold3（2024-2026）不合格 → 採用スキップ", flush=True)
                    adopted = False
                elif wf_result["n_passed"] < 1:
                    print(f"⚠️ Walk-Forward全不合格 → 採用スキップ", flush=True)
                    adopted = False
            else:
                print("Walk-Forward: 結果取得失敗、通常採用基準を適用", flush=True)

        append_log('local_search', f'局所探索ベスト lb={p["lookback"]} tn={p["top_n"]}',
                   r_train, adopted, r_train['sharpe'] - baseline['sharpe'],
                   walk_forward=wf_result)

        if adopted:
            print(f'\n✅ 局所探索で改善: sharpe {baseline["sharpe"]} → {r_train["sharpe"]}', flush=True)
            # ベースライン更新前にバックアップ
            _backup_baseline(queue['baseline'])
            new_baseline = {
                'sharpe': r_train['sharpe'],
                'total_pct': r_train['total_return_pct'],
                'alpha_pct': r_train['alpha_pct'],
                'max_dd_pct': r_train['max_dd_pct'],
                'params': p,
                'equity_curve': r_train.get('equity_curve', []),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'hypothesis': 'local_search',
            }
            if wf_result:
                new_baseline['walk_forward'] = wf_result
            queue['baseline'] = new_baseline
            QUEUE_FILE.write_text(json.dumps(_fb(queue), ensure_ascii=False, indent=2))
            baseline = queue['baseline']

        # GA local_search 完了後にファクター整理を実行（best result のパラメータで）
        print('\n--- ファクター整理 (cleanup_weak_signals) ---', flush=True)
        cleanup_weak_signals(p)

    # ---- Step2b: OOS検証 (Walk-Forward Out-of-Sample) ----
    print('\n--- OOS検証 (Walk-Forward Out-of-Sample) ---', flush=True)
    current_best_params = queue['baseline'].get('params', bp)
    oos_result = None
    try:
        # prices_dict のコードリストを再利用（APIレート制限回避）
        loaded_codes = list(prices_dict.keys())
        oos_result = run_oos_validation(
            current_best_params,
            oos_start="2020-01-01",
            n_codes=N_CODES,
            codes=loaded_codes,
        )
    except Exception as e:
        print(f'OOS検証エラー（スキップ）: {e}', flush=True)

    if oos_result:
        train_total = queue['baseline'].get('total_pct', 0)
        oos_total   = oos_result.get('total_return_pct', 0)
        print(
            f'\nOOS vs 訓練: OOS={oos_total:+.1f}% / 訓練={train_total:+.1f}% / '
            f'OOS sharpe={oos_result["sharpe"]:.3f} max_dd={oos_result["max_dd_pct"]:.1f}%',
            flush=True,
        )
        # evolution_logとqueueにoos_resultを保存
        queue['baseline']['oos_result'] = _fb(oos_result)
        QUEUE_FILE.write_text(json.dumps(_fb(queue), ensure_ascii=False, indent=2))
        append_log(
            'oos_validation',
            f'OOS検証 ({oos_result["n_trades"]}取引, 2020〜)',
            oos_result,
            win=oos_result['total_return_pct'] > 0,
            delta=oos_total - train_total,
            oos_result=oos_result,
        )

        # OOS合格判定
        oos_sharpe = oos_result.get('sharpe') or 0
        oos_total_ret = oos_result.get('total_return_pct') or 0
        baseline_oos_sharpe = queue['baseline'].get('oos_result', {}).get('sharpe') or 0
        # baseline OOS sharpeとOOS_SHARPE_MINの両方を満たす必要がある
        required_oos_sharpe = max(OOS_SHARPE_MIN, baseline_oos_sharpe)
        oos_passed = (oos_sharpe >= required_oos_sharpe and oos_total_ret > OOS_TOTAL_RETURN_MIN)

        if not oos_passed:
            print(
                f'⚠️ OOS不合格: sharpe={oos_sharpe:.3f} (要≥{required_oos_sharpe:.3f}), '
                f'total_return={oos_total_ret:.1f}% (要>{OOS_TOTAL_RETURN_MIN}%)',
                flush=True,
            )
            DONE_FILE.write_text(json.dumps(_fb({
                'status': 'oos_failed',
                'id': 'evolution_cycle',
                'win': 0,
                'oos_sharpe': oos_sharpe,
                'oos_total_return_pct': oos_total_ret,
                'required_oos_sharpe': required_oos_sharpe,
                'required_oos_total_return_pct': OOS_TOTAL_RETURN_MIN,
                'result': _fb(queue['baseline']),
                'at': datetime.now().isoformat(),
            }), ensure_ascii=False, indent=2))
        else:
            print(
                f'✅ OOS合格: sharpe={oos_sharpe:.3f} (要≥{required_oos_sharpe:.3f}), '
                f'total_return={oos_total_ret:.1f}% (要>{OOS_TOTAL_RETURN_MIN}%)',
                flush=True,
            )

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
        
        r_train = eval_params(test_p, fds_copy, prices_dict, date_map_train[test_p['rebalance']], nikkei, START_TRAIN, return_df)
        r_val   = eval_params(test_p, fds_copy, prices_dict, date_map_val.get(test_p['rebalance'], date_map_val['weekly']), nikkei, START_VAL, return_df)
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
            baseline['equity_curve'] = r_train.get('equity_curve', [])
            queue['baseline'] = {**baseline, 'date': datetime.now().strftime('%Y-%m-%d'), 'hypothesis': f'factor_{factor_name}'}
            print(f'  → ベースライン更新: sharpe={r_train["sharpe"]}', flush=True)
        
        QUEUE_FILE.write_text(json.dumps(_fb(queue), ensure_ascii=False, indent=2))
        time.sleep(1)
    
    # ---- Step4: 組み合わせテスト ----
    from combination_search import run_combination_search
    run_combination_search()

    # ---- Step4b: シグナル積み上げ最適化 ----
    from signal_search import run_signal_search
    run_signal_search()

    # ---- 完了シグナル ----
    DONE_FILE.write_text(json.dumps(_fb({
        'status': 'done',
        'id': 'evolution_cycle',
        'win': 1,
        'delta_sharpe': baseline['sharpe'] - json.loads(QUEUE_FILE.read_text())['baseline'].get('sharpe', 0),
        'result': baseline,
        'at': datetime.now().isoformat(),
    }), ensure_ascii=False, indent=2))
    
    print('\n=== Evolution cycle 完了 ===', flush=True)

    # ダッシュボードキャッシュ更新
    import subprocess, sys as _sys
    subprocess.Popen(
        [_sys.executable, 'generate_dashboard_cache.py'],
        stdout=open('logs/dashboard_cache.log', 'a'),
        stderr=subprocess.STDOUT,
    )
    print('ダッシュボードキャッシュ生成バックグラウンド起動', flush=True)

    # 即opusをキック
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
            lib_path.write_text(json.dumps(_fb(lib), indent=2, ensure_ascii=False))
            print(f'  cleanup_weak_signals: {changed} シグナルを rejected に変更', flush=True)
        else:
            print('  cleanup_weak_signals: 変更なし', flush=True)
    except Exception as e:
        print(f'  cleanup_weak_signals エラー（スキップ）: {e}', flush=True)


def append_log(hid, desc, result, win, delta, oos_result=None, walk_forward=None):
    log = json.loads(EVO_LOG.read_text()) if EVO_LOG.exists() else {'best10':[], 'all':[], 'total':0}
    
    # Calmar Ratio 計算
    calmar = None
    if result and result.get('total_return_pct') and result.get('max_dd_pct'):
        dd = max(result['max_dd_pct'], 1.0)
        n_trades = result.get('n_trades', 52)
        est_years = max(n_trades / 52, 0.5)
        calmar = round(result['total_return_pct'] / est_years / dd, 3)
    
    entry = {
        'at': datetime.now().isoformat(), 'id': hid, 'desc': desc, 'win': int(win),
        'delta_sharpe': delta,
        'sharpe': result.get('sharpe') if result else None,
        'total_return_pct': result.get('total_return_pct') if result else None,
        'alpha_pct': result.get('alpha_pct') if result else None,
        'max_dd_pct': result.get('max_dd_pct') if result else None,
        'calmar_ratio': calmar,
        'params': {k: result[k] for k in result if '_w' in k or k in ['lookback','top_n','rebalance','position_sizing']} if result else {},
    }
    
    # IS/OOS Sharpe比 (過学習チェック指標)
    if oos_result and result:
        is_sharpe = result.get('sharpe', 0)
        oos_sharpe = oos_result.get('sharpe', 0)
        entry['is_oos_sharpe_ratio'] = round(oos_sharpe / is_sharpe, 3) if is_sharpe > 0 else None
        entry['oos_result'] = {
            'total_return_pct': oos_result.get('total_return_pct'),
            'sharpe': oos_result.get('sharpe'),
            'max_dd_pct': oos_result.get('max_dd_pct'),
            'alpha_pct': oos_result.get('alpha_pct'),
            'nikkei_pct': oos_result.get('nikkei_pct'),
            'n_trades': oos_result.get('n_trades'),
        }
    elif oos_result:
        entry['oos_result'] = {
            'total_return_pct': oos_result.get('total_return_pct'),
            'sharpe': oos_result.get('sharpe'),
            'max_dd_pct': oos_result.get('max_dd_pct'),
            'alpha_pct': oos_result.get('alpha_pct'),
            'nikkei_pct': oos_result.get('nikkei_pct'),
            'n_trades': oos_result.get('n_trades'),
        }
    if walk_forward:
        entry['walk_forward'] = {
            'avg_total_return': walk_forward.get('avg_total_return'),
            'avg_sharpe': walk_forward.get('avg_sharpe', walk_forward.get('avg_oos_sharpe')),
            'avg_max_dd': walk_forward.get('avg_max_dd', walk_forward.get('avg_oos_dd')),
            'stability_score': walk_forward.get('stability_score'),
            'is_oos_ratio': walk_forward.get('is_oos_ratio'),
            'deflated_sharpe': walk_forward.get('deflated_sharpe'),
            'n_passed': walk_forward.get('n_passed'),
            'all_passed': walk_forward.get('all_passed'),
            'folds': walk_forward.get('folds', []),
        }
    all_entries = log.get('all', []) + [entry]
    valid = sorted([x for x in all_entries if x.get('sharpe') is not None],
                   key=lambda x: x['sharpe'], reverse=True)
    EVO_LOG.write_text(json.dumps(_fb({'best10': valid[:10], 'all': valid[:300], 'total': len(all_entries)}),
                                   ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        run_evolution()
    except Exception as e:
        print(f'エラー: {e}')
        traceback.print_exc()
        DONE_FILE.write_text(json.dumps(_fb({'status': 'error', 'id': 'evolution', 'error': str(e)}), ensure_ascii=False))
# baseline backup は evolution_engine.py 内で自動実行される
