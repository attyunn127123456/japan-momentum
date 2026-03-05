"""
J-Quants V2 全ファンダメンタル・需給データ取得 & キャッシュ。
data/fundamentals/ 以下にparquetで保存。
"""
import time, json, requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

API_KEY = 'cph3PdiF8zxH9GxClcFfShcJdSUzuNpV9ho_zMPm4a8'
HEADERS = {'x-api-key': API_KEY}
BASE = 'https://api.jquants.com/v2'
CACHE_DIR = Path('data/fundamentals')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _get(ep, params=None, timeout=30):
    r = requests.get(f'{BASE}{ep}', headers=HEADERS, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _get_all(ep, params=None):
    """ページネーション対応"""
    all_data = []
    p = dict(params or {})
    while True:
        d = _get(ep, p)
        all_data.extend(d.get('data', []))
        pk = d.get('pagination_key')
        if not pk:
            break
        p['pagination_key'] = pk
    return all_data

# ---- 財務サマリー (EPS, 売上, 利益率) ----
def fetch_fins_summary(codes, start='2022-01-01'):
    """全銘柄の財務サマリー取得"""
    out = Path('data/fundamentals/fins_summary.parquet')
    if out.exists() and (time.time() - out.stat().st_mtime) < 86400:
        print(f'fins_summary: キャッシュ使用 ({len(pd.read_parquet(out))}行)')
        return pd.read_parquet(out)
    
    print('fins_summary 取得中...')
    all_data = []
    for i, code in enumerate(codes):
        try:
            data = _get_all('/fins/summary', {'code': code})
            all_data.extend(data)
        except Exception as e:
            pass
        if (i+1) % 50 == 0:
            print(f'  {i+1}/{len(codes)}', flush=True)
        time.sleep(0.05)
    
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df.to_parquet(out)
    print(f'fins_summary: {len(df)}行保存')
    return df

# ---- 財務詳細 (BS/PL/CF) ----
def fetch_fins_details(codes):
    out = Path('data/fundamentals/fins_details.parquet')
    if out.exists() and (time.time() - out.stat().st_mtime) < 86400:
        print(f'fins_details: キャッシュ使用')
        return pd.read_parquet(out)
    
    print('fins_details 取得中...')
    all_data = []
    for i, code in enumerate(codes[:100]):  # まず上位100銘柄
        try:
            data = _get_all('/fins/details', {'code': code})
            all_data.extend(data)
        except Exception as e:
            pass
        if (i+1) % 20 == 0:
            print(f'  {i+1}/100', flush=True)
        time.sleep(0.1)
    
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df.to_parquet(out)
    print(f'fins_details: {len(df)}行保存')
    return df

# ---- 信用取引残高 ----
def fetch_margin_interest(start='2022-01-01'):
    out = Path('data/fundamentals/margin_interest.parquet')
    if out.exists() and (time.time() - out.stat().st_mtime) < 86400:
        print('margin_interest: キャッシュ使用')
        return pd.read_parquet(out)
    
    print('margin_interest 取得中...')
    try:
        data = _get_all('/markets/margin-interest', {'from': start})
        df = pd.DataFrame(data)
        if not df.empty:
            df.to_parquet(out)
            print(f'margin_interest: {len(df)}行保存')
        return df
    except Exception as e:
        print(f'margin_interest エラー: {e}')
        return pd.DataFrame()

# ---- 空売り残高 ----
def fetch_short_sale(start='2023-01-01'):
    out = Path('data/fundamentals/short_sale.parquet')
    if out.exists() and (time.time() - out.stat().st_mtime) < 86400:
        print('short_sale: キャッシュ使用')
        return pd.read_parquet(out)
    
    print('short_sale 取得中...')
    all_data = []
    # 日付ごとに取得
    d = datetime.strptime(start, '%Y-%m-%d')
    end = datetime.now()
    while d <= end:
        try:
            data = _get_all('/markets/short-sale-report', {'date': d.strftime('%Y-%m-%d')})
            all_data.extend(data)
        except:
            pass
        d += timedelta(days=7)  # 週次データ
        time.sleep(0.1)
    
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df.to_parquet(out)
    print(f'short_sale: {len(df)}行保存')
    return df

# ---- 投資部門別売買 (外国人・機関) ----
def fetch_investor_types(start='2022-01-01'):
    out = Path('data/fundamentals/investor_types.parquet')
    if out.exists() and (time.time() - out.stat().st_mtime) < 86400:
        print('investor_types: キャッシュ使用')
        return pd.read_parquet(out)
    
    print('investor_types 取得中...')
    try:
        data = _get_all('/equities/investor-types', {'section': 'TSEPrime', 'from': start})
        df = pd.DataFrame(data)
        if not df.empty:
            df.to_parquet(out)
            print(f'investor_types: {len(df)}行保存')
        return df
    except Exception as e:
        print(f'investor_types エラー: {e}')
        return pd.DataFrame()

# ---- 空売り比率 (業種別) ----
def fetch_short_ratio(start='2022-01-01'):
    out = Path('data/fundamentals/short_ratio.parquet')
    if out.exists() and (time.time() - out.stat().st_mtime) < 86400:
        print('short_ratio: キャッシュ使用')
        return pd.read_parquet(out)
    
    print('short_ratio 取得中...')
    try:
        data = _get_all('/markets/short-ratio', {'from': start})
        df = pd.DataFrame(data)
        if not df.empty:
            df.to_parquet(out)
            print(f'short_ratio: {len(df)}行保存')
        return df
    except Exception as e:
        print(f'short_ratio エラー: {e}')
        return pd.DataFrame()

# ---- 指数データ (TOPIX, 各指数) ----
def fetch_indices(start='2022-01-01'):
    out = Path('data/fundamentals/indices.parquet')
    if out.exists() and (time.time() - out.stat().st_mtime) < 86400:
        print('indices: キャッシュ使用')
        return pd.read_parquet(out)
    
    print('indices 取得中...')
    try:
        # TOPIX, TOPIX Small, Growth 250
        all_data = []
        for code in ['0028', '0016', '0017']:  # TOPIX系
            try:
                data = _get_all('/indices/bars/daily', {'indexcode': code, 'from': start})
                all_data.extend(data)
            except:
                pass
            time.sleep(0.1)
        df = pd.DataFrame(all_data)
        if not df.empty:
            df.to_parquet(out)
            print(f'indices: {len(df)}行保存')
        return df
    except Exception as e:
        print(f'indices エラー: {e}')
        return pd.DataFrame()

# ---- ファクター計算 ----
def compute_fundamental_factors(codes, date):
    """
    銘柄×日付のファンダメンタルファクターを計算して返す。
    returns: dict {code: {eps_growth, revenue_growth, margin_trend, 
                          credit_ratio, short_interest, foreign_buying}}
    """
    factors = {}
    
    # 財務サマリーから EPS成長率・売上成長率
    fs_path = Path('data/fundamentals/fins_summary.parquet')
    if fs_path.exists():
        fs = pd.read_parquet(fs_path)
        if 'Code' in fs.columns and 'DisclosedDate' in fs.columns:
            fs['DisclosedDate'] = pd.to_datetime(fs['DisclosedDate'], errors='coerce')
            for code in codes:
                c_data = fs[fs['Code'] == code].sort_values('DisclosedDate')
                c_data = c_data[c_data['DisclosedDate'] <= date]
                if len(c_data) >= 2:
                    latest = c_data.iloc[-1]
                    prev   = c_data.iloc[-2]
                    # EPS成長率
                    eps_cur  = float(latest.get('EarningsPerShare', 0) or 0)
                    eps_prev = float(prev.get('EarningsPerShare', 0) or 1)
                    eps_growth = (eps_cur / abs(eps_prev) - 1) if eps_prev != 0 else 0
                    # 売上成長率
                    rev_cur  = float(latest.get('NetSales', 0) or 0)
                    rev_prev = float(prev.get('NetSales', 0) or 1)
                    rev_growth = (rev_cur / abs(rev_prev) - 1) if rev_prev != 0 else 0
                    
                    factors.setdefault(code, {})
                    factors[code]['eps_growth'] = eps_growth
                    factors[code]['rev_growth'] = rev_growth
    
    # 信用倍率 (買い残/売り残)
    mi_path = Path('data/fundamentals/margin_interest.parquet')
    if mi_path.exists():
        mi = pd.read_parquet(mi_path)
        if 'Code' in mi.columns and 'Date' in mi.columns:
            mi['Date'] = pd.to_datetime(mi['Date'], errors='coerce')
            for code in codes:
                c_data = mi[mi['Code'] == code]
                c_data = c_data[c_data['Date'] <= date].sort_values('Date')
                if not c_data.empty:
                    latest = c_data.iloc[-1]
                    buy = float(latest.get('LongMarginTradeVolume', 0) or 0)
                    sell = float(latest.get('ShortMarginTradeVolume', 1) or 1)
                    credit_ratio = buy / sell if sell > 0 else 1.0
                    factors.setdefault(code, {})
                    factors[code]['credit_ratio'] = credit_ratio
    
    return factors

if __name__ == '__main__':
    from universe import get_top_liquid_tickers
    codes = get_top_liquid_tickers(500)
    print(f'対象: {len(codes)}銘柄')
    
    fetch_fins_summary(codes)
    fetch_margin_interest()
    fetch_investor_types()
    fetch_short_ratio()
    fetch_indices()
    # fetch_short_sale()  # 重いので後回し
    
    print('全データ取得完了')
    Path('data/fundamentals/fetch_done.json').write_text(
        json.dumps({'at': datetime.now().isoformat(), 'codes': len(codes)})
    )
