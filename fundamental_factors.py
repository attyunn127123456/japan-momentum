"""
ファンダメンタルズファクター計算モジュール。
バックテストに組み込むための各種ファクターをDataFrameで返す。

J-Quants V2 カラム名マッピング:
  fins_summary: DiscDate, Code(5桁), EPS, Sales, NP, BPS, EqAR(=ROE比率), FEPS(予想EPS)
  earnings_calendar: Date, Code(5桁), CoName, FQ, Section
  dividend: PubDate, Code(5桁), DivRate, ExDate, RecDate
"""
import pandas as pd
import numpy as np
from pathlib import Path

FUNDAMENTALS_DIR = Path("data/fundamentals")


def _load_parquet(name: str) -> pd.DataFrame:
    """キャッシュ済みparquetを読み込む。なければ空DataFrame。"""
    path = FUNDAMENTALS_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _normalize_code(code: str) -> str:
    """4桁コードを5桁(末尾0)に変換。既に5桁ならそのまま。"""
    code = code.replace(".T", "")
    if len(code) == 4:
        return code + "0"
    return code


def compute_fundamental_factors(codes: list, date: str) -> pd.DataFrame:
    """
    指定日時点での各銘柄のファンダメンタルズファクターを計算。

    Parameters
    ----------
    codes : list of str
        銘柄コード（4桁 or 5桁）のリスト
    date : str
        基準日 (YYYY-MM-DD)

    Returns
    -------
    pd.DataFrame
        index=Code, columns:
        - eps_growth_yoy: EPS前年同期比成長率
        - revenue_growth_yoy: 売上前年同期比成長率
        - roe: 直近ROE（自己資本比率 EqAR）
        - earnings_surprise: 決算サプライズ（実績EPS vs 予想EPS）
        - earnings_momentum: 直近2期のEPS改善トレンド
        - dividend_yield: 直近配当金額（DPS）。株価と合わせて利回り算出用。
        - days_to_earnings: 次回決算発表までの日数（短いほど注意）
    """
    cutoff = pd.Timestamp(date)
    results = []

    # ── データ読み込み ──
    fins = _load_parquet("fins_summary")
    earnings_cal = _load_parquet("earnings_calendar")
    dividend = _load_parquet("dividend")

    # ── 前処理 ──
    if not fins.empty:
        if "DiscDate" in fins.columns:
            fins["DiscDate"] = pd.to_datetime(fins["DiscDate"], errors="coerce")
        for col in ["EPS", "Sales", "NP", "BPS", "EqAR", "FEPS"]:
            if col in fins.columns:
                fins[col] = pd.to_numeric(fins[col], errors="coerce")

    if not earnings_cal.empty and "Date" in earnings_cal.columns:
        earnings_cal["Date"] = pd.to_datetime(earnings_cal["Date"], errors="coerce")

    if not dividend.empty and "DivRate" in dividend.columns:
        dividend["DivRate"] = pd.to_numeric(dividend["DivRate"], errors="coerce")

    # ── 銘柄ごとに計算 ──
    for code in codes:
        code5 = _normalize_code(code)
        row = {"Code": code}

        # --- 財務サマリーベースのファクター ---
        if not fins.empty and "Code" in fins.columns and "DiscDate" in fins.columns:
            c_fins = fins[
                (fins["Code"] == code5) & (fins["DiscDate"] <= cutoff)
            ].sort_values("DiscDate")

            if len(c_fins) >= 2:
                latest = c_fins.iloc[-1]
                prev = c_fins.iloc[-2]

                # EPS成長率 (YoY)
                eps_cur = latest.get("EPS", np.nan)
                eps_prev = prev.get("EPS", np.nan)
                if pd.notna(eps_cur) and pd.notna(eps_prev) and eps_prev != 0:
                    row["eps_growth_yoy"] = (eps_cur / abs(eps_prev)) - 1
                else:
                    row["eps_growth_yoy"] = np.nan

                # 売上成長率 (YoY)
                rev_cur = latest.get("Sales", np.nan)
                rev_prev = prev.get("Sales", np.nan)
                if pd.notna(rev_cur) and pd.notna(rev_prev) and rev_prev != 0:
                    row["revenue_growth_yoy"] = (rev_cur / abs(rev_prev)) - 1
                else:
                    row["revenue_growth_yoy"] = np.nan

                # ROE (EqAR = 自己資本比率、ROEの近似として使用)
                roe = latest.get("EqAR", np.nan)
                row["roe"] = roe if pd.notna(roe) else np.nan

                # 決算サプライズ (実績EPS vs 予想EPS)
                forecast_eps = latest.get("FEPS", np.nan)
                if pd.notna(eps_cur) and pd.notna(forecast_eps) and forecast_eps != 0:
                    row["earnings_surprise"] = (eps_cur - forecast_eps) / abs(forecast_eps)
                else:
                    row["earnings_surprise"] = np.nan

                # Earnings Momentum: 直近2期のEPS改善トレンド
                if len(c_fins) >= 3:
                    eps_prev2 = c_fins.iloc[-3].get("EPS", np.nan)
                    if pd.notna(eps_cur) and pd.notna(eps_prev) and pd.notna(eps_prev2):
                        d1 = eps_cur - eps_prev
                        d2 = eps_prev - eps_prev2
                        row["earnings_momentum"] = (d1 + d2) / 2
                    else:
                        row["earnings_momentum"] = np.nan
                elif pd.notna(eps_cur) and pd.notna(eps_prev):
                    row["earnings_momentum"] = eps_cur - eps_prev
                else:
                    row["earnings_momentum"] = np.nan

            elif len(c_fins) == 1:
                latest = c_fins.iloc[-1]
                row["roe"] = latest.get("EqAR", np.nan)
                for k in ["eps_growth_yoy", "revenue_growth_yoy", "earnings_surprise", "earnings_momentum"]:
                    row[k] = np.nan
            else:
                for k in ["eps_growth_yoy", "revenue_growth_yoy", "roe", "earnings_surprise", "earnings_momentum"]:
                    row[k] = np.nan
        else:
            for k in ["eps_growth_yoy", "revenue_growth_yoy", "roe", "earnings_surprise", "earnings_momentum"]:
                row[k] = np.nan

        # --- 配当利回り (DivRate = 1株配当金額) ---
        if not dividend.empty and "Code" in dividend.columns and "DivRate" in dividend.columns:
            c_div = dividend[dividend["Code"] == code5].copy()
            if not c_div.empty:
                # StatCode=1 は確定、ExDateでソート
                if "ExDate" in c_div.columns:
                    c_div["ExDate"] = pd.to_datetime(c_div["ExDate"], errors="coerce")
                    c_div = c_div.dropna(subset=["ExDate"]).sort_values("ExDate")
                latest_div = c_div.iloc[-1]
                dps = latest_div.get("DivRate", np.nan)
                row["dividend_yield"] = dps if pd.notna(dps) else np.nan
            else:
                row["dividend_yield"] = np.nan
        else:
            row["dividend_yield"] = np.nan

        # --- 次回決算発表までの日数 ---
        if not earnings_cal.empty and "Code" in earnings_cal.columns and "Date" in earnings_cal.columns:
            c_cal = earnings_cal[
                (earnings_cal["Code"] == code5) & (earnings_cal["Date"] > cutoff)
            ].sort_values("Date")
            if not c_cal.empty:
                next_date = c_cal.iloc[0]["Date"]
                row["days_to_earnings"] = (next_date - cutoff).days
            else:
                row["days_to_earnings"] = np.nan
        else:
            row["days_to_earnings"] = np.nan

        results.append(row)

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.set_index("Code")
    return df


if __name__ == "__main__":
    test_codes = ["7203", "6758", "9984"]
    date = "2024-01-10"
    print(f"ファクター計算テスト: codes={test_codes}, date={date}")
    df = compute_fundamental_factors(test_codes, date)
    print(df.to_string())
