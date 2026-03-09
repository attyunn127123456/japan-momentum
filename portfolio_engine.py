"""
ポートフォリオリスク管理エンジン

マルチストラテジー型ヘッジファンドモデルの中核。
各ポッド（戦略）へのリスク予算割り当て、ポートフォリオ全体のDD・VaR計算、
ポッド間相関マトリックス管理、リスク制限超過時のアラートを提供する。

Usage:
    from portfolio_engine import PortfolioEngine
    engine = PortfolioEngine(initial_capital=10_000_000)
    engine.add_pod("momentum", allocation_pct=50, dd_limit_pct=15)
    engine.update_pod_nav("momentum", 5_200_000)
    risk = engine.compute_risk_metrics()
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ─── Pod: 個別戦略ユニット ───

class Pod:
    """個別戦略ポッド。独自のNAV・DD・リスク予算を持つ。"""

    def __init__(self, name: str, allocation_pct: float, dd_limit_pct: float,
                 initial_nav: float = 0.0, description: str = ""):
        self.name = name
        self.allocation_pct = allocation_pct
        self.dd_limit_pct = dd_limit_pct
        self.description = description

        self.nav = initial_nav
        self.peak_nav = initial_nav
        self.nav_history: list[dict] = []  # [{"date": ..., "nav": ...}]
        self.daily_returns: list[float] = []

        # ポジション管理
        self.positions: list[dict] = []
        self.closed_trades: list[dict] = []

        # 状態
        self.active = True
        self.halved = False  # DD制限の50%到達でポジション半減済みか

    @property
    def drawdown_pct(self) -> float:
        """現在のドローダウン（%）"""
        if self.peak_nav <= 0:
            return 0.0
        return (self.nav / self.peak_nav - 1) * 100

    @property
    def long_exposure(self) -> float:
        return sum(
            p["quantity"] * p.get("current_price", p["entry_price"])
            for p in self.positions if p["side"] == "long"
        )

    @property
    def short_exposure(self) -> float:
        return sum(
            p["quantity"] * p.get("current_price", p["entry_price"])
            for p in self.positions if p["side"] == "short"
        )

    @property
    def net_exposure(self) -> float:
        return self.long_exposure - self.short_exposure

    @property
    def gross_exposure(self) -> float:
        return self.long_exposure + self.short_exposure

    def update_nav(self, new_nav: float, date: Optional[str] = None):
        """NAVを更新し、ピークとDDを再計算"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        old_nav = self.nav
        self.nav = new_nav
        if new_nav > self.peak_nav:
            self.peak_nav = new_nav
        self.nav_history.append({"date": date, "nav": new_nav})
        if old_nav > 0:
            self.daily_returns.append(new_nav / old_nav - 1)

    def check_dd_limits(self) -> list[dict]:
        """DDリミットチェック。アラートリストを返す。"""
        alerts = []
        dd = self.drawdown_pct  # negative value

        if dd <= -self.dd_limit_pct * 1.5:
            alerts.append({
                "level": "critical",
                "pod": self.name,
                "message": f"DD {dd:.1f}% ≤ -{self.dd_limit_pct * 1.5:.1f}% → ポッド停止",
                "action": "halt",
            })
            self.active = False
        elif dd <= -self.dd_limit_pct:
            alerts.append({
                "level": "warning",
                "pod": self.name,
                "message": f"DD {dd:.1f}% ≤ -{self.dd_limit_pct:.1f}% → ポジション50%縮小",
                "action": "halve",
            })
            self.halved = True
        elif dd <= -self.dd_limit_pct * 0.8:
            alerts.append({
                "level": "caution",
                "pod": self.name,
                "message": f"DD {dd:.1f}%: DDリミット（-{self.dd_limit_pct:.1f}%）の80%に接近",
                "action": "monitor",
            })

        return alerts

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "allocation_pct": self.allocation_pct,
            "dd_limit_pct": self.dd_limit_pct,
            "nav": round(self.nav, 0),
            "peak_nav": round(self.peak_nav, 0),
            "dd_pct": round(self.drawdown_pct, 2),
            "long_exposure": round(self.long_exposure, 0),
            "short_exposure": round(self.short_exposure, 0),
            "net_exposure": round(self.net_exposure, 0),
            "gross_exposure": round(self.gross_exposure, 0),
            "positions_count": len(self.positions),
            "active": self.active,
            "description": self.description,
        }


# ─── PortfolioEngine: ポートフォリオ全体の管理 ───

class PortfolioEngine:
    """マルチストラテジーポートフォリオのリスク管理エンジン"""

    # デフォルトのリスクパラメータ
    DEFAULT_PORTFOLIO_DD_LIMIT = 20.0        # 全体DDリミット（%）
    DEFAULT_MAX_SINGLE_POSITION = 10.0       # 1銘柄上限（対NAV %）
    DEFAULT_MAX_SECTOR_EXPOSURE = 25.0       # 1セクター上限（対NAV %）
    DEFAULT_MAX_GROSS_EXPOSURE = 200.0       # グロスエクスポージャー上限（%）
    DEFAULT_NET_EXPOSURE_RANGE = (-50.0, 150.0)  # ネットエクスポージャー範囲（%）
    DEFAULT_VAR_LOOKBACK = 252               # VaR計算のルックバック日数

    def __init__(self, initial_capital: float = 10_000_000,
                 portfolio_dd_limit: float = None,
                 data_dir: str = None):
        self.initial_capital = initial_capital
        self.portfolio_dd_limit = portfolio_dd_limit or self.DEFAULT_PORTFOLIO_DD_LIMIT
        self.pods: dict[str, Pod] = {}
        self.nav = initial_capital
        self.peak_nav = initial_capital
        self.nav_history: list[dict] = []
        self.alerts: list[dict] = []
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data" / "portfolio"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ─── ポッド管理 ───

    def add_pod(self, name: str, allocation_pct: float, dd_limit_pct: float,
                description: str = "") -> Pod:
        """ポッドを追加"""
        initial_nav = self.initial_capital * allocation_pct / 100
        pod = Pod(name, allocation_pct, dd_limit_pct, initial_nav, description)
        self.pods[name] = pod
        return pod

    def get_pod(self, name: str) -> Optional[Pod]:
        return self.pods.get(name)

    def update_pod_nav(self, pod_name: str, new_nav: float, date: Optional[str] = None):
        """特定ポッドのNAVを更新し、ポートフォリオ全体を再計算"""
        pod = self.pods.get(pod_name)
        if pod is None:
            raise ValueError(f"Pod '{pod_name}' not found")
        pod.update_nav(new_nav, date)
        self._recalculate_portfolio_nav(date)

    def _recalculate_portfolio_nav(self, date: Optional[str] = None):
        """ポートフォリオ全体のNAVを再計算"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        self.nav = sum(p.nav for p in self.pods.values())
        if self.nav > self.peak_nav:
            self.peak_nav = self.nav
        self.nav_history.append({"date": date, "nav": self.nav})

    # ─── リスクメトリクス計算 ───

    @property
    def drawdown_pct(self) -> float:
        if self.peak_nav <= 0:
            return 0.0
        return (self.nav / self.peak_nav - 1) * 100

    def compute_risk_metrics(self) -> dict:
        """ポートフォリオ全体のリスクメトリクスを計算"""
        metrics = {
            "nav": round(self.nav, 0),
            "initial_capital": self.initial_capital,
            "total_pnl": round(self.nav - self.initial_capital, 0),
            "total_pnl_pct": round((self.nav / self.initial_capital - 1) * 100, 2),
            "peak_nav": round(self.peak_nav, 0),
            "current_dd_pct": round(self.drawdown_pct, 2),
            "dd_limit_pct": self.portfolio_dd_limit,
        }

        # 日次リターンの集約
        all_returns = self._get_portfolio_daily_returns()
        if len(all_returns) >= 5:
            arr = np.array(all_returns)
            metrics["volatility_daily"] = round(float(np.std(arr)), 6)
            metrics["volatility_ann"] = round(float(np.std(arr) * np.sqrt(252)), 4)
            mean_ret = float(np.mean(arr))
            std_ret = float(np.std(arr))
            metrics["sharpe_ann"] = round(mean_ret / std_ret * np.sqrt(252), 3) if std_ret > 0 else 0

            # VaR (Historical Simulation)
            var_95 = float(np.percentile(arr, 5))
            var_99 = float(np.percentile(arr, 1))
            metrics["var_95_1d_pct"] = round(var_95 * 100, 3)
            metrics["var_99_1d_pct"] = round(var_99 * 100, 3)
            metrics["var_95_1d_jpy"] = round(var_95 * self.nav, 0)
            metrics["var_99_1d_jpy"] = round(var_99 * self.nav, 0)

            # Max DD from returns
            cum = np.cumprod(1 + arr)
            peak = np.maximum.accumulate(cum)
            dd_series = (cum - peak) / peak
            metrics["max_dd_pct"] = round(float(np.min(dd_series)) * 100, 2)
        else:
            metrics["volatility_daily"] = 0
            metrics["volatility_ann"] = 0
            metrics["sharpe_ann"] = 0
            metrics["var_95_1d_pct"] = 0
            metrics["var_99_1d_pct"] = 0
            metrics["var_95_1d_jpy"] = 0
            metrics["var_99_1d_jpy"] = 0
            metrics["max_dd_pct"] = round(self.drawdown_pct, 2)

        # ポッド別サマリー
        metrics["pods"] = {name: pod.to_dict() for name, pod in self.pods.items()}

        # エクスポージャー
        total_long = sum(p.long_exposure for p in self.pods.values())
        total_short = sum(p.short_exposure for p in self.pods.values())
        metrics["total_long_exposure"] = round(total_long, 0)
        metrics["total_short_exposure"] = round(total_short, 0)
        metrics["net_exposure"] = round(total_long - total_short, 0)
        metrics["gross_exposure"] = round(total_long + total_short, 0)
        metrics["gross_exposure_pct"] = round((total_long + total_short) / max(self.nav, 1) * 100, 1)
        metrics["net_exposure_pct"] = round((total_long - total_short) / max(self.nav, 1) * 100, 1)

        return metrics

    def _get_portfolio_daily_returns(self) -> list[float]:
        """NAV履歴から日次リターンを計算"""
        if len(self.nav_history) < 2:
            return []
        returns = []
        for i in range(1, len(self.nav_history)):
            prev = self.nav_history[i - 1]["nav"]
            curr = self.nav_history[i]["nav"]
            if prev > 0:
                returns.append(curr / prev - 1)
        return returns

    # ─── 相関マトリックス ───

    def compute_correlation_matrix(self, lookback: int = 30) -> dict:
        """ポッド間の相関マトリックスを計算"""
        pod_names = list(self.pods.keys())
        if len(pod_names) < 2:
            return {"matrix": {}, "alerts": []}

        # 各ポッドの日次リターンをDataFrameに
        returns_dict = {}
        for name, pod in self.pods.items():
            rets = pod.daily_returns[-lookback:] if len(pod.daily_returns) >= lookback else pod.daily_returns
            if len(rets) >= 5:
                returns_dict[name] = rets

        if len(returns_dict) < 2:
            return {"matrix": {}, "alerts": []}

        # 長さを揃える
        min_len = min(len(v) for v in returns_dict.values())
        df = pd.DataFrame({k: v[-min_len:] for k, v in returns_dict.items()})
        corr = df.corr()

        # マトリックスをdict化
        matrix = {}
        alerts = []
        for i, n1 in enumerate(corr.columns):
            for n2 in corr.columns[i + 1:]:
                key = f"{n1}_{n2}"
                val = float(corr.loc[n1, n2])
                if math.isnan(val):
                    val = 0.0
                matrix[key] = round(val, 3)
                if abs(val) > 0.7:
                    alerts.append({
                        "level": "warning",
                        "message": f"ポッド相関が高い: {n1} ↔ {n2} = {val:.3f}",
                        "pair": key,
                        "correlation": val,
                    })

        return {"matrix": matrix, "lookback_days": min_len, "alerts": alerts}

    # ─── アラートチェック ───

    def check_all_limits(self) -> list[dict]:
        """全リスク制限をチェックしてアラートリストを返す"""
        alerts = []

        # 1. ポートフォリオ全体DD
        dd = self.drawdown_pct
        if dd <= -self.portfolio_dd_limit:
            alerts.append({
                "level": "critical",
                "message": f"ポートフォリオDD {dd:.1f}% ≤ -{self.portfolio_dd_limit:.1f}% → 全ポッド停止",
                "action": "halt_all",
            })
        elif dd <= -self.portfolio_dd_limit * 0.8:
            alerts.append({
                "level": "warning",
                "message": f"ポートフォリオDD {dd:.1f}%: DDリミット（-{self.portfolio_dd_limit:.1f}%）の80%に接近",
                "action": "reduce_exposure",
            })

        # 2. 各ポッドのDD
        for pod in self.pods.values():
            alerts.extend(pod.check_dd_limits())

        # 3. グロスエクスポージャー
        gross_pct = sum(p.gross_exposure for p in self.pods.values()) / max(self.nav, 1) * 100
        if gross_pct > self.DEFAULT_MAX_GROSS_EXPOSURE:
            alerts.append({
                "level": "warning",
                "message": f"グロスエクスポージャー {gross_pct:.1f}% > {self.DEFAULT_MAX_GROSS_EXPOSURE:.0f}%",
                "action": "reduce_gross",
            })

        # 4. ネットエクスポージャー
        net_pct = sum(p.net_exposure for p in self.pods.values()) / max(self.nav, 1) * 100
        lo, hi = self.DEFAULT_NET_EXPOSURE_RANGE
        if net_pct < lo or net_pct > hi:
            alerts.append({
                "level": "warning",
                "message": f"ネットエクスポージャー {net_pct:.1f}% 範囲外 [{lo:.0f}%, {hi:.0f}%]",
                "action": "rebalance_net",
            })

        # 5. 相関チェック
        corr_result = self.compute_correlation_matrix()
        alerts.extend(corr_result.get("alerts", []))

        self.alerts = alerts
        return alerts

    # ─── ポジションサイズチェック ───

    def check_position_size(self, code: str, side: str, amount: float,
                            sector: Optional[str] = None) -> dict:
        """
        新規ポジションのサイズが制限内かチェック。
        Returns: {"allowed": bool, "reason": str, "max_amount": float}
        """
        nav = max(self.nav, 1)

        # 1銘柄上限
        max_single = nav * self.DEFAULT_MAX_SINGLE_POSITION / 100
        existing = sum(
            p["quantity"] * p.get("current_price", p["entry_price"])
            for pod in self.pods.values()
            for p in pod.positions
            if p["code"] == code
        )
        total = existing + amount
        if total > max_single:
            return {
                "allowed": False,
                "reason": f"1銘柄上限超過: {code} 合計{total:,.0f}円 > {max_single:,.0f}円",
                "max_amount": max(0, max_single - existing),
            }

        # 1セクター上限
        if sector:
            max_sector = nav * self.DEFAULT_MAX_SECTOR_EXPOSURE / 100
            sector_total = sum(
                p["quantity"] * p.get("current_price", p["entry_price"])
                for pod in self.pods.values()
                for p in pod.positions
                if p.get("sector") == sector
            )
            if sector_total + amount > max_sector:
                return {
                    "allowed": False,
                    "reason": f"セクター上限超過: {sector} 合計{sector_total + amount:,.0f}円 > {max_sector:,.0f}円",
                    "max_amount": max(0, max_sector - sector_total),
                }

        return {"allowed": True, "reason": "OK", "max_amount": amount}

    # ─── 資金配分の動的調整提案 ───

    def suggest_rebalance(self) -> list[dict]:
        """各ポッドの目標配分と実際の乖離を計算し、リバランス提案を返す"""
        suggestions = []
        nav = max(self.nav, 1)

        for name, pod in self.pods.items():
            actual_pct = pod.nav / nav * 100
            target_pct = pod.allocation_pct
            diff = actual_pct - target_pct

            if abs(diff) > 3.0:  # 3%以上の乖離
                direction = "売り" if diff > 0 else "買い"
                amount = abs(diff) / 100 * nav
                suggestions.append({
                    "pod": name,
                    "target_pct": target_pct,
                    "actual_pct": round(actual_pct, 1),
                    "diff_pct": round(diff, 1),
                    "action": direction,
                    "amount": round(amount, 0),
                })

        return suggestions

    # ─── 永続化 ───

    def save_state(self, filepath: Optional[str] = None):
        """状態をJSONに保存"""
        if filepath is None:
            filepath = self.data_dir / "portfolio_state.json"
        state = {
            "initial_capital": self.initial_capital,
            "nav": self.nav,
            "peak_nav": self.peak_nav,
            "portfolio_dd_limit": self.portfolio_dd_limit,
            "nav_history": self.nav_history[-365:],  # 直近1年分
            "pods": {},
            "saved_at": datetime.now().isoformat(),
        }
        for name, pod in self.pods.items():
            state["pods"][name] = {
                "allocation_pct": pod.allocation_pct,
                "dd_limit_pct": pod.dd_limit_pct,
                "nav": pod.nav,
                "peak_nav": pod.peak_nav,
                "active": pod.active,
                "description": pod.description,
                "positions": pod.positions,
                "nav_history": pod.nav_history[-365:],
                "daily_returns": pod.daily_returns[-365:],
            }
        Path(filepath).write_text(json.dumps(_sanitize(state), ensure_ascii=False, indent=2))

    def load_state(self, filepath: Optional[str] = None):
        """JSONから状態を復元"""
        if filepath is None:
            filepath = self.data_dir / "portfolio_state.json"
        p = Path(filepath)
        if not p.exists():
            return False
        state = json.loads(p.read_text())
        self.initial_capital = state.get("initial_capital", self.initial_capital)
        self.nav = state.get("nav", self.initial_capital)
        self.peak_nav = state.get("peak_nav", self.nav)
        self.portfolio_dd_limit = state.get("portfolio_dd_limit", self.DEFAULT_PORTFOLIO_DD_LIMIT)
        self.nav_history = state.get("nav_history", [])

        for name, pod_state in state.get("pods", {}).items():
            pod = Pod(
                name=name,
                allocation_pct=pod_state.get("allocation_pct", 0),
                dd_limit_pct=pod_state.get("dd_limit_pct", 15),
                initial_nav=pod_state.get("nav", 0),
                description=pod_state.get("description", ""),
            )
            pod.peak_nav = pod_state.get("peak_nav", pod.nav)
            pod.active = pod_state.get("active", True)
            pod.positions = pod_state.get("positions", [])
            pod.nav_history = pod_state.get("nav_history", [])
            pod.daily_returns = pod_state.get("daily_returns", [])
            self.pods[name] = pod

        return True

    # ─── ユーティリティ ───

    def get_all_positions(self) -> list[dict]:
        """全ポッドのポジションを統合して返す"""
        positions = []
        for name, pod in self.pods.items():
            for p in pod.positions:
                positions.append({**p, "pod": name})
        return positions

    def summary(self) -> dict:
        """ダッシュボード用サマリー"""
        metrics = self.compute_risk_metrics()
        alerts = self.check_all_limits()
        corr = self.compute_correlation_matrix()
        rebalance = self.suggest_rebalance()

        return {
            "portfolio": metrics,
            "correlation": corr,
            "alerts": alerts,
            "rebalance_suggestions": rebalance,
            "as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }


# ─── ヘルパー ───

def _sanitize(obj):
    """NaN/Inf/boolをJSONシリアライズ可能に"""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(i) for i in obj]
    if isinstance(obj, np.ndarray):
        return [_sanitize(i) for i in obj.tolist()]
    return obj


# ─── デフォルト構成でエンジンを生成 ───

def create_default_engine(initial_capital: float = 10_000_000) -> PortfolioEngine:
    """MULTISTRAT_DESIGN.md の設計に基づくデフォルト構成"""
    engine = PortfolioEngine(initial_capital=initial_capital, portfolio_dd_limit=20.0)
    engine.add_pod("momentum", allocation_pct=50, dd_limit_pct=15,
                   description="クロスセクショナルモメンタム（ロング/ショート）")
    engine.add_pod("mean_reversion", allocation_pct=20, dd_limit_pct=10,
                   description="短期過売り反発（RSI+BB）")
    engine.add_pod("event_driven", allocation_pct=15, dd_limit_pct=12,
                   description="決算サプライズ・信用倍率変動")
    engine.add_pod("macro_overlay", allocation_pct=15, dd_limit_pct=8,
                   description="レジーム判定に基づくヘッジ")
    return engine


if __name__ == "__main__":
    # デモ
    engine = create_default_engine()
    print("=== ポートフォリオエンジン初期化 ===")
    print(f"初期資本: {engine.initial_capital:,.0f}円")
    for name, pod in engine.pods.items():
        print(f"  {name}: {pod.nav:,.0f}円 (配分{pod.allocation_pct}%, DDリミット{pod.dd_limit_pct}%)")

    # シミュレーション: momentumポッドが少し上昇
    engine.update_pod_nav("momentum", 5_200_000)
    engine.update_pod_nav("mean_reversion", 1_980_000)
    engine.update_pod_nav("event_driven", 1_520_000)
    engine.update_pod_nav("macro_overlay", 1_480_000)

    metrics = engine.compute_risk_metrics()
    print(f"\nNAV: {metrics['nav']:,.0f}円 (PnL: {metrics['total_pnl']:+,.0f}円, {metrics['total_pnl_pct']:+.2f}%)")
    print(f"DD: {metrics['current_dd_pct']:.2f}%")

    alerts = engine.check_all_limits()
    if alerts:
        print(f"\nアラート ({len(alerts)}件):")
        for a in alerts:
            print(f"  [{a['level']}] {a['message']}")
    else:
        print("\nアラート: なし ✅")

    # 保存テスト
    engine.save_state()
    print(f"\n状態保存完了: {engine.data_dir / 'portfolio_state.json'}")
