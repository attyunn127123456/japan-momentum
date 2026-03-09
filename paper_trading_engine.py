"""
ペーパートレーディングエンジン

リアルな取引コスト（スリッページ・信用金利）を含むペーパートレードシミュレーター。
既存の run_paper_trade_record.py を拡張し、マルチストラテジー対応。

Usage:
    from paper_trading_engine import PaperTradingEngine
    engine = PaperTradingEngine(initial_capital=10_000_000)
    engine.open_position("momentum", "72030", "long", quantity=100, price=2500.0)
    engine.update_prices({"72030": 2600.0})
    engine.close_position("72030", price=2600.0, reason="signal_exit")
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

# ─── 取引コスト設定 ───

class TradingCosts:
    """取引コストモデル"""
    SLIPPAGE_ONE_WAY = 0.0015          # 片道スリッページ 0.15%
    COMMISSION = 0.0                    # 委託手数料 0%（ゼロコース）
    MARGIN_INTEREST_LONG = 0.028        # 信用金利（ロング）年2.80%
    MARGIN_INTEREST_SHORT = 0.011       # 貸株料（ショート）年1.10%
    REVERSE_DAILY_CHARGE = 0.0          # 逆日歩（簡易モデル: 0）
    MIN_UNIT = 100                      # 最低取引単位（単元株）

    @classmethod
    def entry_price_long(cls, market_price: float) -> float:
        """ロングエントリー価格（スリッページ上乗せ）"""
        return market_price * (1 + cls.SLIPPAGE_ONE_WAY)

    @classmethod
    def entry_price_short(cls, market_price: float) -> float:
        """ショートエントリー価格（スリッページ差引）"""
        return market_price * (1 - cls.SLIPPAGE_ONE_WAY)

    @classmethod
    def exit_price_long(cls, market_price: float) -> float:
        """ロングエグジット価格（スリッページ差引）"""
        return market_price * (1 - cls.SLIPPAGE_ONE_WAY)

    @classmethod
    def exit_price_short(cls, market_price: float) -> float:
        """ショートエグジット価格（スリッページ上乗せ）"""
        return market_price * (1 + cls.SLIPPAGE_ONE_WAY)

    @classmethod
    def daily_margin_cost(cls, position_value: float, side: str) -> float:
        """日次の信用金利コスト"""
        if side == "long":
            return position_value * cls.MARGIN_INTEREST_LONG / 365
        else:
            return position_value * cls.MARGIN_INTEREST_SHORT / 365

    @classmethod
    def round_to_unit(cls, quantity: int) -> int:
        """単元株に丸める"""
        return (quantity // cls.MIN_UNIT) * cls.MIN_UNIT


# ─── ポジション ───

class Position:
    """個別ポジション"""

    def __init__(self, pod: str, code: str, name: str, side: str,
                 quantity: int, entry_price: float, entry_date: str,
                 sector: str = "", signal_score: float = 0.0):
        self.pod = pod
        self.code = code
        self.name = name
        self.side = side  # "long" or "short"
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.sector = sector
        self.signal_score = signal_score

        self.current_price = entry_price
        self.trailing_high = entry_price  # ロング用: 最高値
        self.trailing_low = entry_price   # ショート用: 最安値
        self.cost_accrued = 0.0           # 累計コスト（金利等）
        self.days_held = 0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def entry_value(self) -> float:
        return self.quantity * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        """含み損益（コスト差引前）"""
        if self.side == "long":
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_net(self) -> float:
        """含み損益（コスト差引後）"""
        return self.unrealized_pnl - self.cost_accrued

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_value == 0:
            return 0.0
        return self.unrealized_pnl / self.entry_value * 100

    def update_price(self, new_price: float):
        """価格更新"""
        self.current_price = new_price
        if new_price > self.trailing_high:
            self.trailing_high = new_price
        if new_price < self.trailing_low:
            self.trailing_low = new_price

    def accrue_daily_cost(self):
        """日次の信用金利を累積"""
        cost = TradingCosts.daily_margin_cost(self.market_value, self.side)
        self.cost_accrued += cost
        self.days_held += 1
        return cost

    def to_dict(self) -> dict:
        return {
            "pod": self.pod,
            "code": self.code,
            "name": self.name,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": round(self.entry_price, 2),
            "entry_date": self.entry_date,
            "current_price": round(self.current_price, 2),
            "market_value": round(self.market_value, 0),
            "unrealized_pnl": round(self.unrealized_pnl, 0),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 2),
            "cost_accrued": round(self.cost_accrued, 0),
            "unrealized_pnl_net": round(self.unrealized_pnl_net, 0),
            "days_held": self.days_held,
            "trailing_high": round(self.trailing_high, 2),
            "trailing_low": round(self.trailing_low, 2),
            "sector": self.sector,
            "signal_score": round(self.signal_score, 4),
        }


# ─── ペーパートレーディングエンジン ───

class PaperTradingEngine:
    """ペーパートレーディングエンジン"""

    def __init__(self, initial_capital: float = 10_000_000,
                 data_dir: Optional[str] = None):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: list[Position] = []
        self.trade_history: list[dict] = []
        self.daily_snapshots: list[dict] = []
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data" / "paper_trades"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ─── NAV計算 ───

    @property
    def nav(self) -> float:
        """Net Asset Value = 現金 + ロングポジション時価 - ショートポジション含み損益調整"""
        pos_value = sum(p.unrealized_pnl_net for p in self.positions)
        return self.cash + pos_value

    @property
    def total_pnl(self) -> float:
        return self.nav - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        return (self.nav / self.initial_capital - 1) * 100

    @property
    def long_positions(self) -> list[Position]:
        return [p for p in self.positions if p.side == "long"]

    @property
    def short_positions(self) -> list[Position]:
        return [p for p in self.positions if p.side == "short"]

    # ─── ポジション操作 ───

    def open_position(self, pod: str, code: str, side: str, quantity: int,
                      market_price: float, name: str = "", sector: str = "",
                      signal_score: float = 0.0,
                      date: Optional[str] = None) -> Optional[Position]:
        """
        新規ポジションを開く。
        スリッページを考慮した約定価格で記録。
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # 単元株に丸める
        quantity = TradingCosts.round_to_unit(quantity)
        if quantity <= 0:
            return None

        # 約定価格（スリッページ込み）
        if side == "long":
            entry_price = TradingCosts.entry_price_long(market_price)
        else:
            entry_price = TradingCosts.entry_price_short(market_price)

        # 必要資金チェック（ロングは全額、ショートは保証金30%）
        if side == "long":
            required = entry_price * quantity
        else:
            required = entry_price * quantity * 0.30  # 保証金30%
        if required > self.cash:
            # 資金不足: 買える分だけ買う
            if side == "long":
                max_qty = int(self.cash / entry_price)
            else:
                max_qty = int(self.cash / (entry_price * 0.30))
            quantity = TradingCosts.round_to_unit(max_qty)
            if quantity <= 0:
                return None

        pos = Position(
            pod=pod, code=code, name=name, side=side,
            quantity=quantity, entry_price=entry_price,
            entry_date=date, sector=sector, signal_score=signal_score,
        )

        # 現金を減らす
        if side == "long":
            self.cash -= entry_price * quantity
        else:
            self.cash -= entry_price * quantity * 0.30  # 保証金拘束

        self.positions.append(pos)

        # 取引ログ
        self.trade_history.append({
            "type": "open",
            "pod": pod,
            "code": code,
            "name": name,
            "side": side,
            "quantity": quantity,
            "price": round(entry_price, 2),
            "market_price": round(market_price, 2),
            "slippage": round(abs(entry_price - market_price), 2),
            "date": date,
            "signal_score": round(signal_score, 4),
        })

        return pos

    def close_position(self, code: str, market_price: float,
                       reason: str = "signal_exit",
                       date: Optional[str] = None,
                       pod: Optional[str] = None) -> Optional[dict]:
        """
        ポジションをクローズ。PnLを確定。
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # 対象ポジションを探す
        pos = None
        for p in self.positions:
            if p.code == code and (pod is None or p.pod == pod):
                pos = p
                break

        if pos is None:
            return None

        # 約定価格（スリッページ込み）
        if pos.side == "long":
            exit_price = TradingCosts.exit_price_long(market_price)
        else:
            exit_price = TradingCosts.exit_price_short(market_price)

        # PnL計算
        if pos.side == "long":
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity

        net_pnl = gross_pnl - pos.cost_accrued
        pnl_pct = gross_pnl / (pos.entry_price * pos.quantity) * 100

        # 現金を戻す
        if pos.side == "long":
            self.cash += exit_price * pos.quantity
        else:
            # 保証金 + PnL を返却
            self.cash += pos.entry_price * pos.quantity * 0.30 + gross_pnl

        trade_record = {
            "type": "close",
            "pod": pos.pod,
            "code": pos.code,
            "name": pos.name,
            "side": pos.side,
            "quantity": pos.quantity,
            "entry_price": round(pos.entry_price, 2),
            "exit_price": round(exit_price, 2),
            "market_price": round(market_price, 2),
            "entry_date": pos.entry_date,
            "exit_date": date,
            "days_held": pos.days_held,
            "gross_pnl": round(gross_pnl, 0),
            "cost_accrued": round(pos.cost_accrued, 0),
            "net_pnl": round(net_pnl, 0),
            "pnl_pct": round(pnl_pct, 2),
            "reason": reason,
        }

        self.trade_history.append(trade_record)
        self.positions.remove(pos)
        return trade_record

    def close_all_positions(self, prices: dict, reason: str = "rebalance",
                            date: Optional[str] = None) -> list[dict]:
        """全ポジションをクローズ"""
        results = []
        for pos in list(self.positions):
            price = prices.get(pos.code, pos.current_price)
            result = self.close_position(pos.code, price, reason=reason, date=date, pod=pos.pod)
            if result:
                results.append(result)
        return results

    # ─── 日次更新 ───

    def update_prices(self, prices: dict, date: Optional[str] = None):
        """全ポジションの現在価格を更新"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        for pos in self.positions:
            if pos.code in prices:
                pos.update_price(prices[pos.code])

    def accrue_daily_costs(self):
        """全ポジションの日次コスト（信用金利）を計算"""
        total_cost = 0.0
        for pos in self.positions:
            cost = pos.accrue_daily_cost()
            total_cost += cost
        return total_cost

    def take_daily_snapshot(self, date: Optional[str] = None):
        """日次のNAVスナップショットを記録"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        snapshot = {
            "date": date,
            "nav": round(self.nav, 0),
            "cash": round(self.cash, 0),
            "positions_count": len(self.positions),
            "long_count": len(self.long_positions),
            "short_count": len(self.short_positions),
            "long_value": round(sum(p.market_value for p in self.long_positions), 0),
            "short_value": round(sum(p.market_value for p in self.short_positions), 0),
            "total_pnl": round(self.total_pnl, 0),
            "total_pnl_pct": round(self.total_pnl_pct, 2),
            "daily_cost": round(sum(
                TradingCosts.daily_margin_cost(p.market_value, p.side)
                for p in self.positions
            ), 0),
        }
        self.daily_snapshots.append(snapshot)
        return snapshot

    # ─── シグナルベースのリバランス ───

    def rebalance_from_signals(self, signals: list[dict], pod: str,
                               capital_for_pod: float,
                               current_prices: dict,
                               date: Optional[str] = None) -> dict:
        """
        シグナルリストに基づいてリバランスを実行。

        signals: [{"code": "72030", "name": "トヨタ", "side": "long", "score": 0.85, "sector": "..."}, ...]
        capital_for_pod: このポッドに割り当てる資金
        current_prices: {code: price}

        Returns: {"opened": [...], "closed": [...], "held": [...]}
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # 現在のポッドのポジション
        current_codes = {p.code for p in self.positions if p.pod == pod}
        signal_codes = {s["code"] for s in signals}

        # 1. クローズ対象（シグナルに無いポジション）
        closed = []
        for pos in list(self.positions):
            if pos.pod == pod and pos.code not in signal_codes:
                price = current_prices.get(pos.code, pos.current_price)
                result = self.close_position(pos.code, price, reason="signal_exit", date=date, pod=pod)
                if result:
                    closed.append(result)

        # 2. 保持（既にポジションあり）
        held = [s for s in signals if s["code"] in current_codes]

        # 3. 新規オープン
        new_signals = [s for s in signals if s["code"] not in current_codes]
        opened = []
        if new_signals:
            # 等配分
            per_position = capital_for_pod / len(signals) if len(signals) > 0 else 0
            for sig in new_signals:
                code = sig["code"]
                price = current_prices.get(code)
                if price is None or price <= 0:
                    continue
                side = sig.get("side", "long")
                quantity = int(per_position / price)
                quantity = TradingCosts.round_to_unit(quantity)
                if quantity <= 0:
                    continue
                pos = self.open_position(
                    pod=pod, code=code, side=side, quantity=quantity,
                    market_price=price, name=sig.get("name", ""),
                    sector=sig.get("sector", ""),
                    signal_score=sig.get("score", 0.0),
                    date=date,
                )
                if pos:
                    opened.append(pos.to_dict())

        return {"opened": opened, "closed": closed, "held": held}

    # ─── パフォーマンス集計 ───

    def performance_summary(self) -> dict:
        """パフォーマンスサマリー"""
        closed_trades = [t for t in self.trade_history if t["type"] == "close"]
        if not closed_trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_pnl_pct": 0,
                "total_net_pnl": 0,
                "best_trade": None,
                "worst_trade": None,
                "avg_days_held": 0,
            }

        pnls = [t["pnl_pct"] for t in closed_trades]
        net_pnls = [t["net_pnl"] for t in closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return {
            "total_trades": len(closed_trades),
            "win_rate": round(len(wins) / len(closed_trades) * 100, 1),
            "avg_pnl_pct": round(np.mean(pnls), 2),
            "total_net_pnl": round(sum(net_pnls), 0),
            "total_gross_pnl": round(sum(t["gross_pnl"] for t in closed_trades), 0),
            "total_costs": round(sum(t["cost_accrued"] for t in closed_trades), 0),
            "best_trade": max(closed_trades, key=lambda t: t["pnl_pct"]) if closed_trades else None,
            "worst_trade": min(closed_trades, key=lambda t: t["pnl_pct"]) if closed_trades else None,
            "avg_days_held": round(np.mean([t["days_held"] for t in closed_trades]), 1),
            "profit_factor": round(
                sum(t["net_pnl"] for t in closed_trades if t["net_pnl"] > 0) /
                max(abs(sum(t["net_pnl"] for t in closed_trades if t["net_pnl"] < 0)), 1),
                2
            ),
        }

    def equity_curve(self) -> list[dict]:
        """日次NAV推移"""
        return self.daily_snapshots

    # ─── 永続化 ───

    def save_state(self, filepath: Optional[str] = None):
        """状態をJSONに保存"""
        if filepath is None:
            filepath = self.data_dir / "paper_trade_state.json"
        state = {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions": [p.to_dict() for p in self.positions],
            "trade_history": self.trade_history[-1000:],  # 直近1000件
            "daily_snapshots": self.daily_snapshots[-365:],
            "saved_at": datetime.now().isoformat(),
        }
        Path(filepath).write_text(json.dumps(_sanitize(state), ensure_ascii=False, indent=2))

    def load_state(self, filepath: Optional[str] = None) -> bool:
        """JSONから状態を復元"""
        if filepath is None:
            filepath = self.data_dir / "paper_trade_state.json"
        p = Path(filepath)
        if not p.exists():
            return False
        state = json.loads(p.read_text())
        self.initial_capital = state.get("initial_capital", self.initial_capital)
        self.cash = state.get("cash", self.initial_capital)
        self.trade_history = state.get("trade_history", [])
        self.daily_snapshots = state.get("daily_snapshots", [])

        # ポジション復元
        self.positions = []
        for pd_data in state.get("positions", []):
            pos = Position(
                pod=pd_data.get("pod", ""),
                code=pd_data["code"],
                name=pd_data.get("name", ""),
                side=pd_data["side"],
                quantity=pd_data["quantity"],
                entry_price=pd_data["entry_price"],
                entry_date=pd_data.get("entry_date", ""),
                sector=pd_data.get("sector", ""),
                signal_score=pd_data.get("signal_score", 0),
            )
            pos.current_price = pd_data.get("current_price", pos.entry_price)
            pos.trailing_high = pd_data.get("trailing_high", pos.entry_price)
            pos.trailing_low = pd_data.get("trailing_low", pos.entry_price)
            pos.cost_accrued = pd_data.get("cost_accrued", 0)
            pos.days_held = pd_data.get("days_held", 0)
            self.positions.append(pos)

        return True

    # ─── ダッシュボード用エクスポート ───

    def dashboard_data(self) -> dict:
        """ダッシュボードAPI用データ"""
        return {
            "nav": round(self.nav, 0),
            "cash": round(self.cash, 0),
            "initial_capital": self.initial_capital,
            "total_pnl": round(self.total_pnl, 0),
            "total_pnl_pct": round(self.total_pnl_pct, 2),
            "positions": [p.to_dict() for p in self.positions],
            "long_count": len(self.long_positions),
            "short_count": len(self.short_positions),
            "long_value": round(sum(p.market_value for p in self.long_positions), 0),
            "short_value": round(sum(p.market_value for p in self.short_positions), 0),
            "performance": self.performance_summary(),
            "recent_trades": self.trade_history[-20:],
            "equity_curve": self.daily_snapshots[-90:],
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
    return obj


# ─── 既存 run_paper_trade_record.py 互換: シグナルから自動実行 ───

def run_paper_trade_from_signals():
    """
    daily_signal_output.json のシグナルを読み、ペーパートレードを自動実行する。
    既存の run_paper_trade_record.py の上位互換。
    """
    base = Path(__file__).parent

    # シグナル読み込み
    signal_path = base / "backtest" / "daily_signal_output.json"
    if not signal_path.exists():
        print("daily_signal_output.json が見つかりません")
        return None

    sig = json.loads(signal_path.read_text())
    recommended = sig.get("recommended", [])
    if not recommended:
        print("推奨銘柄なし")
        return None

    # エンジン初期化 or 復元
    engine = PaperTradingEngine(initial_capital=10_000_000)
    engine.load_state()

    # 現在価格の取得
    all_scores = sig.get("all_scores", sig.get("top20", []))
    current_prices = {}
    for s in all_scores:
        if "price" in s and s["price"]:
            current_prices[s["code"]] = s["price"]

    # 価格更新
    engine.update_prices(current_prices)

    # 日次コスト計算
    engine.accrue_daily_costs()

    # シグナルからリバランス
    signals = [
        {"code": r["code"], "name": r.get("name", ""), "side": "long", "score": r.get("score", 0)}
        for r in recommended
    ]

    # ポッド割り当て資金（全額momentumポッド）
    capital = engine.nav * 0.90  # 10%は現金保持

    result = engine.rebalance_from_signals(
        signals=signals,
        pod="momentum",
        capital_for_pod=capital,
        current_prices=current_prices,
    )

    # スナップショット
    engine.take_daily_snapshot()

    # 保存
    engine.save_state()

    # サマリー出力
    print(f"NAV: {engine.nav:,.0f}円 (PnL: {engine.total_pnl:+,.0f}円, {engine.total_pnl_pct:+.2f}%)")
    print(f"ポジション: ロング{len(engine.long_positions)}件, ショート{len(engine.short_positions)}件")
    print(f"新規: {len(result['opened'])}件, 決済: {len(result['closed'])}件, 保持: {len(result['held'])}件")

    perf = engine.performance_summary()
    if perf["total_trades"] > 0:
        print(f"勝率: {perf['win_rate']}%, 平均PnL: {perf['avg_pnl_pct']}%, PF: {perf['profit_factor']}")

    return engine.dashboard_data()


if __name__ == "__main__":
    run_paper_trade_from_signals()
