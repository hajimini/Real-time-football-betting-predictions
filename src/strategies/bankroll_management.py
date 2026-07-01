"""
资金管理系统
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from loguru import logger


@dataclass
class Transaction:
    """交易记录"""
    id: int
    type: str  # "bet", "win", "loss", "deposit", "withdrawal"
    amount: float
    balance_after: float
    description: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    bet_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "amount": self.amount,
            "balance_after": self.balance_after,
            "description": self.description,
            "timestamp": self.timestamp,
            "bet_id": self.bet_id
        }


class BankrollManager:
    """资金管理器"""
    
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.transactions: List[Transaction] = []
        self.active_bets: List[Dict] = []
        self.transaction_counter = 0
        
        # 统计信息
        self.stats = {
            "total_bets": 0,
            "wins": 0,
            "losses": 0,
            "total_won": 0.0,
            "total_lost": 0.0,
            "roi": 0.0,
            "max_drawdown": 0.0,
            "peak_balance": initial_balance
        }
        
        logger.info(f"资金管理初始化，初始资金：{initial_balance}")
    
    def place_bet(self, bet: Dict[str, Any]) -> bool:
        """下注"""
        stake = bet.get("stake", 0)
        
        if stake <= 0:
            logger.error("投注金额必须大于 0")
            return False
        
        if stake > self.current_balance:
            logger.error(f"余额不足：当前 {self.current_balance}, 需要 {stake}")
            return False
        
        # 检查风险限制（单笔不超过 5%）
        if stake > self.current_balance * 0.05:
            logger.warning(f"警告：单笔投注超过 5%，建议降低")
        
        # 扣除资金
        self.current_balance -= stake
        self._add_transaction(
            type="bet",
            amount=-stake,
            description=f"投注：{bet.get('selection')} @ {bet.get('odds')}",
            bet_id=len(self.active_bets)
        )
        
        # 记录活跃投注
        bet["stake"] = stake
        bet["placed_at"] = datetime.now().isoformat()
        bet["status"] = "active"
        self.active_bets.append(bet)
        
        self.stats["total_bets"] += 1
        
        logger.info(f"下注成功：{stake} @ {bet.get('odds')} on {bet.get('selection')}")
        return True
    
    def settle_bet(self, bet_index: int, won: bool) -> float:
        """结算投注"""
        if bet_index >= len(self.active_bets):
            logger.error("无效的投注索引")
            return 0
        
        bet = self.active_bets[bet_index]
        stake = bet.get("stake", 0)
        odds = bet.get("odds", 0)
        
        if won:
            winnings = stake * odds
            self.current_balance += winnings
            self.stats["wins"] += 1
            self.stats["total_won"] += winnings
            
            self._add_transaction(
                type="win",
                amount=winnings,
                description=f"赢利：{bet.get('selection')} @ {odds}"
            )
            logger.info(f"投注赢利：+{winnings:.2f}")
        else:
            self.stats["losses"] += 1
            self.stats["total_lost"] += stake
            
            self._add_transaction(
                type="loss",
                amount=0,
                description=f"损失：{bet.get('selection')}"
            )
            logger.info(f"投注损失：-{stake:.2f}")
        
        # 更新状态
        bet["status"] = "settled"
        bet["won"] = won
        
        # 更新峰值和回撤
        if self.current_balance > self.stats["peak_balance"]:
            self.stats["peak_balance"] = self.current_balance
        
        drawdown = (self.stats["peak_balance"] - self.current_balance) / self.stats["peak_balance"]
        if drawdown > self.stats["max_drawdown"]:
            self.stats["max_drawdown"] = drawdown
        
        # 计算 ROI
        total_invested = self.stats["total_won"] + self.stats["total_lost"]
        if total_invested > 0:
            net_profit = self.current_balance - self.initial_balance
            self.stats["roi"] = (net_profit / total_invested) * 100
        
        return self.current_balance
    
    def _add_transaction(
        self, 
        type: str, 
        amount: float, 
        description: str,
        bet_id: Optional[int] = None
    ):
        """添加交易记录"""
        self.transaction_counter += 1
        transaction = Transaction(
            id=self.transaction_counter,
            type=type,
            amount=amount,
            balance_after=self.current_balance,
            description=description,
            bet_id=bet_id
        )
        self.transactions.append(transaction)
    
    def deposit(self, amount: float) -> float:
        """存款"""
        if amount <= 0:
            return self.current_balance
        
        self.current_balance += amount
        self._add_transaction(
            type="deposit",
            amount=amount,
            description="充值"
        )
        logger.info(f"充值：+{amount}")
        return self.current_balance
    
    def withdraw(self, amount: float) -> float:
        """取款"""
        if amount <= 0 or amount > self.current_balance:
            return self.current_balance
        
        self.current_balance -= amount
        self._add_transaction(
            type="withdrawal",
            amount=-amount,
            description="提现"
        )
        logger.info(f"提现：-{amount}")
        return self.current_balance
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "current_balance": self.current_balance,
            "initial_balance": self.initial_balance,
            "active_bets_count": len(self.active_bets),
            "total_transactions": len(self.transactions),
            "win_rate": self.stats["wins"] / max(1, self.stats["total_bets"]) * 100
        }
    
    def export_history(self, filepath: str):
        """导出历史记录"""
        data = {
            "stats": self.get_stats(),
            "transactions": [t.to_dict() for t in self.transactions],
            "active_bets": self.active_bets
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"历史记录已导出到 {filepath}")
    
    def reset(self, new_initial: Optional[float] = None):
        """重置资金管理"""
        self.initial_balance = new_initial or self.initial_balance
        self.current_balance = self.initial_balance
        self.transactions = []
        self.active_bets = []
        self.transaction_counter = 0
        self.stats = {
            "total_bets": 0,
            "wins": 0,
            "losses": 0,
            "total_won": 0.0,
            "total_lost": 0.0,
            "roi": 0.0,
            "max_drawdown": 0.0,
            "peak_balance": self.initial_balance
        }
        logger.info("资金管理已重置")
