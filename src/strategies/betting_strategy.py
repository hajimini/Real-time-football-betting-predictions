"""
投注策略模块
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from loguru import logger


@dataclass
class Bet:
    """投注单"""
    match_id: int
    market: str  # 市场类型：1X2, Over/Under, etc.
    selection: str  # 选择：Home, Draw, Away, Over, Under
    stake: float  # 投注金额
    odds: float  # 赔率
    confidence: float  # 置信度
    expected_value: float  # 期望值
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "market": self.market,
            "selection": self.selection,
            "stake": self.stake,
            "odds": self.odds,
            "confidence": self.confidence,
            "expected_value": self.expected_value
        }


class BettingStrategy(ABC):
    """投注策略抽象基类"""
    
    def __init__(self, min_confidence: float = 0.5, min_ev: float = 1.0):
        self.min_confidence = min_confidence
        self.min_ev = min_ev
    
    @abstractmethod
    def generate_bets(
        self, 
        predictions: Dict[str, Any], 
        odds: Dict[str, float],
        bankroll: float
    ) -> List[Bet]:
        """生成投注建议"""
        pass
    
    def _calculate_ev(self, probability: float, odds: float) -> float:
        """计算期望值"""
        return probability * odds
    
    def _passes_filters(
        self, 
        probability: float, 
        odds: float,
        ev: float
    ) -> bool:
        """检查是否通过过滤条件"""
        if probability < self.min_confidence:
            return False
        if ev < self.min_ev:
            return False
        return True


class ValueBettingStrategy(BettingStrategy):
    """价值投注策略"""
    
    def __init__(
        self, 
        min_confidence: float = 0.4, 
        min_ev: float = 1.05,
        max_stake_percentage: float = 0.05
    ):
        super().__init__(min_confidence, min_ev)
        self.max_stake_percentage = max_stake_percentage
    
    def generate_bets(
        self, 
        predictions: Dict[str, Any], 
        odds: Dict[str, float],
        bankroll: float
    ) -> List[Bet]:
        """基于价值发现生成投注"""
        bets = []
        
        # 1X2 市场
        for outcome in ["home_win", "draw", "away_win"]:
            prob_key = f"{outcome}_prob"
            probability = predictions.get(prob_key, 0)
            
            odd = odds.get(outcome.replace("_win", ""), 0)
            if odd <= 0:
                continue
            
            ev = self._calculate_ev(probability, odd)
            
            if self._passes_filters(probability, odd, ev):
                # Kelly Criterion 简化版计算投注额
                stake = self._kelly_stake(probability, odd, bankroll)
                
                bet = Bet(
                    match_id=predictions.get("match_id", 0),
                    market="1X2",
                    selection=outcome.replace("_win", "").capitalize(),
                    stake=stake,
                    odds=odd,
                    confidence=probability,
                    expected_value=ev
                )
                bets.append(bet)
                logger.info(f"发现价值投注：{bet.selection} @ {odd:.2f}, EV={ev:.3f}")
        
        # 大小球市场
        over_prob = predictions.get("over_2_5_prob", 0)
        under_prob = predictions.get("under_2_5_prob", 0)
        
        over_odd = odds.get("over_2_5", 0)
        under_odd = odds.get("under_2_5", 0)
        
        if over_odd > 0:
            ev_over = self._calculate_ev(over_prob, over_odd)
            if self._passes_filters(over_prob, over_odd, ev_over):
                stake = self._kelly_stake(over_prob, over_odd, bankroll)
                bet = Bet(
                    match_id=predictions.get("match_id", 0),
                    market="Over/Under 2.5",
                    selection="Over",
                    stake=stake,
                    odds=over_odd,
                    confidence=over_prob,
                    expected_value=ev_over
                )
                bets.append(bet)
        
        if under_odd > 0:
            ev_under = self._calculate_ev(under_prob, under_odd)
            if self._passes_filters(under_prob, under_odd, ev_under):
                stake = self._kelly_stake(under_prob, under_odd, bankroll)
                bet = Bet(
                    match_id=predictions.get("match_id", 0),
                    market="Over/Under 2.5",
                    selection="Under",
                    stake=stake,
                    odds=under_odd,
                    confidence=under_prob,
                    expected_value=ev_under
                )
                bets.append(bet)
        
        return bets
    
    def _kelly_stake(
        self, 
        probability: float, 
        odds: float, 
        bankroll: float,
        kelly_fraction: float = 0.25
    ) -> float:
        """
        使用 Kelly Criterion 计算最优投注额
        kelly_fraction 用于降低风险（通常用 1/4 Kelly）
        """
        if odds <= 0:
            return 0
        
        # Kelly formula: f = (bp - q) / b
        # b = odds - 1 (净赔率)
        # p = 获胜概率
        # q = 失败概率 = 1 - p
        b = odds - 1
        p = probability
        q = 1 - p
        
        kelly_percentage = (b * p - q) / b if b > 0 else 0
        
        # 应用分数 Kelly 和最大限制
        adjusted_percentage = kelly_percentage * kelly_fraction
        adjusted_percentage = max(0, min(adjusted_percentage, self.max_stake_percentage))
        
        stake = bankroll * adjusted_percentage
        return round(stake, 2)


class ConservativeBettingStrategy(BettingStrategy):
    """保守投注策略"""
    
    def __init__(self):
        super().__init__(min_confidence=0.6, min_ev=1.1)
    
    def generate_bets(
        self, 
        predictions: Dict[str, Any], 
        odds: Dict[str, float],
        bankroll: float
    ) -> List[Bet]:
        """保守策略只选择高置信度的投注"""
        value_strategy = ValueBettingStrategy(
            min_confidence=0.6,
            min_ev=1.1,
            max_stake_percentage=0.02
        )
        return value_strategy.generate_bets(predictions, odds, bankroll)
