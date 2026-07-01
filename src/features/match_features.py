"""
比赛特征构建器
构建用于滚球预测的高级特征
"""

from typing import Dict, Any, List
import numpy as np
from .feature_extractor import BaseMatchFeatureExtractor


class MatchFeatureBuilder(BaseMatchFeatureExtractor):
    """比赛特征构建器 - 扩展基础特征提取器"""
    
    def __init__(self):
        super().__init__()
        # 添加更多高级特征
        self.feature_names.extend([
            # 时间相关特征
            "time_remaining",
            "time_ratio",
            "is_first_half",
            "is_second_half",
            "injury_time_likely",
            
            # 进球节奏特征
            "goals_per_10min",
            "home_goals_per_10min",
            "away_goals_per_10min",
            "goal_momentum",
            
            # 压力指数
            "home_pressure_index",
            "away_pressure_index",
            
            # 赔率相关特征
            "implied_home_prob",
            "implied_draw_prob",
            "implied_away_prob",
            "odds_movement",
            
            # 目标变量
            "next_goal_home",
            "next_goal_away",
            "over_2_5",
            "btts"
        ])
    
    def extract(self, data: Dict[str, Any], odds_data: Optional[Dict] = None) -> Dict[str, Any]:
        """提取完整特征集"""
        # 获取基础特征
        features = super().extract(data)
        
        # 添加时间相关特征
        elapsed = features["elapsed_time"]
        features["time_remaining"] = max(0, 95 - elapsed)  # 假设最多补时 5 分钟
        features["time_ratio"] = elapsed / 95.0
        features["is_first_half"] = 1 if elapsed < 45 else 0
        features["is_second_half"] = 1 if elapsed >= 45 else 0
        features["injury_time_likely"] = 1 if elapsed >= 44 and elapsed <= 46 else 0
        
        # 进球节奏特征
        total_goals = features["total_goals"]
        elapsed_10min_units = max(0.1, elapsed / 10.0)
        features["goals_per_10min"] = total_goals / elapsed_10min_units
        features["home_goals_per_10min"] = features["home_score"] / elapsed_10min_units
        features["away_goals_per_10min"] = features["away_score"] / elapsed_10min_units
        
        # 进球势头（最近 10 分钟是否有进球）
        features["goal_momentum"] = self._calculate_goal_momentum(data)
        
        # 压力指数
        features["home_pressure_index"] = self._calculate_pressure_index(
            features["home_shots"],
            features["home_shots_on_target"],
            features["home_possession"],
            features["home_attacks"],
            features["home_score"]
        )
        features["away_pressure_index"] = self._calculate_pressure_index(
            features["away_shots"],
            features["away_shots_on_target"],
            features["away_possession"],
            features["away_attacks"],
            features["away_score"]
        )
        
        # 赔率相关特征
        if odds_data:
            odds_features = self._extract_odds_features(odds_data)
            features.update(odds_features)
        else:
            features["implied_home_prob"] = 0.33
            features["implied_draw_prob"] = 0.34
            features["implied_away_prob"] = 0.33
            features["odds_movement"] = 0.0
        
        return features
    
    def _calculate_goal_momentum(self, data: Dict[str, Any]) -> float:
        """计算进球势头"""
        events = data.get("events", [])
        elapsed = data.get("elapsed", 90)
        
        recent_goals = 0
        for event in events:
            if event.get("type") == "Goal":
                event_time = event.get("time", 0)
                if elapsed - event_time <= 10:
                    recent_goals += 1
        
        return min(1.0, recent_goals / 2.0)  # 归一化到 0-1
    
    def _calculate_pressure_index(
        self, shots: int, shots_on_target: int, 
        possession: int, attacks: int, score: int
    ) -> float:
        """计算压力指数"""
        # 简单加权计算
        pressure = (
            shots * 0.2 +
            shots_on_target * 0.3 +
            possession * 0.005 +
            attacks * 0.001
        )
        
        # 落后时压力增加
        if score == 0:
            pressure *= 1.2
        
        return min(10.0, pressure)  # 上限为 10
    
    def _extract_odds_features(self, odds_data: Dict[str, Any]) -> Dict[str, float]:
        """从赔率数据提取特征"""
        features = {}
        
        # 获取平均赔率
        avg_odds = odds_data.get("average_odds", {})
        
        home_odd = avg_odds.get("Home", 3.0)
        draw_odd = avg_odds.get("Draw", 3.0)
        away_odd = avg_odds.get("Away", 3.0)
        
        # 计算隐含概率（去除水钱）
        total_implied = (1/home_odd + 1/draw_odd + 1/away_odd)
        features["implied_home_prob"] = (1/home_odd) / total_implied
        features["implied_draw_prob"] = (1/draw_odd) / total_implied
        features["implied_away_prob"] = (1/away_odd) / total_implied
        
        # 赔率变化（需要历史数据，这里简化）
        features["odds_movement"] = 0.0
        
        return features
    
    def create_training_sample(
        self, 
        match_data: Dict[str, Any], 
        odds_data: Optional[Dict] = None,
        target: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """创建训练样本"""
        features = self.extract(match_data, odds_data)
        
        sample = {
            "match_id": match_data.get("match_id"),
            "timestamp": match_data.get("timestamp"),
            "features": features
        }
        
        if target:
            sample["target"] = target
        
        return sample
