"""
特征提取器基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class FeatureExtractor(ABC):
    """特征提取器抽象基类"""
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or []
    
    @abstractmethod
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """提取特征"""
        pass
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return self.feature_names
    
    def validate_features(self, features: Dict[str, Any]) -> bool:
        """验证特征完整性"""
        for name in self.feature_names:
            if name not in features:
                return False
        return True


class BaseMatchFeatureExtractor(FeatureExtractor):
    """基础比赛特征提取器"""
    
    def __init__(self):
        super().__init__([
            "elapsed_time",
            "home_score",
            "away_score",
            "total_goals",
            "goal_difference",
            "home_shots",
            "away_shots",
            "home_shots_on_target",
            "away_shots_on_target",
            "home_possession",
            "away_possession",
            "home_corners",
            "away_corners",
            "home_attacks",
            "away_attacks"
        ])
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """提取基础比赛特征"""
        features = {}
        
        # 时间特征
        features["elapsed_time"] = data.get("elapsed", 0)
        
        # 比分特征
        home_team = data.get("home_team", {})
        away_team = data.get("away_team", {})
        
        features["home_score"] = home_team.get("score", 0)
        features["away_score"] = away_team.get("score", 0)
        features["total_goals"] = features["home_score"] + features["away_score"]
        features["goal_difference"] = features["home_score"] - features["away_score"]
        
        # 统计特征（如果有）
        stats = data.get("statistics", [])
        stats_dict = self._parse_statistics(stats)
        
        features["home_shots"] = stats_dict.get("home_shots", 0)
        features["away_shots"] = stats_dict.get("away_shots", 0)
        features["home_shots_on_target"] = stats_dict.get("home_shots_on_target", 0)
        features["away_shots_on_target"] = stats_dict.get("away_shots_on_target", 0)
        features["home_possession"] = stats_dict.get("home_possession", 50)
        features["away_possession"] = stats_dict.get("away_possession", 50)
        features["home_corners"] = stats_dict.get("home_corners", 0)
        features["away_corners"] = stats_dict.get("away_corners", 0)
        features["home_attacks"] = stats_dict.get("home_attacks", 0)
        features["away_attacks"] = stats_dict.get("away_attacks", 0)
        
        return features
    
    def _parse_statistics(self, stats: List[Dict]) -> Dict[str, int]:
        """解析统计数据"""
        result = {}
        
        stat_mapping = {
            "Shots on Goal": ("shots_on_target", int),
            "Shots off Goal": ("shots_off_target", int),
            "Total Shots": ("shots", int),
            "Possession %": ("possession", int),
            "Corner Kicks": ("corners", int),
            "Attacks": ("attacks", int)
        }
        
        for stat in stats:
            team = stat.get("team", {}).get("name", "")
            statistics = stat.get("statistics", [])
            
            for s in statistics:
                type_name = s.get("type", "")
                value = s.get("value", "0")
                
                if type_name in stat_mapping:
                    field_name, converter = stat_mapping[type_name]
                    try:
                        parsed_value = converter(value.replace("%", ""))
                    except (ValueError, AttributeError):
                        parsed_value = 0
                    
                    prefix = "home_" if "Home" in team else "away_"
                    result[f"{prefix}{field_name}"] = parsed_value
        
        return result
