"""
进球预测模型
预测下一粒进球的概率以及总进球数
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available, using fallback models")


class GoalPredictor:
    """进球预测器"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.next_goal_model = None
        self.total_goals_model = None
        self.is_trained = False
        
        if SKLEARN_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """初始化模型"""
        # 下一球预测（分类）：主队进球、客队进球、无进球
        self.next_goal_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        # 总进球数预测（回归）
        self.total_goals_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_goal: np.ndarray, 
        y_total: np.ndarray
    ) -> Dict[str, float]:
        """训练模型"""
        if not SKLEARN_AVAILABLE:
            logger.error("Scikit-learn not available for training")
            return {}
        
        logger.info("开始训练进球预测模型...")
        
        # 训练下一球预测模型
        self.next_goal_model.fit(X_train, y_goal)
        
        # 训练总进球数预测模型
        self.total_goals_model.fit(X_train, y_total)
        
        self.is_trained = True
        logger.info("模型训练完成")
        
        # 计算训练分数
        train_score_goal = self.next_goal_model.score(X_train, y_goal)
        train_score_total = self.total_goals_model.score(X_train, y_total)
        
        return {
            "next_goal_accuracy": train_score_goal,
            "total_goals_r2": train_score_total
        }
    
    def predict_next_goal(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        预测下一粒进球
        返回：主队进球概率、客队进球概率、无进球概率
        """
        if not self.is_trained:
            # 使用启发式方法作为 fallback
            return self._heuristic_next_goal_prediction(features)
        
        # 准备特征向量
        X = self._prepare_features(features)
        
        # 预测概率
        probs = self.next_goal_model.predict_proba([X])[0]
        
        return {
            "home_goal_prob": float(probs[0]) if len(probs) > 0 else 0.33,
            "away_goal_prob": float(probs[1]) if len(probs) > 1 else 0.33,
            "no_goal_prob": float(probs[2]) if len(probs) > 2 else 0.34
        }
    
    def predict_total_goals(self, features: Dict[str, Any]) -> Dict[str, float]:
        """预测总进球数"""
        if not self.is_trained:
            return self._heuristic_total_goals_prediction(features)
        
        X = self._prepare_features(features)
        predicted_total = self.total_goals_model.predict([X])[0]
        
        # 计算各种大小球的概率（使用正态分布近似）
        std = 1.0  # 假设标准差
        over_2_5_prob = 1 - self._normal_cdf(2.5, predicted_total, std)
        over_3_5_prob = 1 - self._normal_cdf(3.5, predicted_total, std)
        under_2_5_prob = self._normal_cdf(2.5, predicted_total, std)
        
        return {
            "expected_total_goals": float(predicted_total),
            "over_2_5_prob": float(over_2_5_prob),
            "over_3_5_prob": float(over_3_5_prob),
            "under_2_5_prob": float(under_2_5_prob)
        }
    
    def _heuristic_next_goal_prediction(self, features: Dict[str, Any]) -> Dict[str, float]:
        """启发式下一球预测（fallback）"""
        home_pressure = features.get("home_pressure_index", 5)
        away_pressure = features.get("away_pressure_index", 5)
        
        total_pressure = home_pressure + away_pressure
        if total_pressure == 0:
            total_pressure = 1
        
        home_prob = home_pressure / total_pressure * 0.7  # 70% 概率有进球
        away_prob = away_pressure / total_pressure * 0.7
        no_goal_prob = 0.3
        
        # 归一化
        total = home_prob + away_prob + no_goal_prob
        home_prob /= total
        away_prob /= total
        no_goal_prob /= total
        
        return {
            "home_goal_prob": home_prob,
            "away_goal_prob": away_prob,
            "no_goal_prob": no_goal_prob
        }
    
    def _heuristic_total_goals_prediction(self, features: Dict[str, Any]) -> Dict[str, float]:
        """启发式总进球数预测（fallback）"""
        current_goals = features.get("total_goals", 0)
        time_ratio = features.get("time_ratio", 0.5)
        goals_per_10min = features.get("goals_per_10min", 0.5)
        
        # 预计剩余时间的进球数
        remaining_time_ratio = 1 - time_ratio
        expected_additional_goals = goals_per_10min * remaining_time_ratio * 9
        
        expected_total = current_goals + expected_additional_goals
        
        return {
            "expected_total_goals": float(expected_total),
            "over_2_5_prob": 0.5 if expected_total > 2.5 else 0.4,
            "over_3_5_prob": 0.5 if expected_total > 3.5 else 0.3,
            "under_2_5_prob": 0.5 if expected_total < 2.5 else 0.4
        }
    
    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """准备特征向量"""
        feature_order = [
            "elapsed_time", "home_score", "away_score", "total_goals",
            "goal_difference", "home_shots", "away_shots",
            "home_shots_on_target", "away_shots_on_target",
            "home_possession", "away_possession",
            "home_corners", "away_corners",
            "home_attacks", "away_attacks",
            "time_remaining", "time_ratio",
            "goals_per_10min", "home_goals_per_10min", "away_goals_per_10min",
            "goal_momentum",
            "home_pressure_index", "away_pressure_index",
            "implied_home_prob", "implied_draw_prob", "implied_away_prob"
        ]
        
        return np.array([features.get(f, 0) for f in feature_order])
    
    def _normal_cdf(self, x: float, mu: float, sigma: float) -> float:
        """正态分布累积分布函数"""
        z = (x - mu) / sigma
        return 0.5 * (1 + np.erf(z / np.sqrt(2)))
    
    def save_model(self, path: Optional[str] = None):
        """保存模型"""
        import pickle
        path = path or self.model_path
        if path and self.is_trained:
            with open(path, 'wb') as f:
                pickle.dump({
                    'next_goal_model': self.next_goal_model,
                    'total_goals_model': self.total_goals_model
                }, f)
            logger.info(f"模型已保存到 {path}")
    
    def load_model(self, path: Optional[str] = None):
        """加载模型"""
        import pickle
        path = path or self.model_path
        if path:
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    self.next_goal_model = data['next_goal_model']
                    self.total_goals_model = data['total_goals_model']
                    self.is_trained = True
                logger.info(f"模型已从 {path} 加载")
            except Exception as e:
                logger.error(f"加载模型失败：{e}")
