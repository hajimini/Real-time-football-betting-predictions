"""
比赛结果预测模型
预测比赛最终结果（胜平负）
"""

from typing import Dict, Any, Optional
import numpy as np
from loguru import logger

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class MatchOutcomePredictor:
    """比赛结果预测器"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        
        if SKLEARN_AVAILABLE:
            self._initialize_model()
    
    def _initialize_model(self):
        """初始化模型"""
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """训练模型"""
        if not SKLEARN_AVAILABLE:
            logger.error("Scikit-learn not available")
            return {}
        
        logger.info("开始训练比赛结果预测模型...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        accuracy = self.model.score(X_train, y_train)
        logger.info(f"训练完成，准确率：{accuracy:.4f}")
        
        return {"accuracy": accuracy}
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        预测比赛结果
        返回：主胜概率、平局概率、客胜概率
        """
        if not self.is_trained:
            return self._heuristic_prediction(features)
        
        X = self._prepare_features(features)
        probs = self.model.predict_proba([X])[0]
        classes = self.model.classes_
        
        result = {}
        for i, cls in enumerate(classes):
            if cls == 0:  # 主胜
                result["home_win_prob"] = float(probs[i])
            elif cls == 1:  # 平局
                result["draw_prob"] = float(probs[i])
            elif cls == 2:  # 客胜
                result["away_win_prob"] = float(probs[i])
        
        return result
    
    def _heuristic_prediction(self, features: Dict[str, Any]) -> Dict[str, float]:
        """启发式预测（fallback）"""
        home_score = features.get("home_score", 0)
        away_score = features.get("away_score", 0)
        home_pressure = features.get("home_pressure_index", 5)
        away_pressure = features.get("away_pressure_index", 5)
        time_remaining = features.get("time_remaining", 45)
        
        # 基于当前比分和压力计算
        goal_diff = home_score - away_score
        pressure_diff = home_pressure - away_pressure
        
        # 基础概率
        home_prob = 0.33 + goal_diff * 0.1 + pressure_diff * 0.02
        away_prob = 0.33 - goal_diff * 0.1 - pressure_diff * 0.02
        draw_prob = 1 - home_prob - away_prob
        
        # 时间越少，保持当前结果的可能性越大
        if time_remaining < 15:
            if goal_diff > 0:
                home_prob = min(0.9, home_prob + 0.3)
                draw_prob = max(0.05, draw_prob - 0.15)
                away_prob = max(0.05, away_prob - 0.15)
            elif goal_diff < 0:
                away_prob = min(0.9, away_prob + 0.3)
                draw_prob = max(0.05, draw_prob - 0.15)
                home_prob = max(0.05, home_prob - 0.15)
            else:
                draw_prob = min(0.6, draw_prob + 0.2)
        
        # 归一化
        total = home_prob + draw_prob + away_prob
        home_prob /= total
        draw_prob /= total
        away_prob /= total
        
        return {
            "home_win_prob": home_prob,
            "draw_prob": draw_prob,
            "away_win_prob": away_prob
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
    
    def get_best_bet(self, predictions: Dict[str, float], odds: Dict[str, float]) -> Dict[str, Any]:
        """
        根据预测和赔率找出最佳投注选项
        """
        best_value = 0
        best_bet = None
        
        prob_map = {
            "home": predictions.get("home_win_prob", 0),
            "draw": predictions.get("draw_prob", 0),
            "away": predictions.get("away_win_prob", 0)
        }
        
        for outcome, prob in prob_map.items():
            odd = odds.get(outcome, 0)
            if odd > 0:
                # 期望值 = 概率 * 赔率
                expected_value = prob * odd
                if expected_value > best_value and expected_value > 1:
                    best_value = expected_value
                    best_bet = {
                        "outcome": outcome,
                        "probability": prob,
                        "odd": odd,
                        "expected_value": expected_value
                    }
        
        return best_bet
