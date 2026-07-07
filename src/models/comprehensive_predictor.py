"""
综合盘口预测模型
支持 8 种盘口预测：
1. 胜平负 (1X2)
2. 让球盘 (Asian Handicap)
3. 大小球 (Over/Under)
4. 双方都进球 (BTTS)
5. 角球数 (Corners)
6. 罚牌数 (Cards)
7. 进球时间分布 (Goal Timing)
8. 比分 (Correct Score)
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from sklearn.ensemble import (
        RandomForestClassifier, 
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        VotingClassifier
    )
    from sklearn.linear_model import LogisticRegression, PoissonRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available, using fallback models")


@dataclass
class PredictionResult:
    """预测结果数据结构"""
    market_type: str  # 盘口类型
    prediction: str  # 推荐选项
    probability: float  # 概率
    confidence: str  # 置信度：HIGH/MEDIUM/LOW
    expected_value: float  # 期望值
    odds: Optional[float] = None  # 赔率
    reasoning: str = ""  # 推荐理由
    risk_level: str = "MEDIUM"  # 风险等级


class ComprehensiveBettingPredictor:
    """
    综合盘口预测器
    支持 8 种主流足球博彩盘口的预测和分析
    """
    
    def __init__(self):
        self.models = {}
        self.is_trained = False
        self.feature_columns = []
        
        if SKLEARN_AVAILABLE:
            self._initialize_all_models()
    
    def _initialize_all_models(self):
        """初始化所有 8 个预测模型"""
        
        # 1. 胜平负模型 (1X2) - 多分类
        self.models['1x2'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # 2. 让球盘模型 (Asian Handicap) - 多分类
        self.models['handicap'] = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        )
        
        # 3. 大小球模型 (Over/Under) - 二分类 + 回归
        self.models['over_under'] = {
            'classifier': CalibratedClassifierCV(
                VotingClassifier(
                    estimators=[
                        ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5)),
                        ('rf', RandomForestClassifier(n_estimators=100, max_depth=8))
                    ],
                    voting='soft'
                ),
                method='sigmoid',
                cv=3
            ),
            'regressor': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        }
        
        # 4. 双方都进球模型 (BTTS) - 二分类
        self.models['btts'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        # 5. 角球数模型 (Corners) - 回归 + 分类
        self.models['corners'] = {
            'regressor': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ),
            'classifier': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        }
        
        # 6. 罚牌数模型 (Cards) - 回归 + 分类
        self.models['cards'] = {
            'regressor': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ),
            'classifier': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        }
        
        # 7. 进球时间分布模型 (Goal Timing) - 多分类
        self.models['goal_timing'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        
        # 8. 比分模型 (Correct Score) - 使用泊松分布 + 分类
        self.models['correct_score'] = {
            'poisson_home': PoissonRegressor(alpha=0.1),
            'poisson_away': PoissonRegressor(alpha=0.1),
            'classifier': RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                class_weight='balanced',
                random_state=42
            )
        }
    
    def prepare_features(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        特征工程：从原始比赛数据中提取和构造特征
        
        参数:
            match_data: 包含比赛信息的字典，包括：
                - 球队实力数据（排名、身价、近期战绩）
                - 技术统计（控球率、射门、角球等）
                - 伤停信息
                - 历史交锋
                - 战术特点
                - 机构赔率
        
        返回:
            处理后的特征字典
        """
        features = match_data.copy()
        
        # === 基础实力特征 ===
        features['fifa_rank_diff'] = features.get('home_fifa_rank', 50) - features.get('away_fifa_rank', 50)
        features['value_ratio'] = features.get('home_squad_value', 1) / max(features.get('away_squad_value', 1), 0.1)
        features['form_diff'] = features.get('home_recent_form', 5) - features.get('away_recent_form', 5)
        
        # === 主客场优势 ===
        is_home = features.get('is_neutral_venue', False) == False
        features['home_advantage'] = 1 if is_home else 0
        features['home_advantage_strength'] = features.get('home_win_rate_at_home', 0.5) if is_home else 0.5
        
        # === 进攻能力特征 ===
        features['home_attack_strength'] = features.get('home_goals_per_game', 1.5) / 1.5
        features['away_attack_strength'] = features.get('away_goals_per_game', 1.5) / 1.5
        features['home_xg_per_game'] = features.get('home_xg_per_game', 1.5)
        features['away_xg_per_game'] = features.get('away_xg_per_game', 1.5)
        features['attack_diff'] = features['home_attack_strength'] - features['away_attack_strength']
        
        # === 防守能力特征 ===
        features['home_defense_weakness'] = features.get('home_goals_conceded_per_game', 1.0) / 1.0
        features['away_defense_weakness'] = features.get('away_goals_conceded_per_game', 1.0) / 1.0
        features['home_clean_sheet_rate'] = features.get('home_clean_sheet_rate', 0.3)
        features['away_clean_sheet_rate'] = features.get('away_clean_sheet_rate', 0.3)
        features['defense_diff'] = features['away_defense_weakness'] - features['home_defense_weakness']
        
        # === 进球相关特征 ===
        features['total_goal_expectancy'] = features['home_attack_strength'] * features['away_defense_weakness'] + \
                                           features['away_attack_strength'] * features['home_defense_weakness']
        features['btts_likelihood'] = (1 - features['home_clean_sheet_rate']) * (1 - features['away_clean_sheet_rate'])
        
        # === 角球特征 ===
        features['home_corner_rate'] = features.get('home_corners_per_game', 5.0)
        features['away_corner_rate'] = features.get('away_corners_per_game', 4.0)
        features['home_corner_conceded'] = features.get('home_corners_conceded_per_game', 4.0)
        features['away_corner_conceded'] = features.get('away_corners_conceded_per_game', 5.0)
        features['expected_total_corners'] = (features['home_corner_rate'] + features['away_corner_conceded']) / 2 + \
                                            (features['away_corner_rate'] + features['home_corner_conceded']) / 2
        
        # === 罚牌特征 ===
        features['home_yellow_rate'] = features.get('home_yellow_cards_per_game', 1.5)
        features['away_yellow_rate'] = features.get('away_yellow_cards_per_game', 2.0)
        features['home_red_card_tendency'] = features.get('home_red_cards_per_season', 0.2)
        features['away_red_card_tendency'] = features.get('away_red_cards_per_season', 0.3)
        features['referee_strictness'] = features.get('referee_cards_per_game', 4.0)
        features['match_importance'] = features.get('match_importance', 0.7)  # 淘汰赛更高
        features['expected_total_cards'] = features['home_yellow_rate'] + features['away_yellow_rate'] + \
                                          features['referee_strictness'] * 0.3 + features['match_importance'] * 0.5
        
        # === 战术风格特征 ===
        features['home_possession_style'] = features.get('home_avg_possession', 50) / 100
        features['away_possession_style'] = features.get('away_avg_possession', 50) / 100
        features['home_high_press'] = features.get('home_high_press_intensity', 5) / 10
        features['away_low_block'] = features.get('away_low_block_tendency', 5) / 10
        features['tactical_mismatch'] = abs(features['home_high_press'] - features['away_low_block'])
        
        # === 历史交锋特征 ===
        features['h2h_home_win_rate'] = features.get('h2h_home_win_rate', 0.5)
        features['h2h_btts_rate'] = features.get('h2h_btts_rate', 0.5)
        features['h2h_avg_goals'] = features.get('h2h_avg_goals', 2.5)
        
        # === 伤停影响 ===
        features['home_injury_impact'] = features.get('home_key_players_out', 0) * 0.1
        features['away_injury_impact'] = features.get('away_key_players_out', 0) * 0.1
        
        # === 时间相关特征（用于进球时间预测）===
        features['early_goal_tendency'] = features.get('home_first_15min_goal_rate', 0.3)
        features['late_goal_tendency'] = features.get('away_last_15min_goal_rate', 0.25)
        features['first_half_goals_rate'] = features.get('first_half_goals_percentage', 0.45)
        
        # === 赔率隐含概率（如果有）===
        implied_home = 1.0 / features.get('odds_home', 2.0) if features.get('odds_home', 0) > 1 else 0.5
        implied_draw = 1.0 / features.get('odds_draw', 3.5) if features.get('odds_draw', 0) > 1 else 0.28
        implied_away = 1.0 / features.get('odds_away', 4.0) if features.get('odds_away', 0) > 1 else 0.25
        total_implied = implied_home + implied_draw + implied_away
        features['implied_home_prob'] = implied_home / total_implied
        features['implied_draw_prob'] = implied_draw / total_implied
        features['implied_away_prob'] = implied_away / total_implied
        features['bookmaker_margin'] = total_implied - 1.0
        
        return features
    
    def get_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """将特征字典转换为模型可用的向量"""
        feature_order = [
            'fifa_rank_diff', 'value_ratio', 'form_diff',
            'home_advantage', 'home_advantage_strength',
            'home_attack_strength', 'away_attack_strength',
            'home_xg_per_game', 'away_xg_per_game', 'attack_diff',
            'home_defense_weakness', 'away_defense_weakness',
            'home_clean_sheet_rate', 'away_clean_sheet_rate', 'defense_diff',
            'total_goal_expectancy', 'btts_likelihood',
            'home_corner_rate', 'away_corner_rate',
            'home_corner_conceded', 'away_corner_conceded', 'expected_total_corners',
            'home_yellow_rate', 'away_yellow_rate',
            'home_red_card_tendency', 'away_red_card_tendency',
            'referee_strictness', 'match_importance', 'expected_total_cards',
            'home_possession_style', 'away_possession_style',
            'home_high_press', 'away_low_block', 'tactical_mismatch',
            'h2h_home_win_rate', 'h2h_btts_rate', 'h2h_avg_goals',
            'home_injury_impact', 'away_injury_impact',
            'early_goal_tendency', 'late_goal_tendency', 'first_half_goals_rate',
            'implied_home_prob', 'implied_draw_prob', 'implied_away_prob', 'bookmaker_margin'
        ]
        
        self.feature_columns = feature_order
        return np.array([features.get(f, 0) for f in feature_order])
    
    def train_all_models(self, training_data: pd.DataFrame):
        """
        训练所有 8 个预测模型
        
        参数:
            training_data: 包含历史比赛数据和结果的 DataFrame
        """
        if not SKLEARN_AVAILABLE:
            logger.error("Scikit-learn not available for training")
            return
        
        logger.info("开始训练所有 8 个盘口预测模型...")
        
        # 准备特征
        X = []
        y_1x2 = []
        y_handicap = []
        y_over_under = []
        y_btts = []
        y_corners = []
        y_cards = []
        y_goal_timing = []
        y_correct_score = []
        
        for _, row in training_data.iterrows():
            match_data = row.to_dict()
            features = self.prepare_features(match_data)
            feature_vec = self.get_feature_vector(features)
            
            X.append(feature_vec)
            
            # 收集各标签
            y_1x2.append(match_data.get('result_1x2', 1))  # 0=客胜，1=平，2=主胜
            y_handicap.append(match_data.get('result_handicap', 1))  # 0=下盘，1=走水，2=上盘
            y_over_under.append(1 if match_data.get('total_goals', 2) > 2.5 else 0)
            y_btts.append(1 if match_data.get('btts', False) else 0)
            y_corners.append(match_data.get('total_corners', 9))
            y_cards.append(match_data.get('total_cards', 4))
            y_goal_timing.append(match_data.get('first_goal_time_bucket', 2))  # 0-15,16-30,31-45,46-60,61-75,76-90+
            y_correct_score.append(match_data.get('score_category', 5))  # 比分分类
        
        X = np.array(X)
        
        # 训练各个模型
        logger.info("训练 1. 胜平负模型...")
        self.models['1x2'].fit(X, y_1x2)
        
        logger.info("训练 2. 让球盘模型...")
        self.models['handicap'].fit(X, y_handicap)
        
        logger.info("训练 3. 大小球模型...")
        self.models['over_under']['classifier'].fit(X, y_over_under)
        self.models['over_under']['regressor'].fit(X, [training_data['total_goals'].mean()] * len(X))
        
        logger.info("训练 4. BTTS 模型...")
        self.models['btts'].fit(X, y_btts)
        
        logger.info("训练 5. 角球数模型...")
        self.models['corners']['regressor'].fit(X, y_corners)
        corners_binary = [1 if c > 9.5 else 0 for c in y_corners]
        self.models['corners']['classifier'].fit(X, corners_binary)
        
        logger.info("训练 6. 罚牌数模型...")
        self.models['cards']['regressor'].fit(X, y_cards)
        cards_binary = [1 if c > 4.5 else 0 for c in y_cards]
        self.models['cards']['classifier'].fit(X, cards_binary)
        
        logger.info("训练 7. 进球时间模型...")
        self.models['goal_timing'].fit(X, y_goal_timing)
        
        logger.info("训练 8. 比分模型...")
        self.models['correct_score']['classifier'].fit(X, y_correct_score)
        
        self.is_trained = True
        logger.info("✓ 所有 8 个盘口预测模型训练完成！")
    
    def predict_all_markets(
        self, 
        match_data: Dict[str, Any],
        odds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        对单场比赛进行全方位 8 种盘口预测
        
        参数:
            match_data: 比赛基础数据
            odds: 可选的赔率数据，用于计算期望值
        
        返回:
            包含 8 种盘口预测结果的字典
        """
        if not self.is_trained:
            logger.warning("模型未训练，使用启发式预测")
            return self._heuristic_prediction(match_data, odds)
        
        # 准备特征
        features = self.prepare_features(match_data)
        X = self.get_feature_vector(features).reshape(1, -1)
        
        results = {}
        
        # === 1. 胜平负预测 ===
        results['1x2'] = self._predict_1x2(X, features, odds)
        
        # === 2. 让球盘预测 ===
        results['asian_handicap'] = self._predict_handicap(X, features, odds)
        
        # === 3. 大小球预测 ===
        results['over_under'] = self._predict_over_under(X, features, odds)
        
        # === 4. BTTS 预测 ===
        results['btts'] = self._predict_btts(X, features, odds)
        
        # === 5. 角球数预测 ===
        results['corners'] = self._predict_corners(X, features, odds)
        
        # === 6. 罚牌数预测 ===
        results['cards'] = self._predict_cards(X, features, odds)
        
        # === 7. 进球时间预测 ===
        results['goal_timing'] = self._predict_goal_timing(X, features)
        
        # === 8. 比分预测 ===
        results['correct_score'] = self._predict_correct_score(X, features, odds)
        
        # === 生成综合推荐 ===
        results['summary'] = self._generate_summary(results, odds)
        
        return results
    
    def _predict_1x2(self, X: np.ndarray, features: Dict, odds: Optional[Dict]) -> Dict[str, Any]:
        """胜平负预测"""
        probs = self.models['1x2'].predict_proba(X)[0]
        classes = self.models['1x2'].classes_
        
        prob_map = {}
        for i, cls in enumerate(classes):
            if cls == 2:
                prob_map['home_win'] = float(probs[i])
            elif cls == 1:
                prob_map['draw'] = float(probs[i])
            elif cls == 0:
                prob_map['away_win'] = float(probs[i])
        
        # 确定推荐
        max_prob = max(prob_map.values())
        if prob_map['home_win'] == max_prob:
            recommendation = 'HOME_WIN'
        elif prob_map['draw'] == max_prob:
            recommendation = 'DRAW'
        else:
            recommendation = 'AWAY_WIN'
        
        confidence = 'HIGH' if max_prob > 0.6 else 'MEDIUM' if max_prob > 0.45 else 'LOW'
        
        # 计算期望值
        ev = 0.0
        if odds:
            odd_key = {'HOME_WIN': 'home', 'DRAW': 'draw', 'AWAY_WIN': 'away'}[recommendation]
            odd = odds.get(odd_key, 0)
            ev = max_prob * odd if odd > 0 else 0
        
        return {
            'probabilities': prob_map,
            'recommendation': recommendation,
            'probability': max_prob,
            'confidence': confidence,
            'expected_value': ev,
            'reasoning': self._generate_1x2_reasoning(features, prob_map)
        }
    
    def _predict_handicap(self, X: np.ndarray, features: Dict, odds: Optional[Dict]) -> Dict[str, Any]:
        """让球盘预测"""
        probs = self.models['handicap'].predict_proba(X)[0]
        classes = self.models['handicap'].classes_
        
        prob_map = {}
        for i, cls in enumerate(classes):
            if cls == 2:
                prob_map['upper'] = float(probs[i])  # 上盘
            elif cls == 1:
                prob_map['push'] = float(probs[i])   # 走水
            elif cls == 0:
                prob_map['lower'] = float(probs[i])  # 下盘
        
        # 简化：只推荐上盘或下盘
        upper_prob = prob_map.get('upper', 0.33)
        lower_prob = prob_map.get('lower', 0.33)
        
        if upper_prob > lower_prob:
            recommendation = 'UPPER'
            probability = upper_prob
        else:
            recommendation = 'LOWER'
            probability = lower_prob
        
        confidence = 'HIGH' if probability > 0.55 else 'MEDIUM' if probability > 0.45 else 'LOW'
        
        ev = 0.0
        if odds and recommendation == 'UPPER':
            ev = probability * odds.get('upper', 0)
        elif odds and recommendation == 'LOWER':
            ev = probability * odds.get('lower', 0)
        
        return {
            'probabilities': prob_map,
            'recommendation': recommendation,
            'probability': probability,
            'confidence': confidence,
            'expected_value': ev,
            'reasoning': self._generate_handicap_reasoning(features, prob_map)
        }
    
    def _predict_over_under(self, X: np.ndarray, features: Dict, odds: Optional[Dict]) -> Dict[str, Any]:
        """大小球预测"""
        probs = self.models['over_under']['classifier'].predict_proba(X)[0]
        expected_goals = self.models['over_under']['regressor'].predict(X)[0]
        
        over_prob = float(probs[1]) if len(probs) > 1 else 0.5
        under_prob = float(probs[0]) if len(probs) > 0 else 0.5
        
        recommendation = 'OVER' if over_prob > under_prob else 'UNDER'
        probability = max(over_prob, under_prob)
        confidence = 'HIGH' if probability > 0.6 else 'MEDIUM' if probability > 0.5 else 'LOW'
        
        ev = 0.0
        if odds:
            odd_key = 'over_2_5' if recommendation == 'OVER' else 'under_2_5'
            odd = odds.get(odd_key, 0)
            ev = probability * odd if odd > 0 else 0
        
        return {
            'expected_goals': float(expected_goals),
            'over_probability': over_prob,
            'under_probability': under_prob,
            'recommendation': recommendation,
            'probability': probability,
            'confidence': confidence,
            'expected_value': ev,
            'reasoning': self._generate_over_under_reasoning(features, expected_goals)
        }
    
    def _predict_btts(self, X: np.ndarray, features: Dict, odds: Optional[Dict]) -> Dict[str, Any]:
        """双方都进球预测"""
        probs = self.models['btts'].predict_proba(X)[0]
        
        yes_prob = float(probs[1]) if len(probs) > 1 else features.get('btts_likelihood', 0.5)
        no_prob = float(probs[0]) if len(probs) > 0 else 1 - yes_prob
        
        recommendation = 'YES' if yes_prob > no_prob else 'NO'
        probability = max(yes_prob, no_prob)
        confidence = 'HIGH' if probability > 0.65 else 'MEDIUM' if probability > 0.55 else 'LOW'
        
        ev = 0.0
        if odds:
            odd_key = 'btts_yes' if recommendation == 'YES' else 'btts_no'
            odd = odds.get(odd_key, 0)
            ev = probability * odd if odd > 0 else 0
        
        return {
            'yes_probability': yes_prob,
            'no_probability': no_prob,
            'recommendation': recommendation,
            'probability': probability,
            'confidence': confidence,
            'expected_value': ev,
            'reasoning': self._generate_btts_reasoning(features, yes_prob)
        }
    
    def _predict_corners(self, X: np.ndarray, features: Dict, odds: Optional[Dict]) -> Dict[str, Any]:
        """角球数预测"""
        expected_corners = self.models['corners']['regressor'].predict(X)[0]
        probs = self.models['corners']['classifier'].predict_proba(X)[0]
        
        over_prob = float(probs[1]) if len(probs) > 1 else 0.5
        under_prob = float(probs[0]) if len(probs) > 0 else 0.5
        
        recommendation = 'OVER' if over_prob > under_prob else 'UNDER'
        probability = max(over_prob, under_prob)
        confidence = 'HIGH' if probability > 0.6 else 'MEDIUM' if probability > 0.5 else 'LOW'
        
        ev = 0.0
        if odds:
            odd_key = 'corners_over_9_5' if recommendation == 'OVER' else 'corners_under_9_5'
            odd = odds.get(odd_key, 0)
            ev = probability * odd if odd > 0 else 0
        
        return {
            'expected_corners': float(expected_corners),
            'over_probability': over_prob,
            'under_probability': under_prob,
            'recommendation': recommendation,
            'probability': probability,
            'confidence': confidence,
            'expected_value': ev,
            'reasoning': self._generate_corners_reasoning(features, expected_corners)
        }
    
    def _predict_cards(self, X: np.ndarray, features: Dict, odds: Optional[Dict]) -> Dict[str, Any]:
        """罚牌数预测"""
        expected_cards = self.models['cards']['regressor'].predict(X)[0]
        probs = self.models['cards']['classifier'].predict_proba(X)[0]
        
        over_prob = float(probs[1]) if len(probs) > 1 else 0.5
        under_prob = float(probs[0]) if len(probs) > 0 else 0.5
        
        recommendation = 'OVER' if over_prob > under_prob else 'UNDER'
        probability = max(over_prob, under_prob)
        confidence = 'HIGH' if probability > 0.6 else 'MEDIUM' if probability > 0.5 else 'LOW'
        
        ev = 0.0
        if odds:
            odd_key = 'cards_over_4_5' if recommendation == 'OVER' else 'cards_under_4_5'
            odd = odds.get(odd_key, 0)
            ev = probability * odd if odd > 0 else 0
        
        return {
            'expected_cards': float(expected_cards),
            'over_probability': over_prob,
            'under_probability': under_prob,
            'recommendation': recommendation,
            'probability': probability,
            'confidence': confidence,
            'expected_value': ev,
            'reasoning': self._generate_cards_reasoning(features, expected_cards)
        }
    
    def _predict_goal_timing(self, X: np.ndarray, features: Dict) -> Dict[str, Any]:
        """进球时间分布预测"""
        probs = self.models['goal_timing'].predict_proba(X)[0]
        classes = self.models['goal_timing'].classes_
        
        time_buckets = ['0-15min', '16-30min', '31-45+min', '46-60min', '61-75min', '76-90+min']
        bucket_probs = {}
        
        for i, cls in enumerate(classes):
            if 0 <= cls < len(time_buckets):
                bucket_probs[time_buckets[cls]] = float(probs[i])
        
        # 找出最可能的时间段
        max_bucket = max(bucket_probs, key=bucket_probs.get) if bucket_probs else '0-15min'
        max_prob = bucket_probs.get(max_bucket, 0.2)
        
        confidence = 'HIGH' if max_prob > 0.3 else 'MEDIUM' if max_prob > 0.2 else 'LOW'
        
        return {
            'time_bucket_probabilities': bucket_probs,
            'most_likely_period': max_bucket,
            'probability': max_prob,
            'confidence': confidence,
            'reasoning': self._generate_goal_timing_reasoning(features, bucket_probs)
        }
    
    def _predict_correct_score(self, X: np.ndarray, features: Dict, odds: Optional[Dict]) -> Dict[str, Any]:
        """比分预测（使用泊松分布）"""
        # 预测主队和客队进球数
        home_expected = features.get('home_attack_strength', 1.5) * features.get('away_defense_weakness', 1.0)
        away_expected = features.get('away_attack_strength', 1.5) * features.get('home_defense_weakness', 1.0)
        
        # 使用泊松分布计算各比分概率
        score_probs = {}
        for home_goals in range(5):
            for away_goals in range(5):
                home_prob = self._poisson_pmf(home_goals, home_expected)
                away_prob = self._poisson_pmf(away_goals, away_expected)
                score_probs[f"{home_goals}-{away_goals}"] = home_prob * away_prob
        
        # 归一化
        total = sum(score_probs.values())
        score_probs = {k: v/total for k, v in score_probs.items()}
        
        # 找出最可能的比分
        top_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        most_likely_score = top_scores[0][0]
        most_likely_prob = top_scores[0][1]
        
        confidence = 'HIGH' if most_likely_prob > 0.15 else 'MEDIUM' if most_likely_prob > 0.10 else 'LOW'
        
        ev = 0.0
        if odds:
            odd = odds.get(f'score_{most_likely_score}', 0)
            ev = most_likely_prob * odd if odd > 0 else 0
        
        return {
            'score_probabilities': score_probs,
            'top_3_scores': top_scores,
            'most_likely_score': most_likely_score,
            'probability': most_likely_prob,
            'confidence': confidence,
            'expected_value': ev,
            'reasoning': self._generate_score_reasoning(features, home_expected, away_expected)
        }
    
    def _poisson_pmf(self, k: int, lam: float) -> float:
        """泊松分布概率质量函数"""
        from math import exp, factorial
        return (lam ** k) * exp(-lam) / factorial(k)
    
    def _generate_summary(self, results: Dict, odds: Optional[Dict]) -> Dict[str, Any]:
        """生成综合推荐摘要"""
        high_confidence_bets = []
        value_bets = []
        
        markets = ['1x2', 'asian_handicap', 'over_under', 'btts', 'corners', 'cards']
        
        for market in markets:
            result = results.get(market, {})
            confidence = result.get('confidence', 'LOW')
            ev = result.get('expected_value', 0)
            
            if confidence == 'HIGH':
                high_confidence_bets.append({
                    'market': market,
                    'recommendation': result.get('recommendation'),
                    'probability': result.get('probability', 0)
                })
            
            if ev > 1.05:  # 正期望值
                value_bets.append({
                    'market': market,
                    'recommendation': result.get('recommendation'),
                    'expected_value': ev
                })
        
        return {
            'high_confidence_bets': high_confidence_bets,
            'value_bets': value_bets,
            'total_markets_analyzed': 8,
            'recommendation_summary': f"找到 {len(high_confidence_bets)} 个高信心投注，{len(value_bets)} 个价值投注"
        }
    
    # === 理由生成方法（简化版）===
    def _generate_1x2_reasoning(self, features: Dict, probs: Dict) -> str:
        attack_diff = features.get('attack_diff', 0)
        home_adv = features.get('home_advantage', 0)
        
        if attack_diff > 0.3:
            return "主队进攻实力明显占优"
        elif attack_diff < -0.3:
            return "客队进攻实力明显占优"
        elif home_adv and abs(attack_diff) < 0.2:
            return "主队主场优势关键"
        else:
            return "双方实力接近，平局可能性大"
    
    def _generate_handicap_reasoning(self, features: Dict, probs: Dict) -> str:
        strength_diff = features.get('value_ratio', 1)
        if strength_diff > 1.5:
            return "实力差距明显，但需防赢球输盘"
        elif strength_diff < 0.8:
            return "弱势方有望守住盘口"
        else:
            return "实力接近，盘口胶着"
    
    def _generate_over_under_reasoning(self, features: Dict, expected: float) -> str:
        if expected > 3.0:
            return f"预期进球{expected:.1f}个，大球概率高"
        elif expected < 2.0:
            return f"预期进球{expected:.1f}个，小球概率高"
        else:
            return f"预期进球{expected:.1f}个，大小球均衡"
    
    def _generate_btts_reasoning(self, features: Dict, yes_prob: float) -> str:
        home_cs = features.get('home_clean_sheet_rate', 0.3)
        away_cs = features.get('away_clean_sheet_rate', 0.3)
        
        if home_cs < 0.2 and away_cs < 0.2:
            return "双方防守都不稳，大概率都有进球"
        elif home_cs > 0.5 or away_cs > 0.5:
            return "有一方防守稳固，可能零封"
        else:
            return "双方攻防均衡"
    
    def _generate_corners_reasoning(self, features: Dict, expected: float) -> str:
        if expected > 10:
            return f"预期角球{expected:.1f}个，大角概率高"
        elif expected < 8:
            return f"预期角球{expected:.1f}个，小角概率高"
        else:
            return f"预期角球{expected:.1f}个，中等数量"
    
    def _generate_cards_reasoning(self, features: Dict, expected: float) -> str:
        importance = features.get('match_importance', 0.5)
        ref_strict = features.get('referee_strictness', 4)
        
        if importance > 0.8 or ref_strict > 5:
            return "重要比赛或裁判严格，牌数可能偏多"
        elif expected > 5:
            return f"预期黄牌{expected:.1f}张，大牌概率高"
        else:
            return f"预期黄牌{expected:.1f}张，正常范围"
    
    def _generate_goal_timing_reasoning(self, features: Dict, probs: Dict) -> str:
        early = features.get('early_goal_tendency', 0.3)
        late = features.get('late_goal_tendency', 0.25)
        
        if early > 0.4:
            return "主队开局进攻凶猛，前 15 分钟易进球"
        elif late > 0.35:
            return "客队后程发力，75 分钟后易进球"
        else:
            return "进球时间分布均匀"
    
    def _generate_score_reasoning(self, features: Dict, home_exp: float, away_exp: float) -> str:
        if home_exp > 2 and away_exp < 1:
            return f"主队火力强 ({home_exp:.1f}球)，客队弱 ({away_exp:.1f}球)"
        elif away_exp > 2 and home_exp < 1:
            return f"客队火力强 ({away_exp:.1f}球)，主队弱 ({home_exp:.1f}球)"
        elif abs(home_exp - away_exp) < 0.5:
            return "双方进攻力接近"
        else:
            return "对攻战预期"
    
    def _heuristic_prediction(self, match_data: Dict, odds: Optional[Dict]) -> Dict[str, Any]:
        """当模型未训练时的启发式预测"""
        features = self.prepare_features(match_data)
        
        # 简化的规则-based 预测
        results = {
            '1x2': {
                'recommendation': 'HOME_WIN' if features['attack_diff'] > 0.2 else 'AWAY_WIN' if features['attack_diff'] < -0.2 else 'DRAW',
                'probability': 0.5,
                'confidence': 'LOW',
                'reasoning': '基于实力对比的启发式预测'
            },
            'over_under': {
                'recommendation': 'OVER' if features['total_goal_expectancy'] > 2.5 else 'UNDER',
                'probability': 0.5,
                'confidence': 'LOW',
                'reasoning': '基于进球期望的启发式预测'
            },
            'btts': {
                'recommendation': 'YES' if features['btts_likelihood'] > 0.5 else 'NO',
                'probability': features['btts_likelihood'],
                'confidence': 'LOW',
                'reasoning': '基于双方防守数据的启发式预测'
            },
            'corners': {
                'recommendation': 'OVER' if features['expected_total_corners'] > 9.5 else 'UNDER',
                'probability': 0.5,
                'confidence': 'LOW',
                'reasoning': '基于角球数据的启发式预测'
            },
            'cards': {
                'recommendation': 'OVER' if features['expected_total_cards'] > 4.5 else 'UNDER',
                'probability': 0.5,
                'confidence': 'LOW',
                'reasoning': '基于罚牌数据的启发式预测'
            },
            'goal_timing': {
                'most_likely_period': '0-15min' if features['early_goal_tendency'] > 0.3 else '76-90+min',
                'probability': 0.25,
                'confidence': 'LOW',
                'reasoning': '基于进球时间趋势的启发式预测'
            },
            'correct_score': {
                'most_likely_score': '2-1' if features['attack_diff'] > 0 else '1-1' if abs(features['attack_diff']) < 0.2 else '1-2',
                'probability': 0.1,
                'confidence': 'LOW',
                'reasoning': '基于攻防数据的启发式预测'
            },
            'asian_handicap': {
                'recommendation': 'UPPER' if features['attack_diff'] > 0.2 else 'LOWER',
                'probability': 0.5,
                'confidence': 'LOW',
                'reasoning': '基于实力差距的启发式预测'
            },
            'summary': {
                'high_confidence_bets': [],
                'value_bets': [],
                'recommendation_summary': '启发式预测仅供参考，建议训练模型后使用'
            }
        }
        
        return results
    
    def export_predictions_to_report(self, predictions: Dict[str, Any], match_info: Dict[str, Any]) -> str:
        """
        将预测结果导出为详细报告（类似用户提供的示例格式）
        
        参数:
            predictions: predict_all_markets 的返回结果
            match_info: 比赛基本信息
        
        返回:
            格式化的报告文本
        """
        report = []
        report.append("=" * 80)
        report.append(f"比赛预测报告：{match_info.get('home_team', '主队')} VS {match_info.get('away_team', '客队')}")
        report.append("=" * 80)
        report.append("")
        
        # 1. 胜平负
        r = predictions.get('1x2', {})
        report.append("【1. 胜平负预测】")
        report.append(f"  推荐：{r.get('recommendation', 'N/A')}")
        report.append(f"  概率：{r.get('probability', 0)*100:.1f}%")
        report.append(f"  置信度：{r.get('confidence', 'N/A')}")
        report.append(f"  理由：{r.get('reasoning', '')}")
        report.append("")
        
        # 2. 让球盘
        r = predictions.get('asian_handicap', {})
        report.append("【2. 让球盘预测】")
        report.append(f"  推荐：{r.get('recommendation', 'N/A')}")
        report.append(f"  概率：{r.get('probability', 0)*100:.1f}%")
        report.append(f"  置信度：{r.get('confidence', 'N/A')}")
        report.append(f"  理由：{r.get('reasoning', '')}")
        report.append("")
        
        # 3. 大小球
        r = predictions.get('over_under', {})
        report.append("【3. 大小球预测】")
        report.append(f"  推荐：{r.get('recommendation', 'N/A')} {r.get('expected_goals', 0):.1f}球")
        report.append(f"  概率：{r.get('probability', 0)*100:.1f}%")
        report.append(f"  置信度：{r.get('confidence', 'N/A')}")
        report.append(f"  理由：{r.get('reasoning', '')}")
        report.append("")
        
        # 4. BTTS
        r = predictions.get('btts', {})
        report.append("【4. 双方都进球 (BTTS)】")
        report.append(f"  推荐：{r.get('recommendation', 'N/A')}")
        report.append(f"  概率：{r.get('probability', 0)*100:.1f}%")
        report.append(f"  置信度：{r.get('confidence', 'N/A')}")
        report.append(f"  理由：{r.get('reasoning', '')}")
        report.append("")
        
        # 5. 角球数
        r = predictions.get('corners', {})
        report.append("【5. 角球数预测】")
        report.append(f"  推荐：{r.get('recommendation', 'N/A')} {r.get('expected_corners', 0):.1f}个")
        report.append(f"  概率：{r.get('probability', 0)*100:.1f}%")
        report.append(f"  置信度：{r.get('confidence', 'N/A')}")
        report.append(f"  理由：{r.get('reasoning', '')}")
        report.append("")
        
        # 6. 罚牌数
        r = predictions.get('cards', {})
        report.append("【6. 罚牌数预测】")
        report.append(f"  推荐：{r.get('recommendation', 'N/A')} {r.get('expected_cards', 0):.1f}张")
        report.append(f"  概率：{r.get('probability', 0)*100:.1f}%")
        report.append(f"  置信度：{r.get('confidence', 'N/A')}")
        report.append(f"  理由：{r.get('reasoning', '')}")
        report.append("")
        
        # 7. 进球时间
        r = predictions.get('goal_timing', {})
        report.append("【7. 进球时间分布】")
        report.append(f"  最可能时段：{r.get('most_likely_period', 'N/A')}")
        report.append(f"  概率：{r.get('probability', 0)*100:.1f}%")
        report.append(f"  置信度：{r.get('confidence', 'N/A')}")
        report.append(f"  理由：{r.get('reasoning', '')}")
        report.append("")
        
        # 8. 比分
        r = predictions.get('correct_score', {})
        report.append("【8. 比分预测】")
        report.append(f"  最可能比分：{r.get('most_likely_score', 'N/A')}")
        report.append(f"  概率：{r.get('probability', 0)*100:.1f}%")
        report.append(f"  前三位：{', '.join([f'{s}({p*100:.1f}%)' for s, p in r.get('top_3_scores', [])])}")
        report.append(f"  置信度：{r.get('confidence', 'N/A')}")
        report.append(f"  理由：{r.get('reasoning', '')}")
        report.append("")
        
        # 综合推荐
        summary = predictions.get('summary', {})
        report.append("=" * 80)
        report.append("【综合推荐汇总】")
        report.append(summary.get('recommendation_summary', ''))
        report.append("")
        
        high_conf = summary.get('high_confidence_bets', [])
        if high_conf:
            report.append("高信心投注：")
            for bet in high_conf:
                report.append(f"  - {bet['market']}: {bet['recommendation']} ({bet['probability']*100:.1f}%)")
        
        value = summary.get('value_bets', [])
        if value:
            report.append("价值投注：")
            for bet in value:
                report.append(f"  - {bet['market']}: {bet['recommendation']} (EV={bet['expected_value']:.2f})")
        
        report.append("=" * 80)
        
        return "\n".join(report)


# 使用示例
if __name__ == "__main__":
    # 创建预测器
    predictor = ComprehensiveBettingPredictor()
    
    # 模拟比赛数据
    match_data = {
        'home_team': '美国',
        'away_team': '波黑',
        'home_fifa_rank': 12,
        'away_fifa_rank': 64,
        'home_squad_value': 3.86,  # 亿欧元
        'away_squad_value': 1.27,
        'home_recent_form': 8,  # 近 5 场得分
        'away_recent_form': 5,
        'is_neutral_venue': False,
        'home_win_rate_at_home': 0.75,
        'home_goals_per_game': 3.0,
        'away_goals_per_game': 1.67,
        'home_xg_per_game': 2.1,
        'away_xg_per_game': 1.2,
        'home_goals_conceded_per_game': 0.5,
        'away_goals_conceded_per_game': 2.0,
        'home_clean_sheet_rate': 0.5,
        'away_clean_sheet_rate': 0.0,
        'home_corners_per_game': 7.3,
        'away_corners_per_game': 4.0,
        'home_corners_conceded_per_game': 3.5,
        'away_corners_conceded_per_game': 6.0,
        'home_yellow_cards_per_game': 1.3,
        'away_yellow_cards_per_game': 1.7,
        'home_red_cards_per_season': 0.1,
        'away_red_cards_per_season': 0.3,
        'referee_cards_per_game': 4.5,
        'match_importance': 0.9,  # 淘汰赛
        'home_avg_possession': 63.5,
        'away_avg_possession': 44.0,
        'home_high_press_intensity': 8,
        'away_low_block_tendency': 9,
        'h2h_home_win_rate': 0.67,
        'h2h_btts_rate': 0.67,
        'h2h_avg_goals': 2.3,
        'home_key_players_out': 0,
        'away_key_players_out': 1,
        'home_first_15min_goal_rate': 0.5,
        'away_last_15min_goal_rate': 0.4,
        'first_half_goals_percentage': 0.45,
        'odds_home': 1.53,
        'odds_draw': 3.75,
        'odds_away': 6.5,
    }
    
    odds = {
        'home': 1.53,
        'draw': 3.75,
        'away': 6.5,
        'over_2_5': 1.90,
        'under_2_5': 1.98,
        'btts_yes': 1.61,
        'btts_no': 2.25,
        'corners_over_9_5': 1.92,
        'corners_under_9_5': 1.94,
        'cards_over_4_5': 1.89,
        'cards_under_4_5': 2.00,
    }
    
    # 使用启发式预测（因为没有训练数据）
    print("使用启发式模式预测（无训练数据）:")
    print("=" * 80)
    predictions = predictor.predict_all_markets(match_data, odds)
    
    # 输出报告
    report = predictor.export_predictions_to_report(predictions, match_data)
    print(report)
