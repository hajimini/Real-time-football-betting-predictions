"""
滚球预测核心模型
使用集成学习方法 (XGBoost + LightGBM) 进行多任务预测:
1. 下一粒进球时间预测
2. 比赛最终结果预测 (胜/平/负)
3. 总进球数预测 (大/小球)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


class InPlayPredictor:
    """
    滚球预测主模型
    集成多个基学习器，输出概率校准后的预测结果
    """
    
    def __init__(self):
        self.goal_model = None  # 预测是否有进球
        self.outcome_model = None  # 预测比赛结果
        self.total_goals_model = None  # 预测总进球数
        self.feature_columns = None
        self.models_trained = False
        
    def _create_xgb_model(self, objective='binary:logistic'):
        """创建 XGBoost 模型配置"""
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective=objective,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    
    def _create_lgb_model(self, objective='binary'):
        """创建 LightGBM 模型配置"""
        return lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective=objective,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    
    def prepare_features(self, df):
        """
        特征工程
        从原始数据中提取和构造预测特征
        """
        feature_df = df.copy()
        
        # 基础特征
        base_features = [
            'minute',
            'home_score',
            'away_score',
            'goal_difference',
            'home_xg_pre',
            'away_xg_pre',
            'home_recent_momentum',
            'away_recent_momentum',
            'momentum_diff',
            'home_red_card',
            'away_red_card',
            'time_remaining',
            'total_goals_so_far'
        ]
        
        # 构造衍生特征 (避免重复)
        feature_df['xg_diff'] = feature_df['home_xg_pre'] - feature_df['away_xg_pre']
        
        # 比分状态特征
        feature_df['is_draw'] = (feature_df['goal_difference'] == 0).astype(int)
        feature_df['home_leading'] = (feature_df['goal_difference'] > 0).astype(int)
        feature_df['away_leading'] = (feature_df['goal_difference'] < 0).astype(int)
        
        # 时间阶段特征
        feature_df['early_game'] = (feature_df['minute'] < 30).astype(int)
        feature_df['mid_game'] = ((feature_df['minute'] >= 30) & (feature_df['minute'] < 60)).astype(int)
        feature_df['late_game'] = (feature_df['minute'] >= 60).astype(int)
        feature_df['injury_time_risk'] = (feature_df['minute'] >= 75).astype(int)
        
        # 进球压力特征 (基于比分和剩余时间)
        feature_df['home_pressure'] = np.where(
            feature_df['goal_difference'] < 0,
            (90 - feature_df['minute']) / (abs(feature_df['goal_difference']) + 1),
            0
        )
        feature_df['away_pressure'] = np.where(
            feature_df['goal_difference'] > 0,
            (90 - feature_df['minute']) / (abs(feature_df['goal_difference']) + 1),
            0
        )
        
        #  momentum 变化率
        feature_df['momentum_intensity'] = abs(feature_df['momentum_diff'])
        
        # 总进球期望
        feature_df['total_xg'] = feature_df['home_xg_pre'] + feature_df['away_xg_pre']
        
        # 更新特征列表 (只包含最终需要的特征)
        self.feature_columns = [
            'minute',
            'home_score',
            'away_score',
            'goal_difference',
            'home_xg_pre',
            'away_xg_pre',
            'xg_diff',
            'home_recent_momentum',
            'away_recent_momentum',
            'momentum_diff',
            'home_red_card',
            'away_red_card',
            'time_remaining',
            'total_goals_so_far',
            'is_draw',
            'home_leading',
            'away_leading',
            'early_game',
            'mid_game',
            'late_game',
            'injury_time_risk',
            'home_pressure',
            'away_pressure',
            'momentum_intensity',
            'total_xg'
        ]
        
        return feature_df
    
    def train_goal_model(self, df):
        """
        训练进球预测模型
        预测未来 15 分钟内是否会有进球
        """
        print("正在训练进球预测模型...")
        
        feature_df = self.prepare_features(df)
        X = feature_df[self.feature_columns]
        y = feature_df['label_next_goal_15min']
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 创建集成模型 (XGBoost + LightGBM + Random Forest)
        xgb_clf = self._create_xgb_model()
        lgb_clf = self._create_lgb_model()
        rf_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        
        # Voting 集成
        ensemble = VotingClassifier(
            estimators=[('xgb', xgb_clf), ('lgb', lgb_clf), ('rf', rf_clf)],
            voting='soft'
        )
        
        # 训练
        ensemble.fit(X_train, y_train)
        
        # 概率校准 (使用 Platt Scaling)
        calibrated_model = CalibratedClassifierCV(ensemble, method='sigmoid', cv=3)
        calibrated_model.fit(X_test, y_test)
        
        # 评估
        y_pred = calibrated_model.predict(X_test)
        y_proba = calibrated_model.predict_proba(X_test)[:, 1]
        
        print(f"进球预测模型评估:")
        print(f"  准确率：{accuracy_score(y_test, y_pred):.4f}")
        print(f"  精确率：{precision_score(y_test, y_pred):.4f}")
        print(f"  召回率：{recall_score(y_test, y_pred):.4f}")
        print(f"  F1 分数：{f1_score(y_test, y_pred):.4f}")
        print(f"  AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
        
        self.goal_model = calibrated_model
        return calibrated_model
    
    def train_outcome_model(self, df):
        """
        训练比赛结果预测模型
        预测最终胜平负 (多分类)
        """
        print("\n正在训练比赛结果预测模型...")
        
        feature_df = self.prepare_features(df)
        X = feature_df[self.feature_columns]
        # 转换为 0, 1, 2 (平局，主胜，客胜)
        y = feature_df['label_final_outcome'].map({0: 0, 1: 1, -1: 2}).fillna(0).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # XGBoost 多分类
        xgb_multi = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )
        
        xgb_multi.fit(X_train, y_train)
        
        # 评估
        y_pred = xgb_multi.predict(X_test)
        y_proba = xgb_multi.predict_proba(X_test)
        
        print(f"比赛结果预测模型评估:")
        print(f"  准确率：{accuracy_score(y_test, y_pred):.4f}")
        print(f"  宏平均 F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
        
        self.outcome_model = xgb_multi
        return xgb_multi
    
    def train_total_goals_model(self, df):
        """
        训练总进球数预测模型
        回归问题：预测最终总进球数
        """
        print("\n正在训练总进球数预测模型...")
        
        feature_df = self.prepare_features(df)
        X = feature_df[self.feature_columns]
        y = feature_df['label_total_goals']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # XGBoost 回归
        xgb_reg = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_reg.fit(X_train, y_train)
        
        # 评估
        y_pred = xgb_reg.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"总进球数预测模型评估:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {xgb_reg.score(X_test, y_test):.4f}")
        
        self.total_goals_model = xgb_reg
        return xgb_reg
    
    def train_all(self, df):
        """训练所有模型"""
        self.train_goal_model(df)
        self.train_outcome_model(df)
        self.train_total_goals_model(df)
        self.models_trained = True
        print("\n✓ 所有模型训练完成!")
    
    def predict_match_state(self, match_state: dict) -> dict:
        """
        对单场比赛状态进行实时预测
        
        参数:
            match_state: 字典，包含当前比赛状态
                - minute: 当前分钟
                - home_score: 主队比分
                - away_score: 客队比分
                - home_xg_pre: 主队赛前期望进球
                - away_xg_pre: 客队赛前期望进球
                - home_recent_momentum: 主队近期势头
                - away_recent_momentum: 客队近期势头
                - home_red_card: 主队红牌 (0/1)
                - away_red_card: 客队红牌 (0/1)
                
        返回:
            预测结果字典
        """
        if not self.models_trained:
            raise ValueError("模型尚未训练，请先调用 train_all()")
        
        # 转换为 DataFrame 并计算缺失的特征
        df = pd.DataFrame([match_state])
        df['goal_difference'] = df['home_score'] - df['away_score']
        df['momentum_diff'] = df['home_recent_momentum'] - df['away_recent_momentum']
        df['time_remaining'] = 90 - df['minute']
        df['total_goals_so_far'] = df['home_score'] + df['away_score']
        
        feature_df = self.prepare_features(df)
        X = feature_df[self.feature_columns]
        
        # 进球概率预测
        goal_proba = self.goal_model.predict_proba(X)[0][1]
        
        # 比赛结果预测
        outcome_proba = self.outcome_model.predict_proba(X)[0]
        home_win_prob = outcome_proba[1]
        draw_prob = outcome_proba[0]
        away_win_prob = outcome_proba[2]
        
        # 总进球数预测
        expected_total_goals = self.total_goals_model.predict(X)[0]
        
        # 计算价值投注机会 (简化版)
        recommendations = []
        
        # 检查进球投注价值
        if goal_proba > 0.65:  # 高概率进球
            recommendations.append({
                'market': 'Next Goal (15min)',
                'prediction': 'YES',
                'probability': goal_proba,
                'confidence': 'HIGH' if goal_proba > 0.75 else 'MEDIUM'
            })
        elif goal_proba < 0.35:
            recommendations.append({
                'market': 'Next Goal (15min)',
                'prediction': 'NO',
                'probability': 1 - goal_proba,
                'confidence': 'HIGH' if goal_proba < 0.25 else 'MEDIUM'
            })
        
        # 检查胜平负价值
        max_prob = max(home_win_prob, draw_prob, away_win_prob)
        if max_prob > 0.6:
            if home_win_prob == max_prob:
                pred = 'HOME_WIN'
            elif draw_prob == max_prob:
                pred = 'DRAW'
            else:
                pred = 'AWAY_WIN'
            
            recommendations.append({
                'market': 'Match Result',
                'prediction': pred,
                'probability': max_prob,
                'confidence': 'HIGH' if max_prob > 0.7 else 'MEDIUM'
            })
        
        # 大小球推荐
        if expected_total_goals > 3.5:
            recommendations.append({
                'market': 'Total Goals',
                'prediction': 'OVER 3.5',
                'expected_value': expected_total_goals,
                'confidence': 'MEDIUM'
            })
        elif expected_total_goals < 1.5:
            recommendations.append({
                'market': 'Total Goals',
                'prediction': 'UNDER 1.5',
                'expected_value': expected_total_goals,
                'confidence': 'MEDIUM'
            })
        
        return {
            'minute': match_state['minute'],
            'current_score': f"{match_state['home_score']}-{match_state['away_score']}",
            'predictions': {
                'goal_in_15min_probability': round(goal_proba, 4),
                'home_win_probability': round(home_win_prob, 4),
                'draw_probability': round(draw_prob, 4),
                'away_win_probability': round(away_win_prob, 4),
                'expected_total_goals': round(expected_total_goals, 2)
            },
            'recommendations': recommendations,
            'model_confidence': 'CALIBRATED'
        }
    
    def get_feature_importance(self, model_name='goal') -> pd.DataFrame:
        """获取特征重要性"""
        if model_name == 'goal' and self.goal_model is not None:
            # CalibratedClassifierCV 包装了集成模型，需要访问内部
            try:
                # 尝试从校准后的模型中获取基础估计器
                base_estimator = self.goal_model.calibrated_classifiers_[0].estimator
                importances = []
                
                if hasattr(base_estimator, 'named_estimators_'):
                    # VotingClassifier
                    for name, clf in base_estimator.named_estimators_.items():
                        if hasattr(clf, 'feature_importances_'):
                            importances.append(clf.feature_importances_)
                elif hasattr(base_estimator, 'feature_importances_'):
                    importances.append(base_estimator.feature_importances_)
                
                if importances:
                    avg_importance = np.mean(importances, axis=0)
                    return pd.DataFrame({
                        'feature': self.feature_columns,
                        'importance': avg_importance
                    }).sort_values('importance', ascending=False)
            except Exception as e:
                print(f"Warning: Could not extract goal model importance: {e}")
                return None
        
        elif model_name == 'outcome' and self.outcome_model is not None:
            return pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.outcome_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        elif model_name == 'total_goals' and self.total_goals_model is not None:
            return pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.total_goals_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return None


if __name__ == "__main__":
    # 加载训练数据
    print("加载训练数据...")
    df = pd.read_csv('data/training_data.csv')
    
    # 初始化并训练模型
    predictor = InPlayPredictor()
    predictor.train_all(df)
    
    # 测试预测
    print("\n" + "="*60)
    print("测试实时预测功能")
    print("="*60)
    
    test_cases = [
        {
            'minute': 25,
            'home_score': 0,
            'away_score': 0,
            'home_xg_pre': 1.8,
            'away_xg_pre': 1.2,
            'home_recent_momentum': 2,
            'away_recent_momentum': 0,
            'home_red_card': 0,
            'away_red_card': 0
        },
        {
            'minute': 75,
            'home_score': 1,
            'away_score': 1,
            'home_xg_pre': 1.5,
            'away_xg_pre': 1.5,
            'home_recent_momentum': 1,
            'away_recent_momentum': 1,
            'home_red_card': 0,
            'away_red_card': 1
        },
        {
            'minute': 60,
            'home_score': 0,
            'away_score': 2,
            'home_xg_pre': 2.0,
            'away_xg_pre': 1.0,
            'home_recent_momentum': 3,
            'away_recent_momentum': 0,
            'home_red_card': 0,
            'away_red_card': 0
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试场景 {i}: 第{case['minute']}分钟，比分{case['home_score']}-{case['away_score']}")
        result = predictor.predict_match_state(case)
        
        print(f"  进球概率 (15 分钟内): {result['predictions']['goal_in_15min_probability']:.2%}")
        print(f"  胜平负概率：主{result['predictions']['home_win_probability']:.2%} "
              f"平{result['predictions']['draw_probability']:.2%} "
              f"客{result['predictions']['away_win_probability']:.2%}")
        print(f"  预期总进球：{result['predictions']['expected_total_goals']:.2f}")
        
        if result['recommendations']:
            print("  推荐投注:")
            for rec in result['recommendations']:
                print(f"    - {rec['market']}: {rec['prediction']} "
                      f"(概率:{rec['probability']:.2%}, 信心:{rec['confidence']})")
