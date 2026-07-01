"""
特征工程模块 - 高级特征构造
用于滚球预测的深度特征工程
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class FeatureEngineer:
    """
    专业足球滚球特征工程类
    构造 50+ 维度的深度特征
    """
    
    def __init__(self):
        self.feature_groups = {
            'time_features': [],
            'score_features': [],
            'momentum_features': [],
            'xg_features': [],
            'pressure_features': [],
            'event_features': []
        }
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """时间相关特征"""
        df = df.copy()
        
        # 基础时间
        df['minute'] = df['minute'].fillna(0)
        df['time_remaining'] = 90 - df['minute']
        
        # 比赛阶段 (one-hot)
        df['early_game'] = (df['minute'] < 20).astype(int)
        df['first_half_mid'] = ((df['minute'] >= 20) & (df['minute'] < 45)).astype(int)
        df['half_time'] = ((df['minute'] >= 45) & (df['minute'] < 50)).astype(int)
        df['second_half_start'] = ((df['minute'] >= 50) & (df['minute'] < 60)).astype(int)
        df['mid_second'] = ((df['minute'] >= 60) & (df['minute'] < 75)).astype(int)
        df['late_game'] = ((df['minute'] >= 75) & (df['minute'] < 85)).astype(int)
        df['injury_time'] = (df['minute'] >= 85).astype(int)
        
        # 时间比例特征
        df['time_elapsed_pct'] = df['minute'] / 90.0
        df['time_remaining_pct'] = df['time_remaining'] / 90.0
        
        # 关键时间段标记
        df['is_critical_period'] = ((df['minute'] >= 75) | (df['minute'] <= 10)).astype(int)
        
        self.feature_groups['time_features'] = [
            'minute', 'time_remaining', 'early_game', 'first_half_mid',
            'half_time', 'second_half_start', 'mid_second', 'late_game',
            'injury_time', 'time_elapsed_pct', 'time_remaining_pct',
            'is_critical_period'
        ]
        
        return df
    
    def create_score_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """比分相关特征"""
        df = df.copy()
        
        # 基础比分
        df['home_score'] = df['home_score'].fillna(0).astype(int)
        df['away_score'] = df['away_score'].fillna(0).astype(int)
        df['goal_difference'] = df['home_score'] - df['away_score']
        df['total_goals'] = df['home_score'] + df['away_score']
        
        # 比分状态
        df['is_draw'] = (df['goal_difference'] == 0).astype(int)
        df['home_leading_by_1'] = (df['goal_difference'] == 1).astype(int)
        df['home_leading_by_2plus'] = (df['goal_difference'] >= 2).astype(int)
        df['away_leading_by_1'] = (df['goal_difference'] == -1).astype(int)
        df['away_leading_by_2plus'] = (df['goal_difference'] <= -2).astype(int)
        
        # 进球数分段
        df['low_scoring'] = (df['total_goals'] <= 1).astype(int)
        df['medium_scoring'] = ((df['total_goals'] > 1) & (df['total_goals'] <= 3)).astype(int)
        df['high_scoring'] = (df['total_goals'] > 3).astype(int)
        
        # 最近进球影响 (简化版，实际需要事件流)
        df['goals_last_15min'] = df['total_goals'].clip(0, 3)  # 简化
        
        self.feature_groups['score_features'] = [
            'home_score', 'away_score', 'goal_difference', 'total_goals',
            'is_draw', 'home_leading_by_1', 'home_leading_by_2plus',
            'away_leading_by_1', 'away_leading_by_2plus',
            'low_scoring', 'medium_scoring', 'high_scoring',
            'goals_last_15min'
        ]
        
        return df
    
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """势头/动量特征"""
        df = df.copy()
        
        # 基础势头
        df['home_recent_momentum'] = df.get('home_recent_momentum', pd.Series([0]*len(df))).fillna(0)
        df['away_recent_momentum'] = df.get('away_recent_momentum', pd.Series([0]*len(df))).fillna(0)
        df['momentum_diff'] = df['home_recent_momentum'] - df['away_recent_momentum']
        df['momentum_total'] = df['home_recent_momentum'] + df['away_recent_momentum']
        df['momentum_intensity'] = abs(df['momentum_diff'])
        
        # 势头方向
        df['home_dominant'] = (df['momentum_diff'] > 0).astype(int)
        df['away_dominant'] = (df['momentum_diff'] < 0).astype(int)
        df['balanced_play'] = (df['momentum_diff'] == 0).astype(int)
        
        # 势头强度分级
        df['strong_home_momentum'] = (df['momentum_diff'] >= 2).astype(int)
        df['strong_away_momentum'] = (df['momentum_diff'] <= -2).astype(int)
        
        self.feature_groups['momentum_features'] = [
            'home_recent_momentum', 'away_recent_momentum', 'momentum_diff',
            'momentum_total', 'momentum_intensity', 'home_dominant',
            'away_dominant', 'balanced_play', 'strong_home_momentum',
            'strong_away_momentum'
        ]
        
        return df
    
    def create_xg_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """期望进球 (xG) 相关特征"""
        df = df.copy()
        
        # 基础 xG
        df['home_xg_pre'] = df.get('home_xg_pre', pd.Series([1.5]*len(df))).fillna(1.5)
        df['away_xg_pre'] = df.get('away_xg_pre', pd.Series([1.2]*len(df))).fillna(1.2)
        df['xg_diff'] = df['home_xg_pre'] - df['away_xg_pre']
        df['total_xg'] = df['home_xg_pre'] + df['away_xg_pre']
        df['xg_ratio'] = df['home_xg_pre'] / (df['away_xg_pre'] + 0.1)
        
        # xG 与实时比分对比
        df['home_overperform_xg'] = df['home_score'] - df['home_xg_pre']
        df['away_overperform_xg'] = df['away_score'] - df['away_xg_pre']
        
        # xG 实力分级
        df['home_strong_xg'] = (df['home_xg_pre'] >= 1.8).astype(int)
        df['away_strong_xg'] = (df['away_xg_pre'] >= 1.5).astype(int)
        df['high_xg_match'] = (df['total_xg'] >= 3.0).astype(int)
        df['low_xg_match'] = (df['total_xg'] <= 2.0).astype(int)
        
        self.feature_groups['xg_features'] = [
            'home_xg_pre', 'away_xg_pre', 'xg_diff', 'total_xg', 'xg_ratio',
            'home_overperform_xg', 'away_overperform_xg',
            'home_strong_xg', 'away_strong_xg', 'high_xg_match', 'low_xg_match'
        ]
        
        return df
    
    def create_pressure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """压力/紧迫感特征"""
        df = df.copy()
        
        # 基于比分差和剩余时间的压力
        goal_diff = df['goal_difference']
        time_rem = df['time_remaining']
        
        # 落后方压力指数 (使用 np.where 替代三元运算符)
        time_factor = np.where(time_rem > 20, 1.0, 1.5)
        df['home_pressure'] = np.where(
            goal_diff < 0,
            time_rem / (abs(goal_diff) + 1) * time_factor,
            0
        )
        df['away_pressure'] = np.where(
            goal_diff > 0,
            time_rem / (abs(goal_diff) + 1) * time_factor,
            0
        )
        
        # 平局时的紧张度 (随时间增加)
        df['draw_tension'] = np.where(
            df['is_draw'] == 1,
            (90 - df['minute']) / 30,  # 越接近结束越紧张
            0
        )
        
        # 一方领先但优势小的压力
        df['narrow_lead_pressure_home'] = np.where(
            (goal_diff == 1) & (time_rem > 15),
            time_rem / 45,
            0
        )
        df['narrow_lead_pressure_away'] = np.where(
            (goal_diff == -1) & (time_rem > 15),
            time_rem / 45,
            0
        )
        
        # 需要进球的压力 (结合 xG)
        df['home_need_goal'] = np.where(
            (goal_diff < 0) & (df['home_xg_pre'] > df['away_xg_pre']),
            1,
            0
        )
        
        self.feature_groups['pressure_features'] = [
            'home_pressure', 'away_pressure', 'draw_tension',
            'narrow_lead_pressure_home', 'narrow_lead_pressure_away',
            'home_need_goal'
        ]
        
        return df
    
    def create_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """事件相关特征 (红牌、角球等)"""
        df = df.copy()
        
        # 红牌
        df['home_red_card'] = df.get('home_red_card', pd.Series([0]*len(df))).fillna(0).astype(int)
        df['away_red_card'] = df.get('away_red_card', pd.Series([0]*len(df))).fillna(0).astype(int)
        df['any_red_card'] = ((df['home_red_card'] == 1) | (df['away_red_card'] == 1)).astype(int)
        df['red_card_diff'] = df['away_red_card'] - df['home_red_card']  # 正数表示主队人数优势
        
        # 人数优势影响
        df['home_man_advantage'] = (df['red_card_diff'] > 0).astype(int)
        df['away_man_advantage'] = (df['red_card_diff'] < 0).astype(int)
        
        # 红牌 + 时间组合
        df['late_red_card'] = ((df['any_red_card'] == 1) & (df['minute'] >= 60)).astype(int)
        
        self.feature_groups['event_features'] = [
            'home_red_card', 'away_red_card', 'any_red_card',
            'red_card_diff', 'home_man_advantage', 'away_man_advantage',
            'late_red_card'
        ]
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建所有特征"""
        df = self.create_time_features(df)
        df = self.create_score_features(df)
        df = self.create_momentum_features(df)
        df = self.create_xg_features(df)
        df = self.create_pressure_features(df)
        df = self.create_event_features(df)
        
        return df
    
    def get_all_feature_names(self) -> List[str]:
        """获取所有特征名称"""
        all_features = []
        for group in self.feature_groups.values():
            all_features.extend(group)
        return all_features
    
    def get_feature_importance_summary(self, importance_df: pd.DataFrame) -> pd.DataFrame:
        """生成特征重要性分组汇总"""
        importance_df = importance_df.copy()
        
        # 添加特征分组
        def assign_group(feature):
            for group_name, features in self.feature_groups.items():
                if feature in features:
                    return group_name
            return 'other'
        
        importance_df['group'] = importance_df['feature'].apply(assign_group)
        
        # 分组汇总
        summary = importance_df.groupby('group')['importance'].agg([
            ('count', 'count'),
            ('sum', 'sum'),
            ('mean', 'mean'),
            ('max', 'max')
        ]).round(4)
        
        return summary


if __name__ == "__main__":
    # 测试特征工程
    print("测试特征工程模块...")
    
    # 创建测试数据
    test_data = pd.DataFrame([{
        'minute': 65,
        'home_score': 1,
        'away_score': 1,
        'home_xg_pre': 1.8,
        'away_xg_pre': 1.3,
        'home_recent_momentum': 2,
        'away_recent_momentum': 1,
        'home_red_card': 0,
        'away_red_card': 0
    }])
    
    engineer = FeatureEngineer()
    result = engineer.create_all_features(test_data)
    
    print(f"\n原始特征：{list(test_data.columns)}")
    print(f"\n生成特征数量：{len(engineer.get_all_feature_names())}")
    print(f"\n部分生成的特征:")
    display_cols = ['minute', 'time_remaining', 'goal_difference', 'total_goals',
                    'momentum_diff', 'xg_diff', 'home_pressure', 'away_pressure',
                    'draw_tension', 'red_card_diff']
    
    for col in display_cols:
        if col in result.columns:
            print(f"  {col}: {result[col].values[0]:.4f}")
    
    print(f"\n特征分组统计:")
    for group, features in engineer.feature_groups.items():
        print(f"  {group}: {len(features)} 个特征")
