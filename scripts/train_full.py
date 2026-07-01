"""
完整训练脚本 - 生成数据、训练模型、保存模型
用于学术研究和学习的滚球预测系统
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_simulator import MatchSimulator
from src.models.predictor import InPlayPredictor
from src.features.engineering import FeatureEngineer


def main():
    print("=" * 70)
    print("足球滚球预测系统 - 完整训练流程")
    print("=" * 70)
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # ==================== 步骤 1: 数据生成 ====================
    print("【步骤 1/4】生成模拟比赛数据...")
    print("-" * 50)
    
    simulator = MatchSimulator(seed=42)
    
    # 生成更多比赛数据以提高模型质量
    num_matches = 2000
    matches = simulator.generate_season_data(num_matches)
    print(f"✓ 生成 {num_matches} 场模拟比赛")
    
    # 创建训练数据集
    df = simulator.create_training_dataset(matches)
    print(f"✓ 生成 {len(df)} 个训练样本 (每 5 分钟一个快照)")
    
    # 保存原始数据
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/training_data.csv', index=False)
    print(f"✓ 数据已保存到 data/training_data.csv")
    
    # 数据统计
    print(f"\n数据概览:")
    print(f"  - 平均每场进球：{df['label_total_goals'].mean():.2f}")
    print(f"  - 主胜比例：{(df['label_final_outcome']==1).mean():.2%}")
    print(f"  - 平局比例：{(df['label_final_outcome']==0).mean():.2%}")
    print(f"  - 客胜比例：{(df['label_final_outcome']==-1).mean():.2%}")
    print(f"  - 有进球样本比例：{df['label_next_goal_15min'].mean():.2%}")
    
    # ==================== 步骤 2: 特征工程 ====================
    print("\n【步骤 2/4】特征工程...")
    print("-" * 50)
    
    engineer = FeatureEngineer()
    feature_df = engineer.create_all_features(df)
    
    all_features = engineer.get_all_feature_names()
    print(f"✓ 构造 {len(all_features)} 维特征")
    print(f"✓ 特征分组:")
    for group, features in engineer.feature_groups.items():
        print(f"    - {group}: {len(features)} 个特征")
    
    # ==================== 步骤 3: 模型训练 ====================
    print("\n【步骤 3/4】训练机器学习模型...")
    print("-" * 50)
    
    predictor = InPlayPredictor()
    
    # 使用增强特征训练
    predictor.train_all(feature_df)
    
    # ==================== 步骤 4: 模型保存 ====================
    print("\n【步骤 4/4】保存模型和配置...")
    print("-" * 50)
    
    os.makedirs('models/saved', exist_ok=True)
    
    # 保存模型
    with open('models/saved/goal_model.pkl', 'wb') as f:
        pickle.dump(predictor.goal_model, f)
    print("✓ 进球预测模型已保存")
    
    with open('models/saved/outcome_model.pkl', 'wb') as f:
        pickle.dump(predictor.outcome_model, f)
    print("✓ 比赛结果预测模型已保存")
    
    with open('models/saved/total_goals_model.pkl', 'wb') as f:
        pickle.dump(predictor.total_goals_model, f)
    print("✓ 总进球数预测模型已保存")
    
    # 保存特征列
    with open('models/saved/feature_columns.pkl', 'wb') as f:
        pickle.dump(predictor.feature_columns, f)
    print("✓ 特征列配置已保存")
    
    # 保存特征重要性
    goal_importance = predictor.get_feature_importance('goal')
    if goal_importance is not None:
        goal_importance.to_csv('models/saved/goal_feature_importance.csv', index=False)
        print("✓ 进球预测特征重要性已保存")
        
        print(f"\n Top 10 重要特征 (进球预测):")
        for idx, row in goal_importance.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
    
    outcome_importance = predictor.get_feature_importance('outcome')
    if outcome_importance is not None:
        outcome_importance.to_csv('models/saved/outcome_feature_importance.csv', index=False)
        print("✓ 比赛结果预测特征重要性已保存")
    
    # ==================== 测试验证 ====================
    print("\n【验证】测试模型预测功能...")
    print("-" * 50)
    
    test_scenarios = [
        {
            'name': '开场均势',
            'state': {
                'minute': 10,
                'home_score': 0,
                'away_score': 0,
                'home_xg_pre': 1.6,
                'away_xg_pre': 1.4,
                'home_recent_momentum': 1,
                'away_recent_momentum': 1,
                'home_red_card': 0,
                'away_red_card': 0
            }
        },
        {
            'name': '主队领先',
            'state': {
                'minute': 55,
                'home_score': 2,
                'away_score': 0,
                'home_xg_pre': 1.8,
                'away_xg_pre': 1.0,
                'home_recent_momentum': 2,
                'away_recent_momentum': 0,
                'home_red_card': 0,
                'away_red_card': 0
            }
        },
        {
            'name': '客队绝杀机会',
            'state': {
                'minute': 82,
                'home_score': 1,
                'away_score': 1,
                'home_xg_pre': 1.2,
                'away_xg_pre': 1.5,
                'home_recent_momentum': 0,
                'away_recent_momentum': 3,
                'home_red_card': 1,
                'away_red_card': 0
            }
        },
        {
            'name': '对攻大战',
            'state': {
                'minute': 70,
                'home_score': 2,
                'away_score': 2,
                'home_xg_pre': 2.2,
                'away_xg_pre': 2.0,
                'home_recent_momentum': 2,
                'away_recent_momentum': 2,
                'home_red_card': 0,
                'away_red_card': 0
            }
        }
    ]
    
    print()
    for scenario in test_scenarios:
        result = predictor.predict_match_state(scenario['state'])
        print(f"场景：{scenario['name']} (第{result['minute']}分钟，比分{result['current_score']})")
        print(f"  进球概率 (15min): {result['predictions']['goal_in_15min_probability']:.1%}")
        print(f"  胜平负：主{result['predictions']['home_win_probability']:.1%} "
              f"平{result['predictions']['draw_probability']:.1%} "
              f"客{result['predictions']['away_win_probability']:.1%}")
        print(f"  预期总进球：{result['predictions']['expected_total_goals']:.2f}")
        if result['recommendations']:
            print(f"  推荐：{result['recommendations'][0]['market']} -> {result['recommendations'][0]['prediction']} "
                  f"(信心:{result['recommendations'][0]['confidence']})")
        print()
    
    # ==================== 完成 ====================
    print("=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"\n模型文件位置：models/saved/")
    print(f"训练数据位置：data/training_data.csv")
    print(f"结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n⚠️  重要说明:")
    print("   本系统仅供学术研究和个人学习使用")
    print("   不保证预测准确性，不构成任何投注建议")
    print("   实际投注存在风险，请理性对待")
    
    return predictor


if __name__ == "__main__":
    main()
