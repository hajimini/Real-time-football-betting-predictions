#!/usr/bin/env python3
"""
模型训练脚本
用于训练进球预测和比赛结果预测模型
"""

import sys
import os
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.goal_predictor import GoalPredictor
from src.models.match_predictor import MatchOutcomePredictor
from loguru import logger


def generate_synthetic_data(n_samples: int = 1000):
    """
    生成合成训练数据
    实际应用中应该使用真实历史数据
    """
    np.random.seed(42)
    
    # 特征
    elapsed_time = np.random.uniform(0, 90, n_samples)
    home_score = np.random.poisson(1.5, n_samples)
    away_score = np.random.poisson(1.2, n_samples)
    total_goals = home_score + away_score
    goal_difference = home_score - away_score
    
    home_shots = np.random.poisson(10, n_samples)
    away_shots = np.random.poisson(8, n_samples)
    home_shots_on_target = np.random.poisson(4, n_samples)
    away_shots_on_target = np.random.poisson(3, n_samples)
    
    home_possession = np.random.uniform(30, 70, n_samples)
    away_possession = 100 - home_possession
    
    home_corners = np.random.poisson(5, n_samples)
    away_corners = np.random.poisson(4, n_samples)
    
    home_attacks = np.random.poisson(50, n_samples)
    away_attacks = np.random.poisson(40, n_samples)
    
    time_remaining = 95 - elapsed_time
    time_ratio = elapsed_time / 95
    
    goals_per_10min = total_goals / (elapsed_time / 10 + 0.1)
    
    # 压力指数
    home_pressure = home_shots * 0.2 + home_shots_on_target * 0.3 + home_possession * 0.005
    away_pressure = away_shots * 0.2 + away_shots_on_target * 0.3 + away_possession * 0.005
    
    # 隐含概率（从赔率反推）
    implied_home_prob = np.random.uniform(0.2, 0.6, n_samples)
    implied_draw_prob = np.random.uniform(0.2, 0.4, n_samples)
    implied_away_prob = 1 - implied_home_prob - implied_draw_prob
    
    # 构建特征矩阵
    X = np.column_stack([
        elapsed_time, home_score, away_score, total_goals, goal_difference,
        home_shots, away_shots, home_shots_on_target, away_shots_on_target,
        home_possession, away_possession, home_corners, away_corners,
        home_attacks, away_attacks, time_remaining, time_ratio,
        goals_per_10min, home_pressure, away_pressure,
        implied_home_prob, implied_draw_prob, implied_away_prob
    ])
    
    # 目标变量：下一球（0=主队，1=客队，2=无进球）
    next_goal = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        r = np.random.random()
        if r < home_pressure / (home_pressure + away_pressure) * 0.6:
            next_goal[i] = 0  # 主队进球
        elif r < 0.7:
            next_goal[i] = 1  # 客队进球
        else:
            next_goal[i] = 2  # 无进球
    
    # 目标变量：最终总进球数
    remaining_ratio = time_remaining / 95
    expected_additional = goals_per_10min * remaining_ratio * 9
    final_total = total_goals + np.random.poisson(expected_additional, n_samples)
    
    # 目标变量：比赛结果（0=主胜，1=平局，2=客胜）
    match_outcome = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if home_score > away_score:
            if np.random.random() < 0.7:  # 保持领先概率
                match_outcome[i] = 0
            elif np.random.random() < 0.5:
                match_outcome[i] = 1
            else:
                match_outcome[i] = 2
        elif away_score > home_score:
            if np.random.random() < 0.7:
                match_outcome[i] = 2
            elif np.random.random() < 0.5:
                match_outcome[i] = 1
            else:
                match_outcome[i] = 0
        else:
            if np.random.random() < 0.5:
                match_outcome[i] = 1
            elif np.random.random() < 0.5:
                match_outcome[i] = 0
            else:
                match_outcome[i] = 2
    
    return X, next_goal, final_total, match_outcome


def main():
    """主训练流程"""
    logger.info("=" * 50)
    logger.info("开始训练模型")
    logger.info("=" * 50)
    
    # 生成训练数据
    logger.info("生成训练数据...")
    X, y_goal, y_total, y_outcome = generate_synthetic_data(5000)
    
    logger.info(f"训练数据形状：X={X.shape}, y_goal={y_goal.shape}, y_total={y_total.shape}")
    
    # 划分训练集和测试集
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_goal_train, y_goal_test = y_goal[:split_idx], y_goal[split_idx:]
    y_total_train, y_total_test = y_total[:split_idx], y_total[split_idx:]
    y_outcome_train, y_outcome_test = y_outcome[:split_idx], y_outcome[split_idx:]
    
    # 训练进球预测模型
    logger.info("\n训练进球预测模型...")
    goal_predictor = GoalPredictor()
    goal_results = goal_predictor.train(X_train, y_goal_train, y_total_train)
    logger.info(f"进球预测模型结果：{goal_results}")
    
    # 保存模型
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models")
    os.makedirs(model_path, exist_ok=True)
    
    goal_predictor.save_model(os.path.join(model_path, "goal_predictor.pkl"))
    
    # 训练比赛结果预测模型
    logger.info("\n训练比赛结果预测模型...")
    match_predictor = MatchOutcomePredictor()
    match_results = match_predictor.train(X_train, y_outcome_train)
    logger.info(f"比赛结果预测模型结果：{match_results}")
    
    match_predictor.save_model(os.path.join(model_path, "match_predictor.pkl"))
    
    # 评估模型
    logger.info("\n模型评估...")
    goal_accuracy = goal_predictor.next_goal_model.score(X_test, y_goal_test)
    total_r2 = goal_predictor.total_goals_model.score(X_test, y_total_test)
    outcome_accuracy = match_predictor.model.score(X_test, y_outcome_test)
    
    logger.info(f"进球预测准确率：{goal_accuracy:.4f}")
    logger.info(f"总进球 R²: {total_r2:.4f}")
    logger.info(f"比赛结果准确率：{outcome_accuracy:.4f}")
    
    logger.info("\n" + "=" * 50)
    logger.info("训练完成！")
    logger.info(f"模型已保存到：{model_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
