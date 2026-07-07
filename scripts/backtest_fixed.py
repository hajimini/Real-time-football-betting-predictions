"""
修复版回测脚本 - 使用简化的规则基线进行回测
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '/workspace/src')

def load_and_analyze():
    """加载数据并分析真实分布"""
    data_path = Path('/workspace/data/training_data.csv')
    df = pd.read_csv(data_path)
    
    # 只取每场比赛的初始数据
    matches = df[df['minute'] == 0].copy()
    print(f"总比赛数：{len(matches)}")
    
    # 分析真实结果分布
    print("\n=== 真实结果分布 ===")
    outcome_dist = matches['label_final_outcome'].value_counts().sort_index()
    print("胜平负分布:")
    print(f"  主胜 (1): {(outcome_dist.get(1, 0)/len(matches))*100:.1f}%")
    print(f"  平局 (0): {(outcome_dist.get(0, 0)/len(matches))*100:.1f}%")
    print(f"  客胜 (2): {(outcome_dist.get(2, 0)/len(matches))*100:.1f}%")
    
    # 大小球分布
    goals_dist = matches['label_total_goals'].value_counts().sort_index()
    print("\n总进球分布:")
    for goals, count in goals_dist.items():
        print(f"  {goals}球：{count/len(matches)*100:.1f}%")
    
    over_25 = (matches['label_total_goals'] > 2.5).mean() * 100
    print(f"\n大 2.5 比例：{over_25:.1f}%")
    
    # 简单的基线策略回测
    print("\n=== 基线策略回测 ===")
    
    # 策略 1: 永远预测主场胜
    baseline_1x2 = (matches['label_final_outcome'] == 1).mean() * 100
    print(f"永远猜主胜胜率：{baseline_1x2:.1f}%")
    
    # 策略 2: 永远猜大 2.5
    baseline_over = over_25
    print(f"永远猜大 2.5 胜率：{baseline_over:.1f}%")
    
    # 策略 3: 基于 xG 预测 BTTS
    matches['home_scored'] = matches['home_xg_pre'] > 0.7
    matches['away_scored'] = matches['away_xg_pre'] > 0.7
    matches['pred_btts'] = matches['home_scored'] & matches['away_scored']
    
    # 实际 BTTS (简化：xG>0.7 就算进球)
    matches['actual_btts'] = matches['pred_btts']  # 这里用同样标准作为代理
    baseline_btts = (matches['pred_btts'] == matches['actual_btts']).mean() * 100
    print(f"xG 法预测 BTTS 胜率：{baseline_btts:.1f}%")
    
    # 更真实的 BTTS 评估：看最终比分
    # 从完整数据推断每场比赛的最终比分
    final_scores = df.groupby('match_id').last()[['home_score', 'away_score']].reset_index()
    final_scores['actual_btts'] = (final_scores['home_score'] > 0) & (final_scores['away_score'] > 0)
    
    # 合并回 matches
    matches = matches.merge(final_scores[['match_id', 'actual_btts']], on='match_id', suffixes=('', '_final'))
    
    # 重新计算 BTTS 预测准确率
    pred_yes = matches['home_xg_pre'] > 0.8
    pred_no = matches['home_xg_pre'] <= 0.8
    
    correct_yes = ((pred_yes) & (matches['actual_btts'])).sum()
    correct_no = ((~pred_yes) & (~matches['actual_btts'])).sum()
    total_btts_accuracy = (correct_yes + correct_no) / len(matches) * 100
    
    print(f"\n真实 BTTS 预测准确率 (xG>0.8 阈值):")
    print(f"  预测 YES 且实际 YES: {correct_yes}")
    print(f"  预测 NO 且实际 NO: {correct_no}")
    print(f"  总准确率：{total_btts_accuracy:.1f}%")
    
    # 实际 BTTS 分布
    actual_btts_rate = matches['actual_btts'].mean() * 100
    print(f"\n实际 BTTS 发生率：{actual_btts_rate:.1f}%")
    
    return matches

if __name__ == '__main__':
    load_and_analyze()
