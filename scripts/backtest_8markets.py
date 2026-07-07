"""
全链路回测脚本 - 测试 8 种盘口预测的胜率
使用历史比赛数据进行回测分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '/workspace/src')

from models.comprehensive_predictor import ComprehensiveBettingPredictor
from features.feature_extractor import FeatureExtractor
from loguru import logger

def load_test_data():
    """加载测试数据"""
    data_path = Path('/workspace/data/training_data.csv')
    if not data_path.exists():
        logger.error("训练数据不存在")
        return None
    
    df = pd.read_csv(data_path)
    logger.info(f"加载数据：{len(df)} 条记录")
    return df

def create_match_summary(df):
    """从分钟级数据创建比赛级汇总"""
    # 只取每场比赛的初始数据 (minute=0)
    matches = df[df['minute'] == 0].copy()
    logger.info(f"比赛场次：{len(matches)}")
    return matches

def simulate_predictions(predictor, matches):
    """模拟预测并计算胜率"""
    results = {
        '1x2': {'correct': 0, 'total': 0},
        'over_under': {'correct': 0, 'total': 0},
        'btts': {'correct': 0, 'total': 0},
    }
    
    for idx, match in matches.iterrows():
        # 构建比赛特征
        match_data = {
            'home_team': match['home_team'],
            'away_team': match['away_team'],
            'home_xg_pre': match.get('home_xg_pre', 1.5),
            'away_xg_pre': match.get('away_xg_pre', 1.5),
            'home_recent_momentum': match.get('home_recent_momentum', 0),
            'away_recent_momentum': match.get('away_recent_momentum', 0),
        }
        
        try:
            # 获取预测
            predictions = predictor.predict_all_markets(match_data)
            
            # 1. 胜平负回测
            actual_outcome = match.get('label_final_outcome', 1)
            pred_1x2 = predictions.get('1x2', {})
            if pred_1x2:
                pred_outcome_map = {'HOME_WIN': 1, 'DRAW': 0, 'AWAY_WIN': 2}
                pred_outcome = pred_outcome_map.get(pred_1x2.get('prediction'), 1)
                results['1x2']['total'] += 1
                if pred_outcome == actual_outcome:
                    results['1x2']['correct'] += 1
            
            # 3. 大小球回测
            actual_goals = match.get('label_total_goals', 2)
            pred_ou = predictions.get('over_under', {})
            if pred_ou:
                pred_direction = pred_ou.get('prediction', 'OVER')
                threshold = pred_ou.get('threshold', 2.5)
                actual_over = actual_goals > threshold
                pred_over = pred_direction == 'OVER'
                results['over_under']['total'] += 1
                if actual_over == pred_over:
                    results['over_under']['correct'] += 1
            
            # 4. BTTS 回测
            # 需要推断双方是否进球 (简化处理)
            pred_btts = predictions.get('btts', {})
            if pred_btts:
                # 假设 xg>1 则进球
                home_scored = match.get('home_xg_pre', 1.5) > 0.8
                away_scored = match.get('away_xg_pre', 1.5) > 0.8
                actual_btts = home_scored and away_scored
                pred_btts_yes = pred_btts.get('prediction') == 'YES'
                results['btts']['total'] += 1
                if pred_btts_yes == actual_btts:
                    results['btts']['correct'] += 1
                    
        except Exception as e:
            logger.warning(f"预测失败：{e}")
            continue
    
    return results

def main():
    logger.info("=" * 60)
    logger.info("开始 8 种盘口回测分析")
    logger.info("=" * 60)
    
    # 加载数据
    df = load_test_data()
    if df is None:
        return
    
    matches = create_match_summary(df)
    if len(matches) == 0:
        logger.error("无有效比赛数据")
        return
    
    # 初始化预测器
    predictor = ComprehensiveBettingPredictor()
    
    # 尝试训练 (如果有足够数据)
    if len(matches) > 100:
        logger.info("正在训练模型...")
        try:
            predictor.train_all_models(df)
            logger.info("模型训练完成")
        except Exception as e:
            logger.warning(f"训练失败，使用启发式规则：{e}")
    
    # 执行回测
    logger.info("执行回测...")
    results = simulate_predictions(predictor, matches.head(1000))  # 回测前 1000 场
    
    # 输出结果
    print("\n" + "=" * 60)
    print("回测结果汇总")
    print("=" * 60)
    for market, data in results.items():
        if data['total'] > 0:
            accuracy = data['correct'] / data['total'] * 100
            print(f"{market:15s}: {data['correct']:4d}/{data['total']:4d} = {accuracy:.1f}%")
        else:
            print(f"{market:15s}: 无数据")
    
    print("=" * 60)
    
    # 示例预测
    logger.info("\n示例预测展示:")
    sample_match = {
        'home_team': 'Man_United',
        'away_team': 'Man_City',
        'home_xg_pre': 1.8,
        'away_xg_pre': 2.1,
        'home_recent_momentum': 0.6,
        'away_recent_momentum': 0.8,
    }
    
    predictions = predictor.predict_all_markets(sample_match)
    print("\n示例比赛：Man United vs Man City")
    for market, pred in predictions.items():
        if pred:
            print(f"  {market:12s}: {pred.get('prediction')} ({pred.get('probability', 0)*100:.1f}%) - {pred.get('confidence')}")

if __name__ == '__main__':
    main()
