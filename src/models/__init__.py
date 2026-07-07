"""
预测模型模块
包含各种机器学习/深度学习预测模型
"""

# 原有的滚球预测器
from .predictor import InPlayPredictor

# 新增的综合盘口预测器（支持 8 种盘口）
from .comprehensive_predictor import ComprehensiveBettingPredictor, PredictionResult

__all__ = [
    "InPlayPredictor",
    "ComprehensiveBettingPredictor",
    "PredictionResult"
]
