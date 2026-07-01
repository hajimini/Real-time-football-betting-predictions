"""
预测模型模块
包含各种机器学习/深度学习预测模型
"""

# 使用我们新建的集成学习预测器
from .predictor import InPlayPredictor

__all__ = ["InPlayPredictor"]
