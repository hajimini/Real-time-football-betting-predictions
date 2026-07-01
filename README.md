# 足球滚球预测系统 (In-Play Football Prediction System)

一套基于机器学习的实时足球滚球预测系统，专为学术研究和教学演示设计。

## ⚠️ 免责声明

**本系统仅供学术研究和个人学习使用，不对外提供任何商业服务或投注建议。**

- 所有数据均为模拟生成，非真实比赛数据
- 模型预测结果不保证准确性
- 请勿将本系统用于实际投注或商业用途
- 足球比赛具有高度不确定性，任何预测都存在风险

---

## 核心算法

### 1. 数据生成 (泊松过程)
进球模拟基于泊松分布：P(X=k) = λ^k * e^(-λ) / k!
其中 λ = 球队进攻强度 × 对手防守强度 × 主客场优势

### 2. 特征工程 (59 维)
- 时间特征 (12 维): 比赛阶段、剩余时间、关键时段标记
- 比分特征 (13 维): 进球差、领先状态、进球数分段
- 动量特征 (10 维): 近期势头、主导方、强度分级
- xG 特征 (11 维): 期望进球差、实力对比、超常发挥
- 压力特征 (6 维): 落后方压力、平局紧张度
- 事件特征 (7 维): 红牌、人数优势

### 3. 机器学习模型
采用 Stacking 集成学习架构:
- Goal Model: XGBoost + LightGBM + RF (Voting) - 预测未来 15 分钟是否有进球
- Outcome Model: XGBoost (多分类) - 预测最终胜平负
- Total Goals Model: XGBoost Regressor - 预测总进球数

所有分类模型经过 Platt Scaling 概率校准，输出可靠的概率值。

---

## 快速开始

### 安装依赖
```bash
pip install numpy pandas scikit-learn xgboost lightgbm
```

### 训练模型
```bash
python scripts/train_full.py
```

### 使用模型
```python
import pickle
from src.models.predictor import InPlayPredictor

# 加载模型
with open('models/saved/goal_model.pkl', 'rb') as f:
    goal_model = pickle.load(f)

predictor = InPlayPredictor()
predictor.goal_model = goal_model
predictor.models_trained = True

# 预测
match_state = {
    'minute': 65, 'home_score': 1, 'away_score': 1,
    'home_xg_pre': 1.8, 'away_xg_pre': 1.3,
    'home_recent_momentum': 2, 'away_recent_momentum': 1,
    'home_red_card': 0, 'away_red_card': 0
}
result = predictor.predict_match_state(match_state)
print(result['predictions'])
```

---

## 模型表现

| 模型 | 准确率 | AUC-ROC |
|------|--------|---------|
| 进球预测 | 76.0% | 0.814 |
| 比赛结果 | 73.3% | - |
| 总进球数 | - | RMSE: 0.89 |

---

## 技术栈
Python 3.8+, NumPy, Pandas, Scikit-learn, XGBoost, LightGBM

---

**本项目为个人作业，仅供学习研究使用。理性看待预测，享受足球本身。**
