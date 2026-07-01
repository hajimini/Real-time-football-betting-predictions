"""
FastAPI 应用主文件
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List
from loguru import logger
import uvicorn

from ..models.goal_predictor import GoalPredictor
from ..models.match_predictor import MatchOutcomePredictor
from ..features.match_features import MatchFeatureBuilder
from ..strategies.betting_strategy import ValueBettingStrategy
from ..strategies.bankroll_management import BankrollManager


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    
    app = FastAPI(
        title="足球滚球预测系统",
        description="Real-time Football In-Play Betting Prediction API",
        version="0.1.0"
    )
    
    # CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 初始化组件
    goal_predictor = GoalPredictor()
    match_predictor = MatchOutcomePredictor()
    feature_builder = MatchFeatureBuilder()
    betting_strategy = ValueBettingStrategy()
    bankroll_manager = BankrollManager(initial_balance=1000.0)
    
    @app.get("/")
    async def root():
        return {
            "message": "欢迎使用足球滚球预测系统 API",
            "version": "0.1.0",
            "endpoints": [
                "/health",
                "/predict/next-goal",
                "/predict/match-outcome",
                "/predict/all",
                "/bankroll/stats",
                "/matches/live"
            ]
        }
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}
    
    @app.post("/predict/next-goal")
    async def predict_next_goal(match_data: Dict[str, Any]):
        """预测下一粒进球"""
        try:
            features = feature_builder.extract(match_data)
            prediction = goal_predictor.predict_next_goal(features)
            return {"success": True, "prediction": prediction}
        except Exception as e:
            logger.error(f"预测失败：{e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict/match-outcome")
    async def predict_match_outcome(match_data: Dict[str, Any]):
        """预测比赛结果"""
        try:
            features = feature_builder.extract(match_data)
            prediction = match_predictor.predict(features)
            return {"success": True, "prediction": prediction}
        except Exception as e:
            logger.error(f"预测失败：{e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict/all")
    async def predict_all(match_data: Dict[str, Any], odds_data: Dict[str, float] = None):
        """获取所有预测和投注建议"""
        try:
            # 提取特征
            features = feature_builder.extract(match_data, odds_data)
            
            # 获取预测
            next_goal_pred = goal_predictor.predict_next_goal(features)
            total_goals_pred = goal_predictor.predict_total_goals(features)
            outcome_pred = match_predictor.predict(features)
            
            # 合并预测
            predictions = {
                "match_id": match_data.get("match_id"),
                **next_goal_pred,
                **total_goals_pred,
                **outcome_pred
            }
            
            # 生成投注建议
            bets = []
            if odds_data:
                generated_bets = betting_strategy.generate_bets(
                    predictions, 
                    odds_data, 
                    bankroll_manager.current_balance
                )
                bets = [bet.to_dict() for bet in generated_bets]
            
            return {
                "success": True,
                "predictions": predictions,
                "betting_suggestions": bets,
                "current_bankroll": bankroll_manager.current_balance
            }
        except Exception as e:
            logger.error(f"预测失败：{e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/bankroll/stats")
    async def get_bankroll_stats():
        """获取资金统计"""
        return bankroll_manager.get_stats()
    
    @app.post("/bankroll/bet")
    async def place_bet(bet: Dict[str, Any]):
        """下注"""
        success = bankroll_manager.place_bet(bet)
        if success:
            return {"success": True, "balance": bankroll_manager.current_balance}
        else:
            raise HTTPException(status_code=400, detail="下注失败")
    
    @app.get("/matches/live")
    async def get_live_matches():
        """获取 live 比赛（示例）"""
        # 实际应用中这里会从数据源获取
        return {
            "matches": []
        }
    
    @app.on_event("startup")
    async def startup_event():
        logger.info("应用启动")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("应用关闭")
    
    return app


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
