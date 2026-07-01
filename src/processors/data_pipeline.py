"""
数据处理流水线
"""

from typing import Dict, Any, List, Callable
from datetime import datetime
import json
import asyncio
from loguru import logger


class DataPipeline:
    """数据处理流水线"""
    
    def __init__(self):
        self.stages: List[Callable] = []
        self.output_handlers: List[Callable] = []
    
    def add_stage(self, stage: Callable) -> "DataPipeline":
        """添加处理阶段"""
        self.stages.append(stage)
        return self
    
    def add_output_handler(self, handler: Callable) -> "DataPipeline":
        """添加输出处理器"""
        self.output_handlers.append(handler)
        return self
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行数据处理流水线"""
        logger.info("开始数据处理流水线")
        start_time = datetime.now()
        
        current_data = data
        
        # 执行各个处理阶段
        for i, stage in enumerate(self.stages):
            try:
                logger.debug(f"执行阶段 {i+1}/{len(self.stages)}")
                if asyncio.iscoroutinefunction(stage):
                    current_data = await stage(current_data)
                else:
                    current_data = stage(current_data)
            except Exception as e:
                logger.error(f"阶段 {i+1} 处理失败：{e}")
                raise
        
        # 执行输出处理
        for handler in self.output_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(current_data)
                else:
                    handler(current_data)
            except Exception as e:
                logger.error(f"输出处理失败：{e}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"数据处理完成，耗时：{elapsed:.3f}秒")
        
        return current_data
    
    def run_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量处理数据（同步版本）"""
        results = []
        
        for data in data_list:
            current_data = data
            for stage in self.stages:
                current_data = stage(current_data)
            results.append(current_data)
        
        return results


class MatchDataPipeline(DataPipeline):
    """比赛数据处理流水线"""
    
    def __init__(self, cleaner=None, feature_engineer=None):
        super().__init__()
        self.cleaner = cleaner
        
        # 默认添加清洗阶段
        if self.cleaner:
            self.add_stage(self.cleaner.clean_match_data)
    
    def add_feature_extraction(self, feature_fn: Callable) -> "MatchDataPipeline":
        """添加特征提取阶段"""
        self.add_stage(feature_fn)
        return self
    
    def add_storage(self, storage_fn: Callable) -> "MatchDataPipeline":
        """添加存储阶段"""
        self.add_output_handler(storage_fn)
        return self


class OddsDataPipeline(DataPipeline):
    """赔率数据处理流水线"""
    
    def __init__(self, cleaner=None):
        super().__init__()
        self.cleaner = cleaner
        
        # 默认添加清洗阶段
        if self.cleaner:
            self.add_stage(self.cleaner.clean_odds_data)
    
    def add_odds_analysis(self, analysis_fn: Callable) -> "OddsDataPipeline":
        """添加赔率分析阶段"""
        self.add_stage(analysis_fn)
        return self
