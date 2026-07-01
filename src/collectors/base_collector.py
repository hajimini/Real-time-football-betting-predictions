"""
基础数据采集器抽象类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime
import asyncio
import aiohttp
from loguru import logger


class BaseCollector(ABC):
    """基础数据采集器"""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        self.api_url = api_url
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=self._get_headers()
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "FootballBettingSystem/1.0"
        }
        if self.api_key:
            headers["X-RapidAPI-Key"] = self.api_key
        return headers
    
    @abstractmethod
    async def fetch_data(self, **kwargs) -> Dict[str, Any]:
        """获取数据"""
        pass
    
    @abstractmethod
    async def process_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理原始数据"""
        pass
    
    async def collect(self, **kwargs) -> Dict[str, Any]:
        """采集完整流程"""
        try:
            raw_data = await self.fetch_data(**kwargs)
            processed_data = await self.process_data(raw_data)
            logger.info(f"数据采集成功：{self.__class__.__name__}")
            return processed_data
        except Exception as e:
            logger.error(f"数据采集失败：{e}")
            raise
    
    async def start_continuous_collection(self, interval: int = 10):
        """持续采集数据"""
        self.is_running = True
        logger.info(f"开始持续采集，间隔：{interval}秒")
        
        while self.is_running:
            try:
                await self.collect()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"采集循环错误：{e}")
                await asyncio.sleep(interval)
    
    def stop(self):
        """停止采集"""
        self.is_running = False
        logger.info("停止数据采集")
