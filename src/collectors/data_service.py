"""
数据采集服务

统一接口，管理数据采集流程：
1. 从配置文件加载配置
2. 创建对应的采集器
3. 定时获取实时比赛数据
4. 保存数据到本地或数据库
5. 提供给特征工程模块使用
"""

import asyncio
import yaml
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

from src.collectors.data_collector import (
    CollectorFactory,
    BaseCollector,
    MatchData
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataService:
    """
    数据采集服务
    
    职责：
    1. 加载配置文件
    2. 初始化采集器
    3. 定时轮询获取数据
    4. 数据持久化
    5. 提供数据查询接口
    """
    
    def __init__(self, config_path: str = "configs/collector_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.collector: Optional[BaseCollector] = None
        self.matches_cache: Dict[str, MatchData] = {}
        self.data_dir = Path(self.config.get('storage', {}).get('data_dir', './data/raw'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"数据服务初始化完成，数据源：{self.config.get('collector_type')}")
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    async def initialize(self):
        """初始化采集器"""
        collector_type = self.config.get('collector_type', 'simulator')
        
        if collector_type == 'simulator':
            config = self.config.get('simulator', {})
        elif collector_type == 'api':
            config = self.config.get('api', {})
        elif collector_type == 'scraper':
            config = self.config.get('scraper', {})
        else:
            raise ValueError(f"不支持的采集器类型：{collector_type}")
        
        self.collector = CollectorFactory.create_collector(collector_type, config)
        logger.info(f"采集器初始化成功：{collector_type}")
    
    async def fetch_and_save_matches(self):
        """获取并保存比赛数据"""
        if not self.collector:
            await self.initialize()
        
        try:
            async with self.collector:
                matches = await self.collector.fetch_live_matches()
                
                logger.info(f"获取到 {len(matches)} 场实时比赛")
                
                # 更新缓存
                for match in matches:
                    self.matches_cache[match.match_id] = match
                
                # 保存到文件
                await self._save_matches(matches)
                
                return matches
        
        except Exception as e:
            logger.error(f"获取比赛数据失败：{e}")
            return []
    
    async def _save_matches(self, matches: List[MatchData]):
        """保存比赛数据到文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存为 JSON 文件
        data = [m.to_dict() for m in matches]
        filename = f"matches_{timestamp}.json"
        filepath = self.data_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"数据已保存：{filepath}")
        
        # 清理旧数据（保留最近 N 天）
        await self._cleanup_old_data()
    
    async def _cleanup_old_data(self):
        """清理过期数据"""
        retention_days = self.config.get('storage', {}).get('retention_days', 30)
        cutoff_date = datetime.now().timestamp() - (retention_days * 24 * 60 * 60)
        
        for file in self.data_dir.glob("matches_*.json"):
            if file.stat().st_mtime < cutoff_date:
                file.unlink()
                logger.debug(f"删除过期文件：{file}")
    
    def get_match(self, match_id: str) -> Optional[MatchData]:
        """从缓存获取指定比赛"""
        return self.matches_cache.get(match_id)
    
    def get_all_matches(self) -> List[MatchData]:
        """获取所有缓存的比赛"""
        return list(self.matches_cache.values())
    
    async def run_continuous(self, interval_seconds: int = 10):
        """
        持续运行模式
        
        每隔指定时间获取一次数据，适用于实时监控
        """
        logger.info(f"启动持续数据采集，间隔：{interval_seconds}秒")
        
        iteration = 0
        while True:
            try:
                iteration += 1
                logger.info(f"=== 第 {iteration} 次数据采集 ===")
                
                matches = await self.fetch_and_save_matches()
                
                # 打印摘要
                for m in matches:
                    logger.info(f"{m.home_team} {m.home_score}-{m.away_score} {m.away_team} ({m.minute}')")
                
                await asyncio.sleep(interval_seconds)
            
            except KeyboardInterrupt:
                logger.info("用户中断，停止采集")
                break
            except Exception as e:
                logger.error(f"采集循环出错：{e}")
                await asyncio.sleep(interval_seconds)


async def main():
    """示例用法"""
    
    # 初始化数据服务
    service = DataService(config_path="configs/collector_config.yaml")
    
    # 单次获取
    print("\n=== 单次数据采集 ===\n")
    matches = await service.fetch_and_save_matches()
    
    for m in matches:
        print(f"{m.home_team} {m.home_score}-{m.away_score} {m.away_team}")
        print(f"  时间：{m.minute}' | 状态：{m.status}")
        print(f"  联赛：{m.league} ({m.country})")
        print(f"  赔率：主胜 {m.odds.get('home', 'N/A')} | 平 {m.odds.get('draw', 'N/A')} | 客胜 {m.odds.get('away', 'N/A')}")
        print(f"  统计数据：{m.stats}")
        print()
    
    # 如需持续运行，取消下面注释
    # await service.run_continuous(interval_seconds=10)


if __name__ == '__main__':
    asyncio.run(main())
