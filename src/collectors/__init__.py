"""
数据采集模块
负责从各种数据源采集实时比赛数据和赔率数据
"""

from .data_collector import (
    BaseCollector,
    MatchData,
    APICollector,
    WebScraperCollector,
    SimulatorCollector,
    CollectorFactory
)
from .titan007_collector import Titan007Collector
from .data_service import DataService

__all__ = [
    "BaseCollector",
    "MatchData",
    "APICollector",
    "WebScraperCollector",
    "SimulatorCollector",
    "Titan007Collector",
    "CollectorFactory",
    "DataService"
]
