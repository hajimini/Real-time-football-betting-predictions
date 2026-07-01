"""
实时比赛数据采集器模块

支持多种数据源：
1. API 接口（推荐）：Football-Data.org, API-Football, SportMonks
2. 网页爬虫：FlashScore, 7M, 雷速体育（仅供学习，注意 robots.txt）
3. 本地模拟：用于测试和开发

架构设计：
- BaseCollector: 抽象基类，定义标准接口
- APICollector: RESTful API 数据采集
- WebScraperCollector: 网页爬虫采集
- SimulatorCollector: 本地数据模拟（开发用）
"""

import asyncio
import aiohttp
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatchData:
    """比赛数据模型"""
    
    def __init__(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        minute: int,
        status: str,  # 'LIVE', 'HT', 'FT', 'NS'
        league: str,
        country: str,
        start_time: datetime,
        events: Optional[List[Dict]] = None,
        stats: Optional[Dict] = None,
        odds: Optional[Dict] = None
    ):
        self.match_id = match_id
        self.home_team = home_team
        self.away_team = away_team
        self.home_score = home_score
        self.away_score = away_score
        self.minute = minute
        self.status = status
        self.league = league
        self.country = country
        self.start_time = start_time
        self.events = events or []
        self.stats = stats or {}
        self.odds = odds or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'match_id': self.match_id,
            'home_team': self.home_team,
            'away_team': self.away_team,
            'home_score': self.home_score,
            'away_score': self.away_score,
            'minute': self.minute,
            'status': self.status,
            'league': self.league,
            'country': self.country,
            'start_time': self.start_time.isoformat(),
            'events': self.events,
            'stats': self.stats,
            'odds': self.odds,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MatchData':
        return cls(
            match_id=data['match_id'],
            home_team=data['home_team'],
            away_team=data['away_team'],
            home_score=data['home_score'],
            away_score=data['away_score'],
            minute=data['minute'],
            status=data['status'],
            league=data['league'],
            country=data['country'],
            start_time=datetime.fromisoformat(data['start_time']),
            events=data.get('events', []),
            stats=data.get('stats', {}),
            odds=data.get('odds', {}),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat()))
        )


class BaseCollector(ABC):
    """数据采集器抽象基类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.session = None
    
    @abstractmethod
    async def fetch_live_matches(self) -> List[MatchData]:
        """获取所有正在进行的比赛"""
        pass
    
    @abstractmethod
    async def fetch_match_detail(self, match_id: str) -> MatchData:
        """获取指定比赛的详细信息"""
        pass
    
    @abstractmethod
    async def fetch_odds(self, match_id: str) -> Dict:
        """获取指定比赛的赔率数据"""
        pass
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()


class APICollector(BaseCollector):
    """
    API 数据采集器
    
    支持的 API 服务：
    - Football-Data.org (免费，适合学习)
    - API-Football (付费，数据全面)
    - SportMonks (付费，实时更新)
    
    配置示例：
    {
        "provider": "football_data_org",
        "api_key": "your_api_key",
        "base_url": "https://api.football-data.org/v4/",
        "rate_limit": 10,  # 每秒请求数限制
        "leagues": ["PL", "La Liga", "Serie A"]  # 关注的联赛
    }
    """
    
    PROVIDERS = {
        'football_data_org': {
            'base_url': 'https://api.football-data.org/v4/',
            'auth_header': 'X-Auth-Token'
        },
        'api_football': {
            'base_url': 'https://v3.football.api-sports.io/',
            'auth_header': 'x-apisports-key'
        },
        'sportmonks': {
            'base_url': 'https://api.sportmonks.com/v3/football/',
            'auth_header': 'Authorization'
        }
    }
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.provider = config.get('provider', 'football_data_org')
        self.api_key = config.get('api_key', '')
        self.base_url = self.PROVIDERS[self.provider]['base_url']
        self.auth_header = self.PROVIDERS[self.provider]['auth_header']
        self.rate_limit = config.get('rate_limit', 10)
        self.leagues = config.get('leagues', [])
        self._last_request_time = 0
    
    async def _rate_limit_wait(self):
        """速率限制控制"""
        elapsed = time.time() - self._last_request_time
        min_interval = 1.0 / self.rate_limit
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_request_time = time.time()
    
    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """发送 API 请求"""
        await self._rate_limit_wait()
        
        url = f"{self.base_url}{endpoint}"
        headers = {self.auth_header: self.api_key}
        
        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    logger.warning("API 速率限制，等待后重试")
                    await asyncio.sleep(5)
                    return await self._request(endpoint, params)
                else:
                    logger.error(f"API 错误: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"请求失败: {e}")
            return {}
    
    async def fetch_live_matches(self) -> List[MatchData]:
        """获取所有正在进行的比赛"""
        logger.info("获取实时比赛列表...")
        
        if self.provider == 'football_data_org':
            data = await self._request('matches?live=true')
            matches = data.get('matches', [])
        elif self.provider == 'api_football':
            data = await self._request('fixtures?live=all')
            matches = data.get('response', [])
        else:
            logger.error(f"不支持的 API 提供商: {self.provider}")
            return []
        
        result = []
        for m in matches:
            try:
                match_data = self._parse_match(m)
                if not self.leagues or match_data.league in self.leagues:
                    result.append(match_data)
            except Exception as e:
                logger.error(f"解析比赛数据失败: {e}")
        
        logger.info(f"找到 {len(result)} 场实时比赛")
        return result
    
    def _parse_match(self, raw_data: Dict) -> MatchData:
        """解析原始 API 数据为标准 MatchData 格式"""
        # 根据不同 API 提供商解析数据
        if self.provider == 'football_data_org':
            return MatchData(
                match_id=str(raw_data.get('id', '')),
                home_team=raw_data.get('homeTeam', {}).get('name', ''),
                away_team=raw_data.get('awayTeam', {}).get('name', ''),
                home_score=raw_data.get('score', {}).get('fullTime', {}).get('home', 0) or 0,
                away_score=raw_data.get('score', {}).get('fullTime', {}).get('away', 0) or 0,
                minute=raw_data.get('minute', 0),
                status='LIVE' if raw_data.get('status') == 'IN_PLAY' else raw_data.get('status', ''),
                league=raw_data.get('competition', {}).get('name', ''),
                country=raw_data.get('competition', {}).get('area', {}).get('name', ''),
                start_time=datetime.fromisoformat(raw_data.get('utcDate', '').replace('Z', '+00:00')),
                events=[],
                stats={},
                odds={}
            )
        elif self.provider == 'api_football':
            return MatchData(
                match_id=str(raw_data.get('fixture', {}).get('id', '')),
                home_team=raw_data.get('teams', {}).get('home', {}).get('name', ''),
                away_team=raw_data.get('teams', {}).get('away', {}).get('name', ''),
                home_score=raw_data.get('goals', {}).get('home', 0) or 0,
                away_score=raw_data.get('goals', {}).get('away', 0) or 0,
                minute=raw_data.get('fixture', {}).get('status', {}).get('elapsed', 0),
                status='LIVE' if raw_data.get('fixture', {}).get('status', {}).get('short') == '1H' or raw_data.get('fixture', {}).get('status', {}).get('short') == '2H' else raw_data.get('fixture', {}).get('status', {}).get('short', ''),
                league=raw_data.get('league', {}).get('name', ''),
                country=raw_data.get('league', {}).get('country', ''),
                start_time=datetime.fromtimestamp(raw_data.get('fixture', {}).get('timestamp', 0)),
                events=[],
                stats={},
                odds={}
            )
        else:
            raise ValueError(f"不支持的 API 提供商: {self.provider}")
    
    async def fetch_match_detail(self, match_id: str) -> MatchData:
        """获取指定比赛的详细信息"""
        logger.info(f"获取比赛详情: {match_id}")
        
        if self.provider == 'football_data_org':
            data = await self._request(f'matches/{match_id}')
            return self._parse_match(data)
        elif self.provider == 'api_football':
            data = await self._request(f'fixtures?id={match_id}')
            if data.get('response'):
                return self._parse_match(data['response'][0])
        
        raise ValueError(f"无法获取比赛详情: {match_id}")
    
    async def fetch_odds(self, match_id: str) -> Dict:
        """获取指定比赛的赔率数据"""
        logger.info(f"获取比赛赔率: {match_id}")
        
        if self.provider == 'api_football':
            data = await self._request(f'odds?fixture={match_id}')
            if data.get('response'):
                return self._parse_odds(data['response'][0])
        
        return {}
    
    def _parse_odds(self, raw_data: Dict) -> Dict:
        """解析赔率数据"""
        odds = {}
        bookmakers = raw_data.get('bookmakers', [])
        
        for bm in bookmakers:
            bookmaker_name = bm.get('name', 'unknown')
            for bet in bm.get('bets', []):
                bet_name = bet.get('name', '')
                if 'Match Winner' in bet_name or '1X2' in bet_name:
                    for value in bet.get('values', []):
                        label = value.get('value', '')
                        odd_value = value.get('odd', '')
                        if 'Home' in label or label == '1':
                            odds['home'] = float(odd_value)
                        elif 'Draw' in label or label == 'X':
                            odds['draw'] = float(odd_value)
                        elif 'Away' in label or label == '2':
                            odds['away'] = float(odd_value)
        
        return odds


class WebScraperCollector(BaseCollector):
    """
    网页爬虫数据采集器
    
    ⚠️ 注意事项：
    1. 仅用于个人学习和研究
    2. 遵守目标网站的 robots.txt 协议
    3. 控制爬取频率，避免对服务器造成压力
    4. 商业使用需获得授权
    
    支持网站：
    - FlashScore (flashscore.com)
    - 7M 体育 (7m.cn)
    - 雷速体育 (lei-su.com)
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.target_site = config.get('target_site', 'flashscore')
        self.base_url = config.get('base_url', 'https://www.flashscore.com')
    
    async def fetch_live_matches(self) -> List[MatchData]:
        """从网页爬取实时比赛数据"""
        logger.info(f"从 {self.target_site} 爬取实时比赛...")
        
        # 这里需要实现具体的 HTML 解析逻辑
        # 由于不同网站结构不同，需要根据实际情况编写解析代码
        # 以下是一个示例框架
        
        try:
            async with self.session.get(f"{self.base_url}/football/live/") as response:
                html = await response.text()
                # TODO: 使用 BeautifulSoup 或 lxml 解析 HTML
                # 提取比赛 ID、球队名称、比分、时间等信息
                # 返回 MatchData 对象列表
                logger.warning("HTML 解析逻辑需要根据具体网站实现")
                return []
        except Exception as e:
            logger.error(f"爬取失败: {e}")
            return []
    
    async def fetch_match_detail(self, match_id: str) -> MatchData:
        """爬取指定比赛的详细信息"""
        logger.info(f"爬取比赛详情: {match_id}")
        # TODO: 实现具体逻辑
        raise NotImplementedError("需要实现具体网站的解析逻辑")
    
    async def fetch_odds(self, match_id: str) -> Dict:
        """爬取指定比赛的赔率数据"""
        logger.info(f"爬取比赛赔率: {match_id}")
        # TODO: 实现具体逻辑
        return {}


class SimulatorCollector(BaseCollector):
    """
    模拟器数据采集器
    
    用于开发和测试，生成符合真实分布的模拟比赛数据
    基于泊松过程和球队实力参数
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.teams = config.get('teams', self._default_teams())
        self.active_matches = {}
    
    def _default_teams(self) -> Dict[str, Dict]:
        """默认球队实力参数"""
        return {
            'Manchester City': {'attack': 2.3, 'defense': 0.8},
            'Liverpool': {'attack': 2.1, 'defense': 0.9},
            'Arsenal': {'attack': 2.0, 'defense': 1.0},
            'Real Madrid': {'attack': 2.2, 'defense': 0.9},
            'Barcelona': {'attack': 2.0, 'defense': 1.1},
            'Bayern Munich': {'attack': 2.4, 'defense': 0.8},
            'PSG': {'attack': 2.1, 'defense': 1.0},
            'Inter Milan': {'attack': 1.8, 'defense': 1.0},
        }
    
    async def fetch_live_matches(self) -> List[MatchData]:
        """生成模拟的实时比赛数据"""
        import random
        import numpy as np
        
        logger.info("生成模拟实时比赛...")
        
        # 如果没有活跃比赛，创建一些
        if not self.active_matches:
            team_names = list(self.teams.keys())
            for i in range(0, len(team_names), 2):
                if i + 1 < len(team_names):
                    home = team_names[i]
                    away = team_names[i + 1]
                    match_id = f"sim_{home}_{away}"
                    
                    # 随机开始时间（过去 0-45 分钟）
                    start_minute = random.randint(0, 45)
                    current_minute = random.randint(start_minute, 90)
                    
                    # 基于球队实力生成比分
                    home_lambda = self.teams[home]['attack'] / self.teams[away]['defense']
                    away_lambda = self.teams[away]['attack'] / self.teams[home]['defense']
                    
                    home_goals = np.random.poisson(home_lambda * current_minute / 90)
                    away_goals = np.random.poisson(away_lambda * current_minute / 90)
                    
                    self.active_matches[match_id] = {
                        'home': home,
                        'away': away,
                        'home_score': home_goals,
                        'away_score': away_goals,
                        'minute': current_minute,
                        'start_time': datetime.now() - timedelta(minutes=current_minute)
                    }
        
        # 更新比赛状态
        result = []
        for match_id, data in self.active_matches.items():
            # 随机更新比分（模拟进球事件）
            if random.random() < 0.05 and data['minute'] < 90:  # 5% 概率进球
                if random.random() < 0.6:  # 主场优势
                    data['home_score'] += 1
                else:
                    data['away_score'] += 1
            
            data['minute'] = min(data['minute'] + 1, 90)
            
            status = 'LIVE'
            if data['minute'] >= 90:
                status = 'FT'
            elif data['minute'] >= 45 and data['minute'] < 46:
                status = 'HT'
            
            match_data = MatchData(
                match_id=match_id,
                home_team=data['home'],
                away_team=data['away'],
                home_score=data['home_score'],
                away_score=data['away_score'],
                minute=data['minute'],
                status=status,
                league='Simulation League',
                country='Virtual',
                start_time=data['start_time'],
                events=[],
                stats=self._generate_stats(data),
                odds=self._generate_odds(data)
            )
            result.append(match_data)
        
        # 移除已结束的比赛
        finished = [k for k, v in self.active_matches.items() if v['minute'] >= 90]
        for k in finished:
            del self.active_matches[k]
        
        return result
    
    def _generate_stats(self, data: Dict) -> Dict:
        """生成模拟统计数据"""
        import random
        total_shots = data['home_score'] + data['away_score'] + random.randint(5, 15)
        return {
            'possession_home': random.randint(40, 60),
            'shots_home': random.randint(3, total_shots),
            'shots_away': total_shots - random.randint(3, total_shots),
            'shots_on_target_home': random.randint(1, 5),
            'shots_on_target_away': random.randint(1, 5),
            'corners_home': random.randint(2, 8),
            'corners_away': random.randint(2, 8),
            'attacks_home': random.randint(30, 80),
            'attacks_away': random.randint(30, 80),
            'dangerous_attacks_home': random.randint(15, 40),
            'dangerous_attacks_away': random.randint(15, 40)
        }
    
    def _generate_odds(self, data: Dict) -> Dict:
        """生成模拟赔率"""
        import random
        
        home_strength = self.teams[data['home']]['attack'] / self.teams[data['away']]['defense']
        away_strength = self.teams[data['away']]['attack'] / self.teams[data['home']]['defense']
        
        # 根据当前比分和剩余时间调整赔率
        score_diff = data['home_score'] - data['away_score']
        time_factor = (90 - data['minute']) / 90
        
        home_prob = 0.5 + (home_strength - away_strength) * 0.1 + score_diff * 0.05 * time_factor
        draw_prob = 0.3 - abs(score_diff) * 0.05
        away_prob = 1 - home_prob - draw_prob
        
        # 确保概率为正
        home_prob = max(0.1, home_prob)
        draw_prob = max(0.1, draw_prob)
        away_prob = max(0.1, away_prob)
        
        # 归一化
        total = home_prob + draw_prob + away_prob
        home_prob /= total
        draw_prob /= total
        away_prob /= total
        
        # 转换为赔率（包含庄家优势）
        margin = 1.05
        return {
            'home': round(margin / home_prob, 2),
            'draw': round(margin / draw_prob, 2),
            'away': round(margin / away_prob, 2),
            'over_2_5': round(1.8 + random.uniform(-0.2, 0.2), 2),
            'under_2_5': round(2.0 + random.uniform(-0.2, 0.2), 2)
        }
    
    async def fetch_match_detail(self, match_id: str) -> MatchData:
        """获取模拟比赛详情"""
        matches = await self.fetch_live_matches()
        for m in matches:
            if m.match_id == match_id:
                return m
        raise ValueError(f"比赛不存在: {match_id}")
    
    async def fetch_odds(self, match_id: str) -> Dict:
        """获取模拟赔率"""
        matches = await self.fetch_live_matches()
        for m in matches:
            if m.match_id == match_id:
                return m.odds
        return {}


class CollectorFactory:
    """数据采集器工厂类"""
    
    @staticmethod
    def create_collector(collector_type: str, config: Dict) -> BaseCollector:
        """根据类型创建采集器"""
        collectors = {
            'api': APICollector,
            'scraper': WebScraperCollector,
            'simulator': SimulatorCollector
        }
        
        if collector_type not in collectors:
            raise ValueError(f"不支持的采集器类型: {collector_type}")
        
        return collectors[collector_type](config)


async def main():
    """示例用法"""
    
    # 使用模拟器（开发测试用）
    config = {
        'teams': {}  # 使用默认球队
    }
    
    async with CollectorFactory.create_collector('simulator', config) as collector:
        matches = await collector.fetch_live_matches()
        
        print(f"\n找到 {len(matches)} 场实时比赛:\n")
        for m in matches:
            print(f"{m.home_team} {m.home_score}-{m.away_score} {m.away_team}")
            print(f"  时间：{m.minute}' | 状态：{m.status}")
            print(f"  赔率：主胜 {m.odds.get('home', 'N/A')} | 平 {m.odds.get('draw', 'N/A')} | 客胜 {m.odds.get('away', 'N/A')}")
            print()


if __name__ == '__main__':
    asyncio.run(main())
