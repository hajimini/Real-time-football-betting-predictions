"""
比赛数据采集器
采集实时比赛数据：比分、时间、事件等
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger
from .base_collector import BaseCollector


class MatchDataCollector(BaseCollector):
    """比赛数据采集器"""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        super().__init__(api_url, api_key)
        self.live_matches_cache: Dict[int, Dict] = {}
    
    async def fetch_data(self, league_id: Optional[int] = None) -> Dict[str, Any]:
        """获取比赛数据"""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        endpoint = f"{self.api_url}/fixtures"
        params = {"live": "all"}
        if league_id:
            params["league"] = league_id
        
        async with self.session.get(endpoint, params=params) as response:
            if response.status != 200:
                raise Exception(f"API request failed: {response.status}")
            data = await response.json()
            return data
    
    async def process_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理比赛数据"""
        processed = {
            "timestamp": datetime.now().isoformat(),
            "matches": []
        }
        
        fixtures = raw_data.get("response", [])
        
        for fixture in fixtures:
            match_data = self._parse_fixture(fixture)
            processed["matches"].append(match_data)
            
            # 更新缓存
            match_id = fixture.get("fixture", {}).get("id")
            if match_id:
                self.live_matches_cache[match_id] = match_data
        
        logger.info(f"处理了 {len(processed['matches'])} 场 live 比赛")
        return processed
    
    def _parse_fixture(self, fixture: Dict) -> Dict[str, Any]:
        """解析单场比赛数据"""
        return {
            "match_id": fixture.get("fixture", {}).get("id"),
            "status": fixture.get("fixture", {}).get("status", {}).get("short"),
            "elapsed": fixture.get("fixture", {}).get("status", {}).get("elapsed"),
            "home_team": {
                "id": fixture.get("teams", {}).get("home", {}).get("id"),
                "name": fixture.get("teams", {}).get("home", {}).get("name"),
                "score": fixture.get("goals", {}).get("home"),
                "logo": fixture.get("teams", {}).get("home", {}).get("logo")
            },
            "away_team": {
                "id": fixture.get("teams", {}).get("away", {}).get("id"),
                "name": fixture.get("teams", {}).get("away", {}).get("name"),
                "score": fixture.get("goals", {}).get("away"),
                "logo": fixture.get("teams", {}).get("away", {}).get("logo")
            },
            "league": {
                "id": fixture.get("league", {}).get("id"),
                "name": fixture.get("league", {}).get("name"),
                "country": fixture.get("league", {}).get("country"),
                "season": fixture.get("league", {}).get("year")
            },
            "events": fixture.get("events", []),
            "statistics": fixture.get("statistics", [])
        }
    
    def get_live_matches(self) -> List[Dict]:
        """获取所有 live 比赛"""
        return list(self.live_matches_cache.values())
    
    def get_match_by_id(self, match_id: int) -> Optional[Dict]:
        """根据 ID 获取比赛"""
        return self.live_matches_cache.get(match_id)
