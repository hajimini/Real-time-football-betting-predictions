"""
赔率数据采集器
采集实时赔率数据：胜平负、大小球、亚盘等
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger
from .base_collector import BaseCollector


class OddsDataCollector(BaseCollector):
    """赔率数据采集器"""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        super().__init__(api_url, api_key)
        self.odds_cache: Dict[int, Dict] = {}
    
    async def fetch_data(self, fixture_id: Optional[int] = None) -> Dict[str, Any]:
        """获取赔率数据"""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        endpoint = f"{self.api_url}/odds"
        params = {}
        if fixture_id:
            params["fixture"] = fixture_id
        else:
            params["live"] = "all"
        
        async with self.session.get(endpoint, params=params) as response:
            if response.status != 200:
                raise Exception(f"API request failed: {response.status}")
            data = await response.json()
            return data
    
    async def process_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理赔率数据"""
        processed = {
            "timestamp": datetime.now().isoformat(),
            "odds": []
        }
        
        results = raw_data.get("response", [])
        
        for item in results:
            odds_data = self._parse_odds(item)
            if odds_data:
                processed["odds"].append(odds_data)
                
                # 更新缓存
                fixture_id = item.get("fixture", {}).get("id")
                if fixture_id:
                    self.odds_cache[fixture_id] = odds_data
        
        logger.info(f"处理了 {len(processed['odds'])} 条赔率数据")
        return processed
    
    def _parse_odds(self, item: Dict) -> Optional[Dict[str, Any]]:
        """解析赔率数据"""
        fixture_id = item.get("fixture", {}).get("id")
        if not fixture_id:
            return None
        
        bookmakers = item.get("bookmakers", [])
        parsed_bookmakers = []
        
        for bookie in bookmakers:
            bets = []
            for bet in bookie.get("bets", []):
                bet_info = {
                    "name": bet.get("name"),
                    "values": []
                }
                for value in bet.get("values", []):
                    bet_info["values"].append({
                        "value": value.get("value"),
                        "odd": float(value.get("odd", 0)) if value.get("odd") else None
                    })
                bets.append(bet_info)
            
            parsed_bookmakers.append({
                "id": bookie.get("id"),
                "name": bookie.get("name"),
                "bets": bets
            })
        
        return {
            "fixture_id": fixture_id,
            "update_time": item.get("update"),
            "bookmakers": parsed_bookmakers
        }
    
    def get_latest_odds(self, fixture_id: int) -> Optional[Dict]:
        """获取指定比赛的最新赔率"""
        return self.odds_cache.get(fixture_id)
    
    def extract_market_odds(self, fixture_id: int, market_name: str) -> Dict[str, float]:
        """提取特定市场的赔率"""
        odds_data = self.odds_cache.get(fixture_id)
        if not odds_data:
            return {}
        
        result = {}
        for bookie in odds_data.get("bookmakers", []):
            for bet in bookie.get("bets", []):
                if market_name.lower() in bet.get("name", "").lower():
                    for val in bet.get("values", []):
                        key = f"{bookie['name']}_{val['value']}"
                        result[key] = val['odd']
        
        return result
    
    def get_average_odds(self, fixture_id: int, market_name: str) -> Dict[str, float]:
        """计算特定市场的平均赔率"""
        odds_data = self.odds_cache.get(fixture_id)
        if not odds_data:
            return {}
        
        odds_by_value = {}
        
        for bookie in odds_data.get("bookmakers", []):
            for bet in bookie.get("bets", []):
                if market_name.lower() in bet.get("name", "").lower():
                    for val in bet.get("values", []):
                        v = val['value']
                        odd = val['odd']
                        if odd:
                            if v not in odds_by_value:
                                odds_by_value[v] = []
                            odds_by_value[v].append(odd)
        
        averages = {}
        for value, odds_list in odds_by_value.items():
            if odds_list:
                averages[value] = sum(odds_list) / len(odds_list)
        
        return averages
