"""
数据清洗模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from loguru import logger


class DataCleaner:
    """数据清洗器"""
    
    def __init__(self):
        self.cleaning_stats = {
            "rows_removed": 0,
            "nulls_filled": 0,
            "outliers_removed": 0
        }
    
    def clean_match_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """清洗比赛数据"""
        cleaned = {
            "timestamp": data.get("timestamp"),
            "matches": []
        }
        
        for match in data.get("matches", []):
            if self._is_valid_match(match):
                cleaned_match = self._clean_single_match(match)
                cleaned["matches"].append(cleaned_match)
        
        logger.info(f"清洗完成，有效比赛数：{len(cleaned['matches'])}")
        return cleaned
    
    def _is_valid_match(self, match: Dict) -> bool:
        """验证比赛数据是否有效"""
        required_fields = ["match_id", "home_team", "away_team", "status"]
        for field in required_fields:
            if field not in match or match[field] is None:
                return False
        
        # 检查比分是否有效
        home_score = match.get("home_team", {}).get("score")
        away_score = match.get("away_team", {}).get("score")
        
        if home_score is None or away_score is None:
            return False
        
        if home_score < 0 or away_score < 0:
            return False
        
        return True
    
    def _clean_single_match(self, match: Dict) -> Dict[str, Any]:
        """清洗单场比赛数据"""
        cleaned = match.copy()
        
        # 清理球队信息
        for team_key in ["home_team", "away_team"]:
            team = cleaned.get(team_key, {})
            cleaned[team_key] = {
                "id": team.get("id"),
                "name": str(team.get("name", "")).strip(),
                "score": int(team.get("score", 0))
            }
        
        # 清理时间信息
        elapsed = match.get("elapsed")
        if elapsed is not None:
            cleaned["elapsed"] = max(0, min(90 + 15, int(elapsed)))  # 考虑伤停补时
        
        # 清理事件数据
        events = match.get("events", [])
        cleaned["events"] = self._clean_events(events)
        
        return cleaned
    
    def _clean_events(self, events: List[Dict]) -> List[Dict]:
        """清理事件数据"""
        cleaned_events = []
        
        for event in events:
            if not event:
                continue
            
            cleaned_event = {
                "time": event.get("time", {}).get("elapsed", 0),
                "type": event.get("type", ""),
                "team": event.get("team", ""),
                "player": event.get("player", {}).get("id"),
                "detail": event.get("detail", "")
            }
            
            # 只保留有效事件类型
            valid_types = ["Goal", "Card", "subst", "VAR", "Penalty"]
            if cleaned_event["type"] in valid_types:
                cleaned_events.append(cleaned_event)
        
        return cleaned_events
    
    def clean_odds_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """清洗赔率数据"""
        cleaned = {
            "timestamp": data.get("timestamp"),
            "odds": []
        }
        
        for odds_record in data.get("odds", []):
            if self._is_valid_odds(odds_record):
                cleaned["odds"].append(odds_record)
        
        logger.info(f"清洗完成，有效赔率记录数：{len(cleaned['odds'])}")
        return cleaned
    
    def _is_valid_odds(self, odds: Dict) -> bool:
        """验证赔率数据是否有效"""
        if not odds.get("fixture_id"):
            return False
        
        bookmakers = odds.get("bookmakers", [])
        if not bookmakers:
            return False
        
        # 检查是否有有效的赔率值
        for bookie in bookmakers:
            for bet in bookie.get("bets", []):
                for value in bet.get("values", []):
                    odd = value.get("odd")
                    if odd and odd > 0:
                        return True
        
        return False
    
    def to_dataframe(self, matches: List[Dict]) -> pd.DataFrame:
        """将比赛数据转换为 DataFrame"""
        rows = []
        
        for match in matches:
            row = {
                "match_id": match.get("match_id"),
                "status": match.get("status"),
                "elapsed": match.get("elapsed"),
                "home_team_id": match.get("home_team", {}).get("id"),
                "home_team_name": match.get("home_team", {}).get("name"),
                "home_score": match.get("home_team", {}).get("score"),
                "away_team_id": match.get("away_team", {}).get("id"),
                "away_team_name": match.get("away_team", {}).get("name"),
                "away_score": match.get("away_team", {}).get("score"),
                "league_id": match.get("league", {}).get("id"),
                "league_name": match.get("league", {}).get("name"),
                "total_goals": (match.get("home_team", {}).get("score", 0) + 
                               match.get("away_team", {}).get("score", 0)),
                "goal_difference": (match.get("home_team", {}).get("score", 0) - 
                                    match.get("away_team", {}).get("score", 0))
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
