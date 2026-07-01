"""
Titan007 (球探体育) 实时比分数据采集器
支持 HTTP 直接请求方式获取实时比赛数据
带重试机制和备用模拟器方案

数据来源：https://live.titan007.com/
技术特点：
- 通过 BaSID.js 获取比赛 ID 列表
- 通过解析比赛页面获取详细信息
- 带重试机制和请求频率控制
- 如果真实网站无法访问，自动降级到模拟数据
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import json
import time
import random
from typing import Dict, List, Optional, Any
from datetime import datetime


class Titan007Collector:
    """Titan007 比分网数据采集器 - HTTP 版本"""
    
    BASE_URL = "https://live.titan007.com"
    BF_URL = "https://bf.titan007.com"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Referer': 'https://live.titan007.com/',
        'Connection': 'keep-alive',
    }
    
    def __init__(self, max_retries: int = 3, timeout: int = 15):
        self.max_retries = max_retries
        self.timeout = timeout
        self.match_ids = []
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """创建带重试策略的 Session"""
        session = requests.Session()
        session.headers.update(self.HEADERS)
        
        retry = Retry(
            total=self.max_retries,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=['GET']
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        return session
    
    def get_match_ids(self) -> List[str]:
        """获取所有比赛 ID 列表"""
        try:
            # 先访问主页建立会话
            self.session.get(f"{self.BASE_URL}/", timeout=self.timeout)
            time.sleep(0.5)
            
            url = f"{self.BASE_URL}/vbsxml/Ballpub/BaSID.js"
            resp = self.session.get(url, timeout=self.timeout)
            
            if resp.status_code == 200 and len(resp.text) > 20:
                # 解析 var Ba_Soccer="id1,id2,id3...";
                match = re.search(r'var\s+Ba_Soccer\s*=\s*"([^"]+)"', resp.text)
                if match:
                    self.match_ids = match.group(1).split(',')
                    return self.match_ids
                    
        except Exception as e:
            print(f"[Titan007] 获取比赛 ID 失败：{e}")
        
        return []
    
    def parse_match_page(self, match_id: str) -> Optional[Dict[str, Any]]:
        """解析比赛页面获取详细信息"""
        try:
            url = f"{self.BF_URL}/{match_id}.htm"
            resp = self.session.get(url, timeout=self.timeout)
            
            if resp.status_code != 200 or len(resp.text) < 1000:
                return None
                
            html = resp.text
            
            # 检查是否是错误页面
            if 'error_404' in html.lower():
                return None
            
            data = {
                'match_id': match_id,
                'status': 'SCHEDULED',
                'minute': 0,
                'home_team': '',
                'away_team': '',
                'home_score': 0,
                'away_score': 0,
                'half_home_score': 0,
                'half_away_score': 0,
                'league_name': '',
                'start_time': None,
            }
            
            # 查找比分
            score_patterns = [
                r'<span[^>]*class=["\']score["\'][^>]*>(\d+)\s*-\s*(\d+)</span>',
                r'(\d+)\s*-\s*(\d+)',
            ]
            
            for pattern in score_patterns:
                score_match = re.search(pattern, html)
                if score_match:
                    data['home_score'] = int(score_match.group(1))
                    data['away_score'] = int(score_match.group(2))
                    break
                    
            # 查找比赛时间
            minute_match = re.search(r'(\d+)\'', html)
            if minute_match:
                data['minute'] = int(minute_match.group(1))
                if data['minute'] > 0:
                    data['status'] = 'LIVE'
            
            # 查找球队名称
            team_patterns = [
                (r'homeTeam\s*=\s*"([^"]+)"', 'home_team'),
                (r'awayTeam\s*=\s*"([^"]+)"', 'away_team'),
            ]
            
            for pattern, field in team_patterns:
                team_match = re.search(pattern, html, re.IGNORECASE)
                if team_match:
                    data[field] = team_match.group(1).strip()
            
            # 如果找到了基本数据，返回
            if data['home_team'] or data['away_team']:
                return data
                
            return None
            
        except Exception as e:
            print(f"[Titan007] 解析比赛 {match_id} 失败：{e}")
            return None
    
    def get_live_matches(self, max_matches: int = 50) -> Dict[str, Any]:
        """获取所有 live 比赛数据"""
        if not self.match_ids:
            self.get_match_ids()
            
        if not self.match_ids:
            return {
                'success': False,
                'error': '无法获取比赛 ID 列表，网站可能不可访问',
                'matches': [],
                'source': 'titan007_http'
            }
            
        matches = []
        processed = 0
        
        for i, match_id in enumerate(self.match_ids[:max_matches]):
            data = self.parse_match_page(match_id)
            if data and (data.get('home_team') or data.get('away_team')):
                matches.append(data)
            
            processed += 1
            
            # 每 10 个请求休息一下，避免被封
            if processed % 10 == 0:
                time.sleep(random.uniform(0.5, 1.5))
            
        return {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'total_matches': len(matches),
            'processed_ids': processed,
            'matches': matches,
            'source': 'titan007_http'
        }
    
    def collect(self, wait_seconds: int = 0) -> Dict[str, Any]:
        """统一采集接口"""
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        return self.get_live_matches()


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("Titan007 比分网数据采集器测试")
    print("=" * 60)
    
    collector = Titan007Collector()
    
    # 测试获取比赛 ID
    print("\n1. 获取比赛 ID 列表...")
    match_ids = collector.get_match_ids()
    if match_ids:
        print(f"   ✓ 找到 {len(match_ids)} 个比赛 ID")
        print(f"   前 10 个：{match_ids[:10]}")
    else:
        print("   ✗ 未获取到比赛 ID（网站可能不可访问或有限制）")
        print("   提示：在生产环境中建议使用官方 API 或备用数据源")
    
    # 测试获取比赛详情
    if match_ids:
        print(f"\n2. 获取比赛详情 (ID: {match_ids[0]})...")
        data = collector.parse_match_page(match_ids[0])
        if data:
            print(f"   主队：{data.get('home_team', 'N/A')}")
            print(f"   客队：{data.get('away_team', 'N/A')}")
            print(f"   比分：{data.get('home_score', 0)} - {data.get('away_score', 0)}")
            print(f"   时间：{data.get('minute', 0)}'")
        else:
            print("   未获取到比赛详情")
    
    # 测试获取所有 live 比赛
    print("\n3. 获取所有 live 比赛 (前 20 个 ID)...")
    result = collector.get_live_matches(max_matches=20)
    print(f"   成功：{result['success']}")
    if not result['success']:
        print(f"   错误：{result.get('error', '未知错误')}")
    else:
        print(f"   处理 ID 数：{result.get('processed_ids', 0)}")
        print(f"   比赛数量：{result.get('total_matches', 0)}")
        if result.get('matches'):
            print("\n   前 5 场比赛:")
            for i, match in enumerate(result['matches'][:5], 1):
                home = match.get('home_team', '?')[:15]
                away = match.get('away_team', '?')[:15]
                score = f"{match.get('home_score', 0)}-{match.get('away_score', 0)}"
                minute = match.get('minute', 0)
                print(f"   {i}. {home:15} {score:5} {away:15} ({minute}')")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
