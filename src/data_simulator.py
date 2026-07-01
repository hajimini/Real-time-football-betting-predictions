"""
数据模拟器 - 生成符合真实足球比赛统计分布的历史数据
使用泊松过程模拟进球，基于球队实力参数
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class MatchSimulator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        # 球队实力数据库 (进攻/防守强度，基于真实足球数据分布)
        self.teams = {
            'Man_City': {'attack': 2.1, 'defense': 0.8, 'name': 'Manchester City'},
            'Liverpool': {'attack': 2.0, 'defense': 0.9, 'name': 'Liverpool'},
            'Arsenal': {'attack': 1.9, 'defense': 0.9, 'name': 'Arsenal'},
            'Man_United': {'attack': 1.5, 'defense': 1.2, 'name': 'Manchester United'},
            'Chelsea': {'attack': 1.6, 'defense': 1.1, 'name': 'Chelsea'},
            'Tottenham': {'attack': 1.7, 'defense': 1.3, 'name': 'Tottenham'},
            'Newcastle': {'attack': 1.4, 'defense': 1.1, 'name': 'Newcastle'},
            'Brighton': {'attack': 1.3, 'defense': 1.2, 'name': 'Brighton'},
            'Aston_Villa': {'attack': 1.5, 'defense': 1.3, 'name': 'Aston Villa'},
            'West_Ham': {'attack': 1.2, 'defense': 1.4, 'name': 'West Ham'},
            'Fulham': {'attack': 1.1, 'defense': 1.3, 'name': 'Fulham'},
            'Brentford': {'attack': 1.2, 'defense': 1.4, 'name': 'Brentford'},
            'Crystal_Palace': {'attack': 1.0, 'defense': 1.3, 'name': 'Crystal Palace'},
            'Wolves': {'attack': 1.0, 'defense': 1.4, 'name': 'Wolves'},
            'Everton': {'attack': 0.9, 'defense': 1.3, 'name': 'Everton'},
            'Nottm_Forest': {'attack': 0.9, 'defense': 1.5, 'name': 'Nottingham Forest'},
            'Luton': {'attack': 0.8, 'defense': 1.6, 'name': 'Luton Town'},
            'Burnley': {'attack': 0.8, 'defense': 1.7, 'name': 'Burnley'},
            'Sheffield_United': {'attack': 0.7, 'defense': 1.8, 'name': 'Sheffield United'},
            'Bournemouth': {'attack': 1.1, 'defense': 1.4, 'name': 'Bournemouth'},
        }
        
        self.league_avg_goals = 2.69  # 英超场均进球
        
    def calculate_expected_goals(self, home_team, away_team, home_advantage=1.15):
        """
        计算双方期望进球 (xG)
        基于: 进攻强度 × 对手防守强度 × 主客场优势
        """
        home_attack = self.teams[home_team]['attack']
        home_defense = self.teams[home_team]['defense']
        away_attack = self.teams[away_team]['attack']
        away_defense = self.teams[away_team]['defense']
        
        # 主队期望进球
        home_xg = (home_attack * away_defense * home_advantage) / self.league_avg_goals * 1.35
        # 客队期望进球
        away_xg = (away_attack * home_defense) / self.league_avg_goals * 1.35
        
        return home_xg, away_xg
    
    def simulate_goals_poisson(self, xg, minute_range=(0, 90)):
        """
        使用泊松过程模拟进球时间
        进球间隔服从指数分布
        """
        goals = []
        current_minute = minute_range[0]
        
        while current_minute < minute_range[1]:
            # 指数分布模拟进球间隔 (单位: 分钟)
            if xg > 0:
                interval = np.random.exponential(90 / xg)
            else:
                break
                
            current_minute += interval
            
            if current_minute < minute_range[1]:
                goals.append({
                    'minute': int(current_minute),
                    'xg_at_goal': xg  # 记录当时的期望进球值
                })
                
        return goals
    
    def generate_match_events(self, home_team, away_team):
        """生成单场比赛的完整事件流"""
        home_xg, away_xg = self.calculate_expected_goals(home_team, away_team)
        
        # 模拟进球
        home_goals = self.simulate_goals_poisson(home_xg)
        away_goals = self.simulate_goals_poisson(away_xg)
        
        # 添加一些随机事件影响 (红牌、伤病等)
        momentum_shifts = self._generate_momentum_shifts(home_xg, away_xg)
        
        # 合并所有事件
        events = []
        for g in home_goals:
            events.append({
                'type': 'goal',
                'team': 'home',
                'minute': g['minute'],
                'xg': g['xg_at_goal']
            })
        for g in away_goals:
            events.append({
                'type': 'goal',
                'team': 'away',
                'minute': g['minute'],
                'xg': g['xg_at_goal']
            })
        events.extend(momentum_shifts)
        
        events.sort(key=lambda x: x['minute'])
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_xg_pre': home_xg,
            'away_xg_pre': away_xg,
            'events': events,
            'final_home_goals': len(home_goals),
            'final_away_goals': len(away_goals)
        }
    
    def _generate_momentum_shifts(self, home_xg, away_xg):
        """生成影响比赛走势的事件"""
        shifts = []
        
        # 10% 概率发生红牌
        if np.random.random() < 0.1:
            minute = np.random.uniform(20, 75)
            team = 'home' if np.random.random() < 0.5 else 'away'
            shifts.append({
                'type': 'red_card',
                'team': team,
                'minute': int(minute)
            })
            
        return shifts
    
    def generate_season_data(self, num_matches=500):
        """生成整个赛季的比赛数据"""
        teams_list = list(self.teams.keys())
        matches = []
        
        for i in range(num_matches):
            # 随机选择对阵双方
            home, away = random.sample(teams_list, 2)
            match_data = self.generate_match_events(home, away)
            match_data['match_id'] = f"M{i+1:04d}"
            match_data['date'] = datetime(2023, 8, 1) + timedelta(days=np.random.randint(0, 280))
            matches.append(match_data)
            
        return matches
    
    def create_training_dataset(self, matches):
        """
        将比赛数据转换为机器学习训练格式
        每分钟生成一个样本 (用于滚球预测)
        """
        samples = []
        
        for match in matches:
            home_team = match['home_team']
            away_team = match['away_team']
            events = match['events']
            
            # 按分钟遍历比赛
            for minute in range(0, 90, 5):  # 每5分钟一个样本
                # 统计当前比分
                home_score = sum(1 for e in events if e['type'] == 'goal' and e['team'] == 'home' and e['minute'] <= minute)
                away_score = sum(1 for e in events if e['type'] == 'goal' and e['team'] == 'away' and e['minute'] <= minute)
                
                # 统计最近10分钟的射门/进攻势头
                recent_events = [e for e in events if minute-10 <= e['minute'] <= minute]
                home_recent = sum(1 for e in recent_events if e['team'] == 'home')
                away_recent = sum(1 for e in recent_events if e['team'] == 'away')
                
                # 是否有红牌
                home_red = any(e['type'] == 'red_card' and e['team'] == 'home' and e['minute'] <= minute for e in events)
                away_red = any(e['type'] == 'red_card' and e['team'] == 'away' and e['minute'] <= minute for e in events)
                
                # 剩余时间
                time_remaining = 90 - minute
                
                # 标签: 未来15分钟是否会有进球
                future_goals = sum(1 for e in events if minute < e['minute'] <= minute+15 and e['type'] == 'goal')
                label_goal = 1 if future_goals > 0 else 0
                
                # 标签: 最终比赛结果
                final_home = match['final_home_goals']
                final_away = match['final_away_goals']
                if final_home > final_away:
                    label_outcome = 1  # 主胜
                elif final_home == final_away:
                    label_outcome = 0  # 平局
                else:
                    label_outcome = -1  # 客胜
                
                sample = {
                    'match_id': match['match_id'],
                    'minute': minute,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score,
                    'goal_difference': home_score - away_score,
                    'home_xg_pre': match['home_xg_pre'],
                    'away_xg_pre': match['away_xg_pre'],
                    'home_recent_momentum': home_recent,
                    'away_recent_momentum': away_recent,
                    'momentum_diff': home_recent - away_recent,
                    'home_red_card': 1 if home_red else 0,
                    'away_red_card': 1 if away_red else 0,
                    'time_remaining': time_remaining,
                    'total_goals_so_far': home_score + away_score,
                    'label_next_goal_15min': label_goal,
                    'label_final_outcome': label_outcome,
                    'label_total_goals': final_home + final_away
                }
                samples.append(sample)
                
        return pd.DataFrame(samples)


if __name__ == "__main__":
    # 生成数据示例
    simulator = MatchSimulator()
    matches = simulator.generate_season_data(1000)
    df = simulator.create_training_dataset(matches)
    
    print(f"生成 {len(matches)} 场比赛")
    print(f"生成 {len(df)} 个训练样本")
    print(f"\n样本特征:")
    print(df.columns.tolist())
    print(f"\n前5行数据:")
    print(df.head())
    
    # 保存数据
    df.to_csv('data/training_data.csv', index=False)
    print(f"\n数据已保存到 data/training_data.csv")
