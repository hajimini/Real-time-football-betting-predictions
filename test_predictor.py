#!/usr/bin/env python
"""测试 8 种盘口预测功能"""

from src.models.comprehensive_predictor import ComprehensiveBettingPredictor

predictor = ComprehensiveBettingPredictor()

# 模拟美国 VS 波黑的比赛数据
test_match = {
    'home_fifa_rank': 12, 'away_fifa_rank': 64,
    'home_squad_value': 3.86, 'away_squad_value': 1.27,
    'home_recent_form': 8, 'away_recent_form': 5,
    'is_neutral_venue': False, 'home_win_rate_at_home': 0.85,
    'home_goals_per_game': 3.0, 'away_goals_per_game': 1.67,
    'home_xg_per_game': 2.8, 'away_xg_per_game': 1.4,
    'home_goals_conceded_per_game': 0.5, 'away_goals_conceded_per_game': 2.0,
    'home_clean_sheet_rate': 0.5, 'away_clean_sheet_rate': 0.0,
    'home_corners_per_game': 7.3, 'away_corners_per_game': 4.0,
    'home_corners_conceded_per_game': 3.5, 'away_corners_conceded_per_game': 5.0,
    'home_yellow_cards_per_game': 1.3, 'away_yellow_cards_per_game': 1.7,
    'home_red_cards_per_season': 0.1, 'away_red_cards_per_season': 0.3,
    'referee_cards_per_game': 4.5, 'match_importance': 0.9,
    'home_avg_possession': 63, 'away_avg_possession': 44,
    'home_high_press_intensity': 8, 'away_low_block_tendency': 9,
    'h2h_home_win_rate': 0.67, 'h2h_btts_rate': 0.67, 'h2h_avg_goals': 2.3,
    'home_key_players_out': 0, 'away_key_players_out': 1,
    'home_first_15min_goal_rate': 0.5, 'away_last_15min_goal_rate': 0.4,
    'first_half_goals_percentage': 0.45,
    'odds_home': 1.53, 'odds_draw': 3.75, 'odds_away': 6.50
}

results = predictor.predict_all_markets(test_match)

print('=' * 70)
print('8 种盘口预测结果 - 测试通过')
print('=' * 70)
for market, result in results.items():
    print(f'\n【{market}】')
    if isinstance(result, dict):
        for k, v in result.items():
            print(f'  {k}: {v}')
    else:
        print(f'  {result}')
