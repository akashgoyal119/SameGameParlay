import requests
import datetime
import os
import tqdm
import pandas as pd
from ag_draftking_utils.util import get_current_chicago_time


API_KEY = os.environ['odds_api_api_key']
BASE_URL = 'https://api.the-odds-api.com'

def get_markets_string():
    player_props = [
            'player_field_goals',
            'player_kicking_points',
            'player_pass_attempts',
            'player_pass_interceptions',
            'player_pass_longest_completion',
            'player_pass_tds',
            'player_pass_completions',
            'player_pass_yds_alternate',
            'player_pass_yds',
            'player_reception_yds',
            'player_reception_longest',
            'player_receptions',
            'player_reception_yds_alternate',
            'player_rush_yds',
            'player_rush_yds_alternate',
            'player_rush_reception_yds',
            'player_rush_reception_yds_alternate',
            'player_anytime_td',
            'player_1st_td',
            'player_rush_reception_tds_alternate'
        ]
    game_markets = [
            'h2h',
            'spreads',
            'totals',
            'alternate_spreads',
            'alternate_totals',
            'team_totals',
            'alternate_team_totals',
            'h2h_q1',
            'h2h_h1',
            'spreads_q1',
            'spreads_h1',
            'alternate_spreads_q1',
            'alternate_spreads_h1',
            'totals_q1',
            'totals_h1',
            'alternate_totals_q1',
            'alternate_totals_h1',
            'team_totals_h1',
            'team_totals_q1',
            'alternate_team_totals_q1',
            'alternate_team_totals_h1'
        ]
    return ','.join(player_props+game_markets)


def response_to_df_live(js_obj, event_id):
    l = []
    for bookmaker in js_obj['bookmakers']:
        bookie = bookmaker['key']
        original_link = bookmaker['link']
        for market in bookmaker['markets']:
            stat = market['key']
            last_update = market['last_update']
            for outcome in market['outcomes']:
                if stat in ['alternate_totals', 'alternate_totals_h1', 'alternate_totals_q1',
                            'totals', 'totals_h1', 'totals_q1']:
                    market_type = outcome['name']
                    player = '' 
                elif 'description' not in outcome.keys():
                    market_type = ''
                    player = outcome['name']
                else:
                    market_type = outcome['name']
                    player = outcome.get('description', None)
                    
                price = outcome['price']
                if 'link' in outcome.keys() and outcome['link'] is not None:
                    link = outcome['link']
                    link_to_record = link
                else:
                    link_to_record = original_link
                if market_type == 'Yes':
                    market_type = 'Over'
                    line = 0.5
                elif market_type == 'No':
                    market_type = 'Under'
                    line = 0.5
                else:
                    line = outcome.get('point', -99)
                l.append({
                    'id': event_id,
                    'bookie': bookie,
                    'stat': stat,
                    'last_update': last_update,
                    'market_type': market_type,
                    'player_name': player,
                    'price': price,
                    'line': line,
                    'link': link_to_record,
                    'bet_limit': outcome.get('bet_limit', None),
                    'selectionId': outcome.get('sid', None),
                })
    l = pd.DataFrame(l)
    return pd.DataFrame(l)


def main():
    url = f'{BASE_URL}/v4/sports/americanfootball_nfl/events?apiKey={API_KEY}'
    response = requests.get(url)
    events = pd.DataFrame(response.json())
    events['commence_time'] = pd.to_datetime(events['commence_time']).dt.tz_localize(None)
    current_events = events[
        events['commence_time'] <= (datetime.datetime.utcnow() + datetime.timedelta(days=4))
    ]

    NOW = get_current_chicago_time()
    current_day = str(NOW.date())

    if not os.path.exists(f'data/straights/{current_day}'):
        os.mkdir(f'data/straights/{current_day}')

    for event_id in tqdm.tqdm(current_events['id']):
        live_events_url = (f'{BASE_URL}/v4/sports/americanfootball_nfl/events/{event_id}/odds?apiKey={API_KEY}&regions=us&markets=' +
                            get_markets_string() + '&oddsFormat=american&includeLinks=true&includeBetLimits=true&includeSids=true')
        response_props = requests.get(live_events_url)
        odds_data = response_to_df_live(response_props.json(), event_id)
        odds_data = odds_data.merge(current_events, how='inner', on='id')
        odds_data.to_parquet(f'data/straights/{current_day}/{event_id}.parquet', index=False)


if __name__ == '__main__':
    main()