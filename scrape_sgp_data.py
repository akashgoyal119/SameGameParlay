import os 
import pandas as pd 
import requests
import smart_open
import numpy as np
from ag_draftking_utils.util import get_current_chicago_time, save_pandas_df
from ag_draftking_utils.odds_conversions import *
import tqdm
import json
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

NOW = get_current_chicago_time()
TODAY = str(NOW.date())
PROJECT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))


class Sgp:

    def __init__(self, bookie, minimum_straight_ev=-50, maximum_straight_ev=100,
                 max_combinations_per_game=250,
                 sgp_num_legs=2, num_threads=5):
        self.bookie = bookie
        self.minimum_straight_ev = minimum_straight_ev
        self.maximum_straight_ev = maximum_straight_ev
        self.max_combinations_per_game = max_combinations_per_game
        self.sgp_num_legs = sgp_num_legs
        self.num_threads = num_threads

    def get_bet_sid(self, game_df, bet_index):
        return game_df.loc[bet_index]['sid']

    def process_combo(self, combination, game_df):
        bet_uuid = uuid.uuid4()
        try:
            sid_list = []
            for bet_index in combination:
                sid_list.append(self.get_bet_sid(game_df, bet_index))

            bet1 = combination[0]
            link = game_df.loc[bet1]['link']
            event_id = game_df.loc[bet1]['id']

            response = self.get_single_sgp_request(sid_list, link=link)
            save_folder = os.path.join(PROJECT_DIRECTORY, f'data/odds_data/sgp/{self.bookie}/{TODAY}/')
            save_file_name = '____'.join(sid_list) + '.json'
            try:
                with smart_open.open(
                        os.path.join(save_folder, 'json', save_file_name), 'w') as f:
                   json.dump(response.json(), f)
            except Exception as e:
                # filename too long...
                pass
            price = self.get_price_from_response(response)

            pricing_info = []
            for i in range(len(combination)):
                pricing_info.append([event_id, bet_uuid, sid_list[i], price, response.status_code])
            return pd.DataFrame(pricing_info, columns=['event_id', 'bet_uuid', 'sid', 'sgp_price', 'response_status_code'])
        except KeyError as e:
            #print(f'fail due to {e}.')
            return pd.DataFrame()
        except Exception as e:
            sid_name = ','.join(sid_list)
            #print(f'Error for {sid_name}, {event_id} due to {e}.')
            return pd.DataFrame()

    def filter_out_book_specific_combination_types(self, df):
        """
        The idea here is that certain books don't allow SGPs for certain stat pairings, so
        add some custom logic here to remove them. Generally I should override this function
        in the subclasses...
        """
        return df

    def postprocess_book_specific_combos(self, game_df, combos):
        """
        Certain combination types aren't valid at specific books, so this removes some of these
        combos so that they are not sampled...
        """
        second_pass_combos = []
        for combo in combos:
            subset = game_df.loc[list(combo), ['stat', 'market_type', 'line', 'player_name']]
            # if the main values here are the same we shouldnt bother checking for SGPs
            if all(subset[col].unique().shape[0] == 1 for col in subset.columns):
                continue
            second_pass_combos.append(combo)
        return second_pass_combos

    def create_combinations(self, possible_bets_df, event_id, angle_type='nba'):
        book_df = possible_bets_df[
            (possible_bets_df['bookie'] == self.bookie) &
            (possible_bets_df['expected_value'].between(self.minimum_straight_ev, self.maximum_straight_ev)) &
            (possible_bets_df['id'] == event_id)
        ].reset_index(drop=True)

        def fast_random_combinations(indices, r, k):
            indices = np.array(indices)
            combos = [tuple(np.random.choice(indices, r, replace=False)) for _ in range(k)]
            return combos

        if angle_type == 'nfl':
            return self.create_combinations_touchdown_angle(book_df)
        elif angle_type == 'nba':
            return self.create_combinations_nba(book_df)

        game_df = self.filter_out_book_specific_combination_types(book_df)
        game_df = game_df.sample(frac=1).reset_index(drop=True)
        combos = fast_random_combinations(game_df.index, self.sgp_num_legs, self.max_combinations_per_game)
        second_pass_combos = self.postprocess_book_specific_combos(game_df, combos)
        return game_df, second_pass_combos

    def create_combinations_nba(self, book_df):
        book_df = book_df.sort_values(by=['price'])
        home_runs = book_df.iloc[1:]
        home_runs.index = list(range(home_runs.shape[0]))
        non_hr = book_df.iloc[:1]
        non_hr_shape = non_hr.shape[0]
        start_index = home_runs.shape[0]
        non_hr.index = list(range(start_index, start_index + non_hr_shape))

        game_df = pd.concat([home_runs, non_hr])
        combos = []
        for i in range(home_runs.shape[0]):
            home_runs_index = home_runs.iloc[i].name
            non_sampled_qty = min(10, non_hr.shape[0])
            sampled_non_hr = non_hr.sample(n=non_sampled_qty)
            for j in range(sampled_non_hr.shape[0]):
                non_hr_index = sampled_non_hr.iloc[j].name
                combos.append((home_runs_index, non_hr_index))
        return game_df, combos


    def create_combinations_touchdown_angle(self, book_df):
        home_runs = book_df[
            (book_df['stat'].str.contains('player_anytime_td'))
        ]
        home_runs.index = list(range(home_runs.shape[0]))
        non_hr = book_df[
            ~(book_df['stat'].str.contains('player_anytime_td'))
        ]
        non_hr_shape = non_hr.shape[0]
        start_index = home_runs.shape[0]
        non_hr.index = list(range(start_index, start_index + non_hr_shape))

        game_df = pd.concat([home_runs, non_hr])
        combos = []
        for i in range(home_runs.shape[0]):
            home_runs_index = home_runs.iloc[i].name
            non_sampled_qty = min(10, non_hr.shape[0])
            sampled_non_hr = non_hr.sample(n=non_sampled_qty)
            for j in range(sampled_non_hr.shape[0]):
                non_hr_index = sampled_non_hr.iloc[j].name
                combos.append((home_runs_index, non_hr_index))
        return game_df, combos

    def get_all_prices(self, possible_bets_df, game_pk):
        game_df, combos = self.create_combinations(possible_bets_df, game_pk)

        folder = os.path.join(PROJECT_DIRECTORY, f'data/odds_data/sgp/{self.bookie}/json/{game_pk}')
        if not os.path.exists(folder):
            os.makedirs(folder)
        sgp_prices = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.process_combo, combo, game_df) for combo in combos]
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                sgp_prices.append(future.result())
        if len(sgp_prices) > 0:
            sgp_prices = pd.concat(sgp_prices)
            sgp_prices['bet_uuid'] = sgp_prices['bet_uuid'].astype(str)
        else:
            sgp_prices = pd.DataFrame()
        return sgp_prices

    def combine_sgp_prices_w_original_probabilities(self, sgp_prices, straight_odds_df):
        straight_odds_df = straight_odds_df[straight_odds_df['bookie'] == self.bookie]
        sgp_final = sgp_prices.merge(
            straight_odds_df[['id', 'stat', 'player_name', 'market_type', 'line',
                              'selectionId', 'home_team', 'away_team', 'price',
                              'model_probability', 'link', 'expected_value']].rename(columns={
                'selectionId': 'sid', 'id': 'event_id'}
            ),
            how='inner',
            on=['event_id', 'sid']
        )
        sgp_final['imp_prob_straight'] = american_odds_to_breakeven_probability_vectorized(sgp_final, price_col='price')
        sgp_final['sgp_implied_probability'] = american_odds_to_breakeven_probability_vectorized(
            sgp_final, price_col='sgp_price')
        return sgp_final

    @staticmethod
    def clean_up_sgp_pricing(raw_sgp_price_df):
        raw_sgp_price_df = raw_sgp_price_df[
            (raw_sgp_price_df['sgp_price'].notnull()) &
            (raw_sgp_price_df['response_status_code'] == 200)
            ].drop(columns=['response_status_code'])
        raw_sgp_price_df['sgp_price'] = raw_sgp_price_df['sgp_price'].str.replace(r'−', '-').astype(int)
        raw_sgp_price_df = raw_sgp_price_df.sort_values(by=['event_id', 'bet_uuid'])
        return raw_sgp_price_df


    def main(self):
        straight_odds_and_probabilities_df = pd.read_parquet('data/straights_with_probabilities/2025-10-26.parquet')
        straight_odds_and_probabilities_df['selectionId'] = straight_odds_and_probabilities_df['link'].str.split('options=').str[-1].str.split('&').str[0]
        straight_odds_and_probabilities_df = straight_odds_and_probabilities_df[
            straight_odds_and_probabilities_df['home_team'].isin(['Sacramento Kings', 'Los Angeles Clippers'])
        ]
        sgp_prices = []
        for i, event_id in enumerate(straight_odds_and_probabilities_df['id'].unique()):
            _sgp_prices = self.get_all_prices(straight_odds_and_probabilities_df, event_id)
            if _sgp_prices.shape[0] > 0:
                sgp_prices.append(_sgp_prices)
                save_folder = os.path.join(PROJECT_DIRECTORY, f'data/odds_data/sgp/{self.bookie}/parquet/{TODAY}')
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_pandas_df(_sgp_prices, os.path.join(save_folder, f'{event_id}.parquet'))
            else:
                continue
        sgp_prices = pd.concat(sgp_prices)
        save_folder = os.path.join(PROJECT_DIRECTORY, f'data/odds_data/sgp/{self.bookie}/aggregated_parquet')
        save_pandas_df(sgp_prices, os.path.join(save_folder, f'{NOW}.parquet'))

        sgp_prices = self.clean_up_sgp_pricing(sgp_prices)
        sgp_and_straights_df = self.combine_sgp_prices_w_original_probabilities(
            sgp_prices, straight_odds_and_probabilities_df)

        sgp_and_straights_df['sgp_probability'] = sgp_and_straights_df.groupby('bet_uuid')['model_probability'].transform('prod')
        sgp_and_straights_df = calculate_expected_value_vectorized(sgp_and_straights_df, 'sgp_probability', price_col='sgp_price',
            expected_value_col='sgp_expected_value')
        return sgp_and_straights_df



class MgmSgp(Sgp):
    def add_parlay_deeplink(self, df):
        links = df['link'].str.split('options=').str[-1].str.split('&').str[0]
        links = ','.join([str(x) for x in links])
        return f'https://www.nj.betmgm.com/en/sports?options={links}'

    def get_price_from_response(self, response):
        js = response.json()
        keys = list(js['betBuilderPricingGroups'].keys())
        if len(keys) != 1:
            return None
        bet_id = keys[0]
        return str(int(js['betBuilderPricingGroups'][bet_id]['odds']['americanOdds']))

    def get_bet_sid(self, game_df, bet_index):
        """Need to override this from base-class because the sid lacks the fixture/game/result id pieces of
        information and just has 1 of these."""
        return game_df.loc[bet_index]['link'].split('options=')[-1].split('&')[0]

    def get_single_sgp_request(self, url_list, link=None):
        payload = {"tv1Picks": [], "tv2Picks": []}
        for url in url_list:
            try:
                fixture_id, game_id, _, result_id = url.split('-')
                result_id = -int(result_id)
            except ValueError:
                fixture_id, game_id, result_id = url.split('-')
                result_id = int(result_id)

            payload['tv1Picks'].append({
                'fixtureId': f'{fixture_id}',
                'gameId': int(game_id),
                'resultId': result_id,
                'useLiveFallback': False
            })
        headers = {
            "content-type": "application/json",
            "user-agent": "Mozilla/5.0",
            "origin": "https://www.il.betmgm.com"
        }
        url = "https://www.il.betmgm.com/cds-api/bettingoffer/picks"
        params = {
            #"x-bwin-accessid": "N2Q4OGJjODYtODczMi00NjhhLWJlMWItOGY5MDUzMjYwNWM5",
            "x-bwin-accessid": "ZTg4YWEwMTgtZTlhYy00MWRkLWIzYWYtZjMzODI5ZDE0Mjc5",
            "lang": "en-us",
            "country": "US",
            "userCountry": "US",
            "subdivision": "US-Illinois",
        }
        initial_request = requests.post(url, params=params, json=payload, headers=headers)
        pick_group_id = initial_request.json()['fixturePage']['fixtures'][0]['addons']['betBuilderId']

        for i in range(len(url_list)):
            payload['tv1Picks'][i]['pickGroupId'] = pick_group_id
        response = requests.post(url, params=params, json=payload, headers=headers)
        return response


if __name__ == '__main__':
    sgp = {
        'betmgm': MgmSgp('betmgm', max_combinations_per_game=8000, num_threads=10, sgp_num_legs=2)
    }
    sgp_books_to_scrape = ['betmgm']
    for book in sgp_books_to_scrape:
        obj = sgp[book]
        prices = obj.main()
        save_pandas_df(
            prices,
            os.path.join(
                PROJECT_DIRECTORY, f'data/odds_data/sgp/{book}/{NOW}.parquet')
        )