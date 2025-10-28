import scrape_odds_api
import scrape_sgp_data
import get_fair_value
import os
import numpy as np
from ag_draftking_utils.util import save_pandas_df, get_current_chicago_time, get_most_recent_file_from_directory, read_pandas_df
from ag_draftking_utils.odds_conversions import *

NOW = get_current_chicago_time()


def main():
    odds_data = scrape_odds_api.main()
    odds_data = get_fair_value.main(odds_data, load_saved_files=False)
    mgm_sgp = scrape_sgp_data.MgmSgp(
        'betmgm',
        max_combinations_per_game=8000,
        num_threads=20,
        sgp_num_legs=2
    )
    sgp_prices = mgm_sgp.main(odds_data)
    save_pandas_df(
        sgp_prices,
        os.path.join(
            scrape_sgp_data.PROJECT_DIRECTORY, f'data/odds_data/sgp/betmgm/{NOW}.parquet'
        )
    )
    sgp_prices['is_suspended'] = sgp_prices['suspension_state'] == 'MarketSuspended'
    sgp_prices['num_suspended_markets'] = sgp_prices.groupby('bet_uuid')['is_suspended'].transform('sum')
    sgp_prices['matchup'] = sgp_prices['away_team'] + ' @ ' + sgp_prices['home_team']

    sgp_prices = sgp_prices.groupby('bet_uuid').filter(lambda x: x.shape[0] == 2)

    main_players = sgp_prices \
        .sort_values(by=['sgp_expected_value', 'price'], ascending=[False, True]) \
        .groupby('bet_uuid') \
        .tail(1)
    main_players = main_players.rename(columns={'stat': 'odds_api_stat_name'})

    most_recent_predictions = get_most_recent_file_from_directory(
        'prod_bets/', use_s3=True, bucket_name='draft-kings-2022')
    model_predictions = read_pandas_df(most_recent_predictions)
    model_predictions = model_predictions[model_predictions['bookie'] == 'betmgm'].drop(columns=['bookie']).rename(
        columns={'PLAYER_NAME': 'player_name', 'expected_value': 'model_ev'})

    error_lines = main_players[
        (main_players['num_suspended_markets'] == 0) &
        (main_players['sgp_expected_value'] >= 5)
    ] \
        .sort_values(by=['sgp_expected_value'], ascending=False)[['bet_uuid', 'matchup', 'player_name', 'odds_api_stat_name',
                                                                  'market_type', 'line', 'sgp_expected_value',
                                                                  'sgp_probability', 'model_probability',
                                                                  'price', 'sgp_price', 'expected_value']] \
        .rename(columns={'price': 'price_straight', 'expected_value': 'E[V]_straight',
                         'model_probability': 'mid_market_broken_player_probability'})

    error_lines['mid_market_lock_probability'] = (
            error_lines['sgp_probability'] /error_lines['mid_market_broken_player_probability'])

    error_lines = error_lines.merge(
        model_predictions,
        how='inner',
        on=['player_name', 'odds_api_stat_name', 'market_type', 'line', 'matchup']
    )
    # choose the most conservative estimate of the player's probability of hitting.
    # i.e. if the model is bullish, then use mid-market, and if the model is bearish, then use
    # this probability instead, since it's better to have false negatives than false positives.
    error_lines['model_probability'] = np.minimum(
        error_lines['model_probability'],
        error_lines['mid_market_broken_player_probability']
    )
    error_lines['adjusted_sgp_probability']  = (
        error_lines['mid_market_lock_probability'] * error_lines['model_probability']
    )
    error_lines = calculate_expected_value_vectorized(
        error_lines,
        'adjusted_sgp_probability',
        price_col='sgp_price',
        expected_value_col='sgp_expected_value')

    # Only care about the cases where the model thinks the E[V] is better than the market-based EV method.
    error_lines = error_lines[error_lines['sgp_expected_value'] > 0]
    error_lines['odds_api_name'] = error_lines['stat']
    save_pandas_df(
        error_lines[['player_name', 'matchup', 'market_type', 'line',
                     'model_probability', 'sgp_expected_value', 'odds_api_stat_name', 'price',
                     'sgp_price']],
        f'data/broken_mgm_sgp_lines/{NOW}.parquet',
        index=False
    )


if __name__ == '__main__':
    main()