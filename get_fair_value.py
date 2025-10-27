import os 
import pandas as pd 
from ag_draftking_utils.util import get_current_chicago_time
from ag_draftking_utils.odds_conversions import *
import warnings
import numpy as np
warnings.filterwarnings('ignore')

today = str(get_current_chicago_time().date())


def main():
    bets = []
    folder = f'data/nba/{today}'
    for file in os.listdir(folder):
        df = pd.read_parquet(os.path.join(folder, file))
        bets.append(df.copy())
    bets = pd.concat(bets)
    bets['implied_probability'] = american_odds_to_breakeven_probability_vectorized(
        bets, price_col='price')
    bets = get_opposite_price(bets)

    two_way_lines = bets[bets['opposite_price'].notnull()]
    one_way_lines = bets[bets['opposite_price'].isna()]

    two_way_lines['implied_probability_opposite'] = american_odds_to_breakeven_probability_vectorized(
        two_way_lines, price_col='opposite_price')

    two_way_lines['model_probability'] = (
        two_way_lines['implied_probability'] / 
        (two_way_lines['implied_probability'] + two_way_lines['implied_probability_opposite'])
    )

    one_way_lines['payout_given_win'] = american_odds_to_payout_given_win_vectorized(
        one_way_lines, price_col='price')
    # make some conservation assumptions on the hold percentage... 
    one_way_lines['xHold'] = np.where(
        one_way_lines['implied_probability'] < 0.05, -40,
        np.where(
            one_way_lines['implied_probability'] < 0.2, -20,
            np.where(
                one_way_lines['implied_probability'] < 0.45, 
                -15, 
                -10
            )
        )
    )
    # given the hold % and payouts, it solves for model probability
    one_way_lines['model_probability'] = (one_way_lines['xHold'] + 100) / (100 + one_way_lines['payout_given_win'])
    
    combined_bets = pd.concat([one_way_lines, two_way_lines])
    combined_bets = calculate_expected_value_vectorized(
        combined_bets, 'model_probability', price_col='price', expected_value_col='expected_value'
    )
    combined_bets = combined_bets.drop(columns=['xHold', 'payout_given_win', 'implied_probability_opposite'])
    combined_bets.to_parquet(f'data/straights_with_probabilities/{today}.parquet', index=False)


def get_opposite_price(bets):
    # now find the best price for the opposite bet...
    bets2 = bets.copy()
    bets2['opposite_player_name'] = np.where(
        bets2['market_type'].isin(['Over', 'Under']),
        bets2['player_name'],
        np.where(
            bets2['player_name'] == bets2['home_team'],
            bets2['away_team'],
            np.where(
                bets2['player_name'] == bets2['away_team'],
                bets2['home_team'],
                'UNKNOWN'
            )
        )
    )
    bets2['opposite_line'] = np.where(
        bets2['stat'].str.contains('spread'),
        -bets2['line'],
        bets2['line']
    )
    bets2['opposite_market_type'] = np.where(
        bets2['market_type'] == '', 
        '',
        np.where(
            bets2['market_type'] == 'Over',
            'Under',
            np.where(
                bets2['market_type'] == 'Under', 'Over', 'unknown'
            )
        )
    )
    bets = bets.merge(
        bets2[[
                'id', 'bookie', 'stat', 'opposite_player_name', 'opposite_line', 'opposite_market_type', 'price']].rename(
            columns={
                'price': 'opposite_price',
                'opposite_player_name': 'player_name',
                'opposite_line': 'line',
                'opposite_market_type': 'market_type'
            }
        ),
        how='left',
        on=['id', 'bookie', 'stat', 'player_name', 'line', 'market_type']
    )
    return bets


if __name__ == '__main__':
    main()
