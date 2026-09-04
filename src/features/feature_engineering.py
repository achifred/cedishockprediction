import pandas as pd
import numpy as np

from src.config.config import FEATURE_NAMES
from src.config.looger_config import get_logger
from src.utils.utils import convert_volume , get_path
from src.utils.utils import save_to_csv, slice_data

logger = get_logger(name = __name__, log_file = "feature_engineering.log")

def feature_engineer_data(file_path:str, output_file:str):

  try:
    source_file = get_path(file_path)
    df = pd.read_csv(source_file)

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    df['usdgh_exchange_rate'] = pd.to_numeric(df['usdgh_exchange_rate'], errors='coerce')

    # defining 30 days moving average of the usd/gh exchange rate
    df['usd_gh_30ma'] = df['usdgh_exchange_rate'].rolling(30).mean()
    # calculating 5 days rolling depreciation relative to the 30 days moving average
    df['percentage_rolling_depreciation_5d'] = (df['usdgh_exchange_rate'] - df['usd_gh_30ma'].shift(5)) / df[
      'usd_gh_30ma'].shift(5)

    # calculating the possibility of a shock for historical data by checking if the 5 days rolling depreciation
    # percentage
    # is
    # greater that 5%
    df['is_shock_occurring'] = (df['percentage_rolling_depreciation_5d'] >= 0.05).astype(int)

    # defining the target variable for forward-looking window. that is predicting if a shock will happen 2 to 4 weeks
    # ahead. Two weeks(14 days) and for weeks(28 days).
    window_start_days = 14  # 2 weeks
    window_end_days = 28  # 4 weeks
    window_len = window_end_days - window_start_days + 1  # 15
    df['target_shock_window'] = df['is_shock_occurring'].shift(-window_start_days).rolling(window_len,min_periods=window_len).max().shift(
      -window_start_days)

    # stale_source_cols = [
    #    'non_food_inf' , 'tbill' ,
    #   'monetary_policy_rate' , 'net_international_reserve'
    # ]
    # for col in stale_source_cols:
    #   changed = df[ col ].diff().fillna( 0 ) != 0
    #   changed.iloc[ 0 ] = True  # anchor the first row as a "change" to start counting from
    #   group_id = changed.cumsum()
    #   df[ f'days_since_{col}_update' ] = df.groupby( group_id ).cumcount()

    # calculating daily return of gold and oil prices
    df['gold_close_price'] = df['gold_close_price'].astype(str).str.replace(',','', regex=False).astype(float)
    df['crude_close_price'] = df['crude_close_price'].astype(str).str.replace(',','', regex=False).astype(float)
    df['gold_return'] = np.log1p(df['gold_close_price']).diff()
    df['crude_return'] = np.log1p(df['crude_close_price']).diff()

    inflation_horizon_days = 30
    # df['food_inf_change'] =  df['food_inf'].diff(inflation_horizon_days)
    df['non_food_inf_change'] = df['non_food_inf'].diff(inflation_horizon_days)


    # normalizing gold and oil volumes
    df['crude_volume'] = df['crude_volume'].apply(convert_volume)
    df['gold_volume'] = df['gold_volume'].apply(convert_volume)
    df['crude_log_volume'] = np.log1p(df['crude_volume'])
    df['gold_log_volume'] = np.log1p(df['gold_volume'])

    # calculates the sovereign risk spread: that is the difference between what the government pays to borrow from the public(treasury bill rate) and baseline rate set by the central bank(monetary policy rate)
    spread_horizon_days = 30
    df['tbill_mpr_spread'] = df['tbill'] - df['monetary_policy_rate']
    df['tbill_mpr_spread_change']  = df['tbill_mpr_spread'].diff(spread_horizon_days)

    reserve_momentum_horizon_days = 90
    df[ 'net_reserve_momentum' ] = df[ 'net_international_reserve' ].pct_change( reserve_momentum_horizon_days )
    df[ 'net_reserve_momentum_30d' ] = df[ 'net_international_reserve' ].pct_change( 30 )
    # df[ 'net_reserve_momentum_60d' ] = df[ 'net_international_reserve' ].pct_change( 60 )
    # df[ 'severe_reserve_depletion' ] = (df[ 'net_reserve_momentum' ] < -0.2).astype( int )

    # reserve LEVEL adequacy, distinct from momentum. Motivated by
    # 2025 episode research pointing at reserve-adequacy anxiety even
    # while reserves were technically stable/growing -- a related but
    # distinct story from "reserves are actively falling."
    # RESERVE_LEVEL_WINDOW = 365
    # roll_mean = df[ 'net_international_reserve' ].rolling( RESERVE_LEVEL_WINDOW , min_periods = 90 ).mean().shift( 1 )
    # roll_std = df[ 'net_international_reserve' ].rolling( RESERVE_LEVEL_WINDOW , min_periods = 90 ).std().shift( 1 )
    # df[ 'net_reserve_level_zscore' ] = (df[ 'net_international_reserve' ] - roll_mean) / (roll_std + 1e-8)

    # calculates the global yield spread: that is the difference between ghana's central bank interest rate and
    # United States' central bank interest rate
    df['global_rate_gap'] = df['monetary_policy_rate'] - df['us_federal_fund_rate']
    df['global_rate_gap_change'] = df['global_rate_gap'].diff(30)

    # calculates the bank liquidity premium: that is the gap between the rate government pays for three months loans
    # and what banks charge each other for overnight loans
    # df['banking_premium_gap'] = df['tbill'] - df['interbank_interest_rate']

    # Interbank vs policy rate: how much banks are paying each other for cash,
    # relative to where the central bank has set the policy rate. A widening
    # gap signals liquidity stress the policy rate hasn't caught up to yet.
    INTERBANK_HORIZON_DAYS = 30  # matches its ~31-day real update cadence

    df[ 'interbank_mpr_spread' ] = df[ 'interbank_interest_rate' ] - df[ 'monetary_policy_rate' ]
    df[ 'interbank_mpr_spread_change' ] = df[ 'interbank_mpr_spread' ].diff( INTERBANK_HORIZON_DAYS )

    # Interbank vs T-bill: banks' overnight cost of cash vs. what government
    # pays for 91-day borrowing -- this is the "banking premium gap" your
    # original code had commented out (df['tbill_mpr_spread'] left the
    # interbank version unbuilt). Same treatment: change, not level.
    df[ 'interbank_tbill_spread' ] = df[ 'interbank_interest_rate' ] - df[ 'tbill' ]
    df[ 'interbank_tbill_spread_change' ] = df[ 'interbank_tbill_spread' ].diff( INTERBANK_HORIZON_DAYS )

    # staleness, same convention as your other monthly variables
    # changed = df[ 'interbank_interest_rate' ].diff().fillna( 0 ) != 0
    # changed.iloc[ 0 ] = True
    # df[ 'days_since_interbank_update' ] = changed.cumsum().pipe( lambda g : df.groupby( g ).cumcount() )

    #dropping features that could confuse the model and  introduce noise
    features_to_drop = ['crude_volume','gold_volume','monetary_policy_rate','us_federal_fund_rate','tbill',
                       'interbank_interest_rate','gold_close_price','crude_close_price','is_shock_occurring',
                       'percentage_rolling_depreciation_5d','usd_gh_30ma','usdgh_exchange_rate',
                        'net_international_reserve','food_inf','non_food_inf','tbill_mpr_spread',
                        'interbank_mpr_spread','global_rate_gap']

    df = df.drop(columns = features_to_drop)
    # df = df.dropna()
    df = slice_data(start_date = "2010-01-01", end_date = "2025-12-31", data = df)
    save_to_csv(output_file = output_file, data = df, logger = logger, set_index = True)
    return df

  except Exception as e:
    logger.error(e)



def get_correlation_matrix(df:pd.DataFrame):
    try:
      feature_cols = list(df.drop( columns = [ "target_shock_window" ] ).columns)
      correlation_matrix = df.corr()
      strong_correlation = correlation_matrix.unstack()
      strong_pairs = strong_correlation[(abs(strong_correlation) > 0.8) & (strong_correlation !=1)]
      print(f"strongly correlated features: {strong_pairs}")


      # correlations = df[
      #   feature_cols + [ 'target_shock_window' ]
      #   ].corr()[ 'target_shock_window' ].sort_values(
      #   ascending = False
      # )
      # print(f"correlation matrix: {correlations}")

    except Exception as e:
      print(e)


def get_range(df:pd.DataFrame):

  train = df.loc[ '2010-01-01' :'2021-12-31' ]
  val = df.loc[ '2022-01-01' :'2023-12-31' ]
  test = df.loc[ '2024-01-01' :'2025-12-31' ]
  for col in FEATURE_NAMES :
    train_min , train_max = train[ col ].min() , train[ col ].max()
    test_min , test_max = test[ col ].min() , test[ col ].max()
    pct_test_outside = (
      ((test[ col ] < train_min) | (test[ col ] > train_max)).mean() * 100
    )
    print( f"{col}: train range [{train_min:.2f}, {train_max:.2f}], "
           f"test range [{test_min:.2f}, {test_max:.2f}], "
           f"{pct_test_outside:.1f}% of test points outside train range" )
