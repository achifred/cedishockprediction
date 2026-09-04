import numpy as np
import pandas as pd

from src.config.looger_config import get_logger
from src.config.config import COMBINED_DATASET_FILE , USDGH_DATASET_FILE
from src.features.feature_engineering import feature_engineer_data, get_correlation_matrix
from src.features.feature_engineering_new import engineer_features
from src.utils.utils import get_path

logger = get_logger(name=__name__, log_file="feature_engineering.log")
if __name__ == "__main__":

  file_path  = get_path(COMBINED_DATASET_FILE)
  data = pd.read_csv(file_path)
  # df = feature_engineer_data(file_path = COMBINED_DATASET_FILE, output_file = USDGH_DATASET_FILE)
  engineer_features(df=data, output_file=USDGH_DATASET_FILE, logger=logger)

  # get_correlation_matrix(df = df)
  # get_range(df = df)

  # df.index = pd.to_datetime( df.index )
  # df[ 'period' ] = np.where( df.index < '2022-01-01' , 'pre_2022' , 'post_2022' )
  #
  # # --- Table 1: Summary stats of tbill_mpr_spread, split by period AND shock label ---
  # summary = df.groupby( [ 'period' , 'target_shock_window' ] )[ 'tbill_mpr_spread_change' ].agg(
  #   count = 'count' ,
  #   mean = 'mean' ,
  #   median = 'median' ,
  #   std = 'std' ,
  #   min = 'min' ,
  #   max = 'max'
  # ).round( 2 )
  #
  # print( "Table 1: Spread distribution by period and shock label" )
  # print( summary )
  # print()
  #
  # # --- Table 2: Binned spread ranges vs shock rate, split by period ---
  # # This is the key table -- it directly shows whether a given spread LEVEL
  # # corresponds to a similar shock rate before vs after 2022
  #
  # bins = [ -16 , -10 , -5 , 0 , 5 , 10 , 16 ]  # adjust to cover your full range
  # labels = [ '<-10' , '-10 to -5' , '-5 to 0' , '0 to 5' , '5 to 10' , '>10' ]
  #
  # df[ 'spread_bin' ] = pd.cut( df[ 'tbill_mpr_spread_change' ] , bins = bins , labels = labels )
  #
  # shock_rate_table = df.groupby( [ 'spread_bin' , 'period' ] )[ 'target_shock_window' ].agg(
  #   n_obs = 'count' ,
  #   n_shocks = 'sum' ,
  #   shock_rate = 'mean'
  # ).round( 3 )
  #
  # shock_rate_table[ 'shock_rate_pct' ] = (shock_rate_table[ 'shock_rate' ] * 100).round( 1 )
  #
  # print( "Table 2: Shock rate by spread bin, split by period" )
  # print( shock_rate_table )
  # print()
  #
  # # --- Table 3: Lead-lag view -- spread level in the weeks BEFORE each shock ---
  # # Shows whether shocks are preceded by a similar spread signature in each period
  #
  # shock_dates = df[ df[ 'target_shock_window' ] == 1 ].index
  #
  # lookback_weeks = 4  # matches your 2-4 week prediction horizon
  # rows = [ ]
  # for d in shock_dates :
  #   window = df.loc[ d - pd.Timedelta( weeks = lookback_weeks ) :d , 'tbill_mpr_spread_change' ]
  #   period = 'pre_2022' if d < pd.Timestamp( '2022-01-01' ) else 'post_2022'
  #   rows.append( {
  #     'shock_date' : d ,
  #     'period' : period ,
  #     'spread_mean_before' : window.mean().round( 2 ) ,
  #     'spread_min_before' : window.min().round( 2 ) ,
  #     'spread_max_before' : window.max().round( 2 )
  #   } )
  #
  # lead_lag_table = pd.DataFrame( rows )
  # print( f"Table 3: Spread in {lookback_weeks} weeks before each shock" )
  # print( lead_lag_table.to_string( index = False ) )