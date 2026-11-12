import pandas as pd
import numpy as np

from src.features.feature_engineering_weekly import prepare_data
from src.utils.utils import save_to_csv , slice_data


def clean_numeric_string ( series ) :
  """
  Cleans string columns containing commas, 'K', or 'M' into clean floats.
  """
  if series.dtype == object :
    series = series.astype( str ).str.replace( ',' , '' , regex = False )

    # Safely convert 'K' and 'M' text scales to float multipliers
    mask_k = series.str.endswith( 'K' , na = False )
    mask_m = series.str.endswith( 'M' , na = False )

    series = series.str.replace( 'K' , '' , regex = False ).str.replace( 'M' , '' , regex = False )
    series = pd.to_numeric( series , errors = 'coerce' )

    series = np.where( mask_k , series * 1e3 , series )
    series = np.where( mask_m , series * 1e6 , series )
  return pd.to_numeric( series , errors = 'coerce' )


def pipeline_feature_engineering ( df:pd.DataFrame, output_file, logger ) :
  """
  Reads daily raw financial data, generates rolling indicators, constructs
  the target safe from leakage, and downsamples logically to a weekly view.
  """
  # 1. Load data and parse foundational schema
  start_horizon = 10
  end_horizon = 20
  shock_threshold = 0.03


  df = prepare_data(df)
  print(f"head: {df.head}")

  # Explicitly forward fill macro blocks that are updated monthly/quarterly
  macro_cols = [ 'food_inf' , 'non_food_inf' , 'external-debt-to-gdp' ,
                 'domestic-debt-to-gdp' , 'total-public-debt-to-gdp' ,
                 'monetary_policy_rate' , 'net_international_reserve' , 'tbill' ]
  df[ macro_cols ] = df[ macro_cols ].ffill().bfill()

  # 2. Target Creation (Calculated strictly on Daily Trading Days to avoid leakage)
  fx_col = "usdgh_exchange_rate"

  future_prices = pd.concat(
    { h : df[ fx_col ].shift( -h ) for h in range( start_horizon , end_horizon + 1 ) } ,
    axis = 1
  )
  future_max_usdgh = future_prices.max( axis = 1 )
  complete_window = future_prices.notna().all( axis = 1 )
  future_max_depreciation = (future_max_usdgh / df[ fx_col ]) - 1

  df[ "target_shock_window" ] = np.where(
    complete_window ,
    (future_max_depreciation >= shock_threshold).astype( int ) ,
    np.nan
  )

  # 3. Daily Feature Engineering (Historical metrics strictly safe from leakage)
  # Volatility and Return momentum on currency
  df[ 'usd_ghs_log_return_1d' ] = np.log( df[ fx_col ] / df[ fx_col ].shift( 1 ) )

  # Commodity Returns (Gold and Oil signals)
  df[ 'gold_ret_daily' ] = df[ 'gold_close_price' ].pct_change()
  df[ 'crude_ret_daily' ] = df[ 'crude_close_price' ].pct_change()

  # 4. Resample carefully to Weekly cadences (Using Sunday to match model expectation)
  # Set Date index temporarily for clean groupby resampling
  #df.set_index( 'date' , inplace = True )

  # Group using rules that reflect end-of-period availability (no looking into future days)
  weekly_df = pd.DataFrame()

  # Market close rates are represented by the final day's actual recorded sequence
  weekly_df[ 'usd_ghs_close' ] = df[ fx_col ].resample( 'W' ).last()
  weekly_df[ 'gold_close' ] = df[ 'gold_close_price' ].resample( 'W' ).last()
  weekly_df[ 'crude_close' ] = df[ 'crude_close_price' ].resample( 'W' ).last()

  # Macro metrics are snapshot tracking metrics at the exact week boundaries
  weekly_df[ 'food_inf' ] = df[ 'food_inf' ].resample( 'W' ).last()
  weekly_df[ 'non_food_inf' ] = df[ 'non_food_inf' ].resample( 'W' ).last()
  weekly_df[ 'monetary_policy_rate' ] = df[ 'monetary_policy_rate' ].resample( 'W' ).last()
  weekly_df[ 'tbill' ] = df[ 'tbill' ].resample( 'W' ).last()
  weekly_df[ 'interbank_interest_rate' ] = df[ 'interbank_interest_rate' ].resample( 'W' ).last()
  weekly_df[ 'us_federal_fund_rate' ] = df[ 'us_federal_fund_rate' ].resample( 'W' ).last()
  weekly_df[ 'net_international_reserve' ] = df[ 'net_international_reserve' ].resample( 'W' ).last()

  # Grab target flag anchored to the end of the current baseline calculation weekend
  weekly_df[ 'target_shock_window' ] = df[ 'target_shock_window' ].resample( 'W' ).last()
  print(f"week: {weekly_df.head()}")
  # 5. Engineering High-Signal Weekly Features (Matches target schema values)
  # Weekly returns
  weekly_df[ 'usd_ghs_log_return_1w' ] = np.log(
    weekly_df[ 'usd_ghs_close' ] / weekly_df[ 'usd_ghs_close' ].shift( 1 ) )
  weekly_df[ 'usd_ghs_12w_depreciation' ] = (weekly_df[ 'usd_ghs_close' ] / weekly_df[ 'usd_ghs_close' ].shift(
    12 )) - 1

  # Rates change differentials (1-period differences on monthly scales match velocity trends)
  weekly_df[ 'food_inf_change' ] = weekly_df[ 'food_inf' ].diff( 1 )
  weekly_df[ 'non_food_inf_change' ] = weekly_df[ 'non_food_inf' ].diff( 1 )

  # Medium-term momentum macro variables (8-week lookback intervals)
  weekly_df[ 'monetary_policy_rate_8period_change' ] = weekly_df[ 'monetary_policy_rate' ].diff( 8 )
  weekly_df[ 'tbill_8period_change' ] = weekly_df[ 'tbill' ].diff( 8 )
  weekly_df[ 'interbank_interest_rate_8period_change' ] = weekly_df[ 'interbank_interest_rate' ].diff( 8 )
  weekly_df[ 'us_federal_fund_rate_8period_change' ] = weekly_df[ 'us_federal_fund_rate' ].diff( 8 )

  # Domestic Interest Spread metric
  weekly_df[ 'tbill_policy_spread' ] = weekly_df[ 'tbill' ] - weekly_df[ 'monetary_policy_rate' ]

  # International Asset Reserves and Global Commodities momentum tracking
  weekly_df[ 'nir_momentum_8w' ] = (weekly_df[ 'net_international_reserve' ] / weekly_df[
    'net_international_reserve' ].shift( 8 )) - 1
  weekly_df[ 'gold_return_1w' ] = weekly_df[ 'gold_close' ].pct_change( 1 )
  weekly_df[ 'gold_return_12w' ] = weekly_df[ 'gold_close' ].pct_change( 12 )
  weekly_df[ 'crude_return_1w' ] = weekly_df[ 'crude_close' ].pct_change( 1 )
  weekly_df[ 'crude_return_12w' ] = weekly_df[ 'crude_close' ].pct_change( 12 )
  print(f"weekly: {weekly_df.head()}")
  # Clean up index format and return structural matrix matching usdgh file definitions
  weekly_df = weekly_df.reset_index()
  print(f"index weekly: {weekly_df.head()}")
  # Select final structured columns in exact order
  final_cols = [
    'date' , 'usd_ghs_log_return_1w' , 'usd_ghs_12w_depreciation' , 'food_inf_change' ,
    'non_food_inf_change' , 'monetary_policy_rate_8period_change' , 'tbill_8period_change' ,
    'interbank_interest_rate_8period_change' , 'us_federal_fund_rate_8period_change' ,
    'tbill_policy_spread' , 'nir_momentum_8w' , 'gold_return_1w' , 'gold_return_12w' ,
    'crude_return_1w' , 'crude_return_12w' , 'target_shock_window'
  ]

  df = weekly_df[ final_cols ]
  print( f"samples: {df}" )

  # df = df.drop( columns = [col for col in features_to_drop if col in df.columns] )

  # df = df.loc[pd.to_datetime("2010-01-01"):pd.to_datetime("2025-12-31")]
#slice_data( start_date = "2010-01-01" , end_date = "2025-12-31" , data = df )
  print( f"samplesnew: {df}" )

  # df = df.dropna()

  save_to_csv( output_file = output_file , data = df , logger = logger , set_index = True )
  return df
# Execution Example:
# processed_df = pipeline_feature_engineering("combined_data.csv")
# processed_df.to_csv("model_features_weekly.csv", index=False)
