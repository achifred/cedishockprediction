from logging import Logger
from pathlib import Path

import numpy as np
import pandas as pd


from src.utils.utils import convert_volume , save_to_csv , slice_data

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATE_COL = "date"
FX_COL = "usdgh_exchange_rate"

# Forecast window:
#  2 weeks
#  4 weeks
START_HORIZON = 10
END_HORIZON = 20

SHOCK_THRESHOLD = 0.03

CRUDE_CLOSE_PRICE_COL = "crude_close_price"
CRUDE_VOLUME_COL = "crude_volume"
GOLD_CLOSE_PRICE_COL = "gold_close_price"
GOLD_VOLUME_COL = "gold_volume"


# ============================================================
# 2. PREPARE DATA
# ============================================================

def prepare_data(df):
    df = df.copy()

    # Convert date
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Sort chronologically
    df = df.sort_values(DATE_COL)

    # Remove duplicate dates
    df = df.drop_duplicates(subset=[DATE_COL])

    # Set date index
    df = df.set_index(DATE_COL)

    df[ CRUDE_CLOSE_PRICE_COL ] = df[ CRUDE_CLOSE_PRICE_COL ].astype( str ).str.replace( ',' , '' , regex = False ).astype(
      float )
    df[ GOLD_CLOSE_PRICE_COL ] = df[ GOLD_CLOSE_PRICE_COL ].astype( str ).str.replace( ',' , '' , regex = False ).astype(
      float )
    df[ CRUDE_VOLUME_COL ] = df[ CRUDE_VOLUME_COL ].apply( convert_volume )
    df[ GOLD_VOLUME_COL ] = df[ GOLD_VOLUME_COL ].apply( convert_volume )

    df[ FX_COL ] = pd.to_numeric( df[ FX_COL ] , errors = 'coerce' )

    return df


# ============================================================
# 3. USD/GHS FEATURES
# ============================================================

def create_fx_features(df):

    df = df.copy()

    fx = df[FX_COL]


    # --------------------------------------------------------
    # Log return
    # --------------------------------------------------------
    # log_exchange_rate = np.log(fx)
    # df["usd_ghs_log_return_4w"] = np.log( fx / fx.shift( 4 ) )
    df[ "usd_ghs_log_return_1w" ] = np.log( fx / fx.shift( 1 ) )


    # --------------------------------------------------------
    # Percentage returns
    # --------------------------------------------------------

    # for window in [4]:
    #
    #     df[f"usd_ghs_return_{window}w"] = (
    #         fx / fx.shift(window) - 1
    #     )



    # --------------------------------------------------------
    # Depreciation acceleration
    # --------------------------------------------------------

    # df["usd_ghs_depreciation_acceleration"] = (
    #     df["usd_ghs_return_5w"] -
    #     df["usd_ghs_return_20w"]
    # )

    # --------------------------------------------------------
    # Recent maximum depreciation
    # --------------------------------------------------------
    # df[ "usd_ghs_max_10w_depreciation" ] = (fx / fx.rolling( 14 ).min()) - 1
    df[ "usd_ghs_12w_depreciation" ] = (
      fx / fx.rolling( 30 ).min() - 1
    )

    return df


# ============================================================
# 4. INFLATION FEATURES
# ============================================================

def create_inflation_features(df):

    df = df.copy()

    inflation_cols = [
        "food_inf",
        "non_food_inf"
    ]

    for col in inflation_cols :
      if col in df.columns :
        df[ f"{col}_change" ] = df[ col ].diff(  )




    return df


# ============================================================
# 5. INTEREST RATE FEATURES
# ============================================================

def create_interest_rate_features(df):

    df = df.copy()

    rate_cols = [
      "monetary_policy_rate" ,
      "tbill" ,
      "interbank_interest_rate" ,
      "us_federal_fund_rate"
    ]

    for col in rate_cols :

      if col not in df.columns :
        continue

      for w in [90]:
        df[ f"{col}_{w}period_change" ] = (
          df[ col ] - df[ col ].shift( w)
        )

    # --------------------------------------------------------
    # T-bill vs policy rate
    # --------------------------------------------------------

    if (
        "tbill" in df.columns and
        "monetary_policy_rate" in df.columns
    ):

        df["tbill_policy_spread"] = df["tbill"] -df["monetary_policy_rate"]


    # --------------------------------------------------------
    # Interbank vs policy rate
    # --------------------------------------------------------

    # if (
    #     "interbank_interest_rate" in df.columns and
    #     "tbill" in df.columns
    # ):
    #
    #     df["interbank_policy_spread"] = df["interbank_interest_rate"] - df["tbill"]



    # --------------------------------------------------------
    # Ghana-US interest rate differential
    # --------------------------------------------------------

    # if (
    #     "monetary_policy_rate" in df.columns and
    #     "us_federal_fund_rate" in df.columns
    # ):
    #
    #     df["ghana_us_rate_differential"] =  df["monetary_policy_rate"] - df["us_federal_fund_rate"]


        # df["ghana_us_rate_differential_change"] = df["ghana_us_rate_differential"].diff()

    return df


# ============================================================
# 6. NET INTERNATIONAL RESERVE FEATURES
# ============================================================

def create_reserve_features(df):

    df = df.copy()

    col = "net_international_reserve"

    if col in df.columns:

      # Changes over available periods
      df["nir_momentum_8w"] =  df[col].pct_change(30)
      #df[ 'nir_momentum_90d' ] = df[ col ].pct_change( 90 )


    return df


# ============================================================
# 7. GOLD FEATURES
# ============================================================

def create_commodity_features ( df: pd.DataFrame , price_col: str , volume_col: str , prefix: str ) -> pd.DataFrame :
  df = df.copy()
  if price_col not in df.columns :
    return df

  # Raw returns and pure linear standard deviation (NO LOG VARIANCE PENALTY)

  df[ f"{prefix}_return_1w" ] = np.log(
    df[ price_col ] / df[ price_col ].shift( 1 )
  )

  for w in [30]:
    df[ f"{prefix}_return_{w}w" ] =np.log(
    df[ price_col ] / df[ price_col ].shift( w )
  )





  return df



def create_debt_features(df:pd.DataFrame ):
  df = df.copy()
  df['total_debt_to_gdp_change'] = df['total-public-debt-to-gdp'].diff(12)
  # df['domestic_debt_to_gdp_change'] = df['domestic-debt-to-gdp'].diff(90)
  external_share = df['external-debt-to-gdp'] / df['total-public-debt-to-gdp']
  df['external_debt_share_change'] = external_share.diff( 8 )
  # df['debt_composition_ratio'] =  external_change / domestic_change

  # changed = df[ 'total-public-debt-to-gdp' ].diff().fillna( 0 ) != 0
  # changed.iloc[ 0 ] = True
  # df[ 'days_since_debt_update' ] = changed.cumsum().pipe( lambda g : df.groupby( g ).cumcount() )

  return df



# ============================================================
# 9. CREATE FUTURE SHOCK TARGET
# ============================================================

def create_shock_target (
  df
) :
  """
  Create a forward-looking USD/GHS shock target.

  A shock occurs if USD/GHS depreciates by at least
  `shock_threshold` at any point between +10 and +20
  trading days from the prediction date.
  """

  df = df.copy()

  # Future horizons: +10, +11, ..., +20 trading days
  future_prices = pd.concat(
    {
      h : df[ FX_COL ].shift( -h )
      for h in range( START_HORIZON , END_HORIZON + 1 )
    } ,
    axis = 1
  )

  # Maximum USD/GHS rate within the future prediction window
  future_max_usdgh = future_prices.max( axis = 1 )

  # Ensure the entire +10 to +20 window exists
  complete_window = future_prices.notna().all( axis = 1 )

  # Maximum future depreciation relative to today's USD/GHS
  future_max_depreciation = ( future_max_usdgh / df[ FX_COL ]) - 1

  # Binary shock target
  df[ "target_shock_window" ] = np.where(
    complete_window ,
    (future_max_depreciation >= SHOCK_THRESHOLD).astype( int ) ,
    np.nan
  )

  return df


# def define_shock_target( df:pd.DataFrame ) -> pd.DataFrame:
#   df[ 'usd_gh_30ma' ] = df[ 'usdgh_exchange_rate' ].rolling( 30 ).mean()
#   # calculating 5 days rolling depreciation relative to the 30 days moving average
#   df[ 'percentage_rolling_depreciation_5d' ] = (df[ 'usdgh_exchange_rate' ] - df[ 'usd_gh_30ma' ].shift( 5 )) / df[
#     'usd_gh_30ma' ].shift( 5 )
#
#   # calculating the possibility of a shock for historical data by checking if the 5 days rolling depreciation
#   # percentage
#   # is
#   # greater that 5%
#   df[ 'is_shock_occurring' ] = (df[ 'percentage_rolling_depreciation_5d' ] >= 0.03).astype( int )
#
#   # defining the target variable for forward-looking window. that is predicting if a shock will happen 2 to 4 weeks
#   # ahead. Two weeks(14 days) and for weeks(28 days).
#   window_start_days = 14  # 2 weeks
#   window_end_days = 28  # 4 weeks
#   window_len = window_end_days - window_start_days + 1  # 15
#   df[ 'target_shock_window' ] = df[ 'is_shock_occurring' ].shift( -window_start_days ).rolling( window_len ,
#                                                                                                 min_periods = window_len ).max().shift(
#     -window_start_days )
#   return df






# ============================================================
# 10. COMPLETE FEATURE ENGINEERING PIPELINE
# ============================================================

def engineer_features(df:pd.DataFrame, output_file:str|Path,logger:Logger) -> pd.DataFrame:

    df = prepare_data(df)



    # FX features
    df = create_fx_features(df)

    # Inflation features
    df = create_inflation_features(df)
    #
    # # Interest rates features
    df = create_interest_rate_features(df)

    # # Reserves features
    df = create_reserve_features(df)

    # # commodity features
    df = create_commodity_features( df , GOLD_CLOSE_PRICE_COL , GOLD_VOLUME_COL , "gold" )
    df = create_commodity_features( df , CRUDE_CLOSE_PRICE_COL , CRUDE_VOLUME_COL , "oil" )
    # df = create_debt_features(df)

    # Shock target
    df = create_shock_target( df )



    features_to_drop = [ 'crude_volume' , 'gold_volume' , 'monetary_policy_rate' , 'us_federal_fund_rate' , 'tbill' ,
                         'interbank_interest_rate' , 'gold_close_price' , 'crude_close_price' ,
                           'usdgh_exchange_rate' ,
                         'net_international_reserve' , 'food_inf' , 'non_food_inf' ,
                          'future_max_depreciation','future_max_usdgh',
                         'gold_volume_ma_20d','crude_volume_ma_20d',
                         'is_shock_occurring','percentage_rolling_depreciation_5d','usd_gh_30ma',
                         'domestic-debt-to-gdp','external-debt-to-gdp','total-public-debt-to-gdp']


    df = df.drop( columns = [col for col in features_to_drop if col in df.columns] )
    # df  = df.dropna()
    df = df.resample( 'W' ).last()
    df = slice_data( start_date = "2010-01-01" , end_date = "2025-12-31" , data = df )

    df = df.dropna()

    save_to_csv( output_file = output_file , data = df , logger = logger , set_index = True )

    return df