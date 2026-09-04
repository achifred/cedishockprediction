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
# 10 trading days ≈ 2 weeks
# 20 trading days ≈ 4 weeks
MIN_HORIZON = 10
MAX_HORIZON = 20

# Economic shock threshold
ABSOLUTE_SHOCK_THRESHOLD = 0.05   # 5% depreciation

# Volatility-adjusted threshold
VOLATILITY_MULTIPLIER = 2.0

# Historical volatility window
VOL_WINDOW = 60


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

    return df


# ============================================================
# 3. USD/GHS FEATURES
# ============================================================

def create_fx_features(df):

    df = df.copy()
    df[ FX_COL ] = pd.to_numeric( df[ FX_COL ] , errors = 'coerce' )

    fx = df[FX_COL]


    # --------------------------------------------------------
    # Log return
    # --------------------------------------------------------

    df["usd_ghs_log_return_1d"] = np.log(fx / fx.shift(1))

    # --------------------------------------------------------
    # Percentage returns
    # --------------------------------------------------------

    for window in [3, 5, 10, 20, 30]:

        df[f"usd_ghs_return_{window}d"] = (
            fx / fx.shift(window) - 1
        )

    # --------------------------------------------------------
    # Rolling volatility
    # --------------------------------------------------------

    # daily_return = df["usd_ghs_log_return_1d"]
    #
    # for window in [5, 10, 20, 30, 60]:
    #
    #     df[f"usd_ghs_vol_{window}d"] = (
    #         daily_return
    #         .rolling(window)
    #         .std()
    #     )

    # --------------------------------------------------------
    # Volatility acceleration
    # --------------------------------------------------------

    # df["usd_ghs_vol_ratio_10_60"] = (
    #     df["usd_ghs_vol_10d"] /
    #     df["usd_ghs_vol_60d"]
    # )
    #
    # df["usd_ghs_vol_ratio_20_60"] = (
    #     df["usd_ghs_vol_20d"] /
    #     df["usd_ghs_vol_60d"]
    # )

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    for window in [10, 20, 30]:

        df[f"usd_ghs_ma_{window}d"] = (
            fx.rolling(window).mean()
        )

    # --------------------------------------------------------
    # Distance from moving averages
    # --------------------------------------------------------

    for window in [20]:

        df[f"usd_ghs_ma_distance_{window}d"] = (
            fx / df[f"usd_ghs_ma_{window}d"] - 1
        )

    # --------------------------------------------------------
    # Z-scores
    # --------------------------------------------------------

    # for window in [20, 60]:
    #
    #     rolling_mean = fx.rolling(window).mean()
    #     rolling_std = fx.rolling(window).std()
    #
    #     df[f"usd_ghs_zscore_{window}d"] = (
    #         (fx - rolling_mean) / rolling_std
    #     )

    # --------------------------------------------------------
    # Depreciation acceleration
    # --------------------------------------------------------

    df["usd_ghs_depreciation_acceleration"] = (
        df["usd_ghs_return_5d"] -
        df["usd_ghs_return_20d"]
    )

    # --------------------------------------------------------
    # Recent maximum depreciation
    # --------------------------------------------------------

    df["usd_ghs_max_5d_depreciation"] = (
        fx / fx.rolling(5).min() - 1
    )

    df["usd_ghs_max_20d_depreciation"] = (
        fx / fx.rolling(20).min() - 1
    )

    # --------------------------------------------------------
    # Number of recent depreciating days
    # --------------------------------------------------------

    # df["usd_ghs_depreciation_days_5d"] = (
    #     (daily_return > 0)
    #     .rolling(5)
    #     .sum()
    # )
    #
    # df["usd_ghs_depreciation_days_10d"] = (
    #     (daily_return > 0)
    #     .rolling(10)
    #     .sum()
    # )

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

    for col in inflation_cols:

        if col not in df.columns:
            continue

        # Change from previous available observation
        df[f"{col}_change"] = df[col].diff()

        # Longer-term changes
        # df[f"{col}_3period_change"] = (
        #     df[col] - df[col].shift(3)
        # )
        #
        # df[f"{col}_6period_change"] = (
        #     df[col] - df[col].shift(6)
        # )

    # Food vs non-food inflation gap

    if (
        "food_inf" in df.columns and
        "non_food_inf" in df.columns
    ):

        df["food_nonfood_inflation_gap"] = (
            df["food_inf"] -
            df["non_food_inf"]
        )

    return df


# ============================================================
# 5. INTEREST RATE FEATURES
# ============================================================

def create_interest_rate_features(df):

    df = df.copy()

    rate_cols = [
        "monetary_policy_rate",
        "tbill",
        "interbank_interest_rate",
        "us_federal_fund_rate"
    ]

    for col in rate_cols:

        if col not in df.columns:
            continue

        df[f"{col}_change"] = df[col].diff()

        df[f"{col}_3period_change"] = (
            df[col] - df[col].shift(3)
        )

        df[f"{col}_6period_change"] = (
            df[col] - df[col].shift(6)
        )

    # --------------------------------------------------------
    # T-bill vs policy rate
    # --------------------------------------------------------

    if (
        "tbill" in df.columns and
        "monetary_policy_rate" in df.columns
    ):

        df["tbill_policy_spread"] = (
            df["tbill"] -
            df["monetary_policy_rate"]
        )

    # --------------------------------------------------------
    # Interbank vs policy rate
    # --------------------------------------------------------

    if (
        "interbank_interest_rate" in df.columns and
        "monetary_policy_rate" in df.columns
    ):

        df["interbank_policy_spread"] = (
            df["interbank_interest_rate"] -
            df["monetary_policy_rate"]
        )

    # --------------------------------------------------------
    # Ghana-US interest rate differential
    # --------------------------------------------------------

    if (
        "monetary_policy_rate" in df.columns and
        "us_federal_fund_rate" in df.columns
    ):

        df["ghana_us_rate_differential"] = (
            df["monetary_policy_rate"] -
            df["us_federal_fund_rate"]
        )

        df["ghana_us_rate_differential_change"] = (
            df["ghana_us_rate_differential"].diff()
        )




    return df


# ============================================================
# 6. NET INTERNATIONAL RESERVE FEATURES
# ============================================================

def create_reserve_features(df):

    df = df.copy()

    col = "net_international_reserve"

    if col not in df.columns:
        return df

    # Changes over available periods
    df["nir_change"] = df[col].diff()

    df["nir_change_30d"] = (
        df[col] - df[col].shift(30)
    )

    # df["nir_change_60d"] = (
    #     df[col] - df[col].shift(60)
    # )

    # Percentage changes
    # df["nir_pct_change_30d"] = (
    #     df[col] / df[col].shift(30) - 1
    # )
    #
    # df["nir_pct_change_60d"] = (
    #     df[col] / df[col].shift(60) - 1
    # )

    return df


# ============================================================
# 7. GOLD FEATURES
# ============================================================

def create_gold_features(df):
    df = df.copy()

    price_col = "gold_close_price"
    volume_col = "gold_volume"

    if price_col not in df.columns :
      return df

    # ========================================================
    # GOLD PRICE RETURNS
    # ========================================================
    df[ price_col ] = df[ price_col ].astype( str ).str.replace( ',' , '' , regex = False ).astype(
      float )

    df[ "gold_return_1d" ] = np.log(
      df[ price_col ] / df[ price_col ].shift( 1 )
    )

    for window in [ 5 , 10 , 20 , 30 ] :
      df[ f"gold_return_{window}d" ] = np.log(
        df[ price_col ] /
        df[ price_col ].shift( window )
      )

    # ========================================================
    # GOLD PRICE VOLATILITY
    # ========================================================

    for window in [ 5 , 10 , 20 , 30 ] :
      df[ f"gold_volatility_{window}d" ] = (
        df[ "gold_return_1d" ]
        .rolling( window )
        .std()
      )


    # =======================================================
    # GOLD VOLUME FEATURES
    # =======================================================
    df[ volume_col ] = df[ volume_col ].apply( convert_volume )

    df[ "gold_volume_ma_20d" ] = (
      df[ "gold_volume" ].rolling( 20 ).mean()
    )

    df[ "gold_relative_volume_20d" ] = np.log(
      df[ "gold_volume" ] /
      df[ "gold_volume_ma_20d" ]
    )

    df[ "gold_volume_ma_20d" ] =np.log(df[ "gold_volume_ma_20d" ])


    return df


# ============================================================
# 8. CRUDE OIL FEATURES
# ============================================================

def create_oil_features(df):
    df = df.copy()

    price_col = "crude_close_price"
    volume_col = "crude_volume"

    if price_col not in df.columns :
      return df

    # ========================================================
    # CRUDE OIL PRICE RETURNS
    # ========================================================
    df[ price_col ] = df[ price_col ].astype( str ).str.replace( ',' , '' , regex = False ).astype(
      float )

    df[ "crude_return_1d" ] = np.log(
      df[ price_col ] /
      df[ price_col ].shift( 1 )
    )

    for window in [ 5 , 10 , 20 , 30 ] :
      df[ f"crude_return_{window}d" ] = np.log(
        df[ price_col ] /
        df[ price_col ].shift( window )
      )

    # ========================================================
    # CRUDE OIL VOLATILITY
    # ========================================================

    for window in [ 5 , 10 , 20 , 30 ] :
      df[ f"crude_volatility_{window}d" ] = np.log(
        df[ "crude_return_1d" ]
        .rolling( window )
        .std()
      )


    # ======================================================
    # CRUDE OIL VOLUME FEATURES
    # ======================================================
    df[ volume_col ] = df[ volume_col ].apply( convert_volume )
    df[ "crude_volume_ma_20d" ] = (
      df[ "crude_volume" ].rolling( 20 ).mean()
    )

    df[ "crude_relative_volume_20d" ] = np.log(
      df[ "crude_volume" ] /
      df[ "crude_volume_ma_20d" ]
    )

    df[ "crude_volume_ma_20d" ] = np.log( df[ "crude_volume_ma_20d" ] )

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
  price_col = "usdgh_exchange_rate"
  start_horizon = 14
  end_horizon = 28
  shock_threshold = 0.03

  # Future horizons: +10, +11, ..., +20 trading days
  future_prices = pd.concat(
    {
      h : df[ price_col ].shift( -h )
      for h in range( start_horizon , end_horizon + 1 )
    } ,
    axis = 1
  )

  # Maximum USD/GHS rate within the future prediction window
  df[ "future_max_usdgh" ] = future_prices.max( axis = 1 )

  # Ensure the entire +10 to +20 window exists
  complete_window = future_prices.notna().all( axis = 1 )

  # Maximum future depreciation relative to today's USD/GHS
  df[ "future_max_depreciation" ] = (
                                      df[ "future_max_usdgh" ] / df[ price_col ]
                                    ) - 1

  # Binary shock target
  df[ "target_shock_window" ] = np.where(
    complete_window ,
    (df[ "future_max_depreciation" ] >= shock_threshold).astype( int ) ,
    np.nan
  )

  return df

# ============================================================
# 10. COMPLETE FEATURE ENGINEERING PIPELINE
# ============================================================

def engineer_features(df:pd.DataFrame, output_file:str|Path,logger:Logger) -> pd.DataFrame:

    df = prepare_data(df)

    # FX
    # df = create_fx_features(df)

    # Inflation
    df = create_inflation_features(df)

    # Interest rates
    df = create_interest_rate_features(df)

    # Reserves
    df = create_reserve_features(df)

    # Gold
    df = create_gold_features(df)

    # Oil
    df = create_oil_features(df)

    # Shock target
    df = create_shock_target(df)

    features_to_drop = [ 'crude_volume' , 'gold_volume' , 'monetary_policy_rate' , 'us_federal_fund_rate' , 'tbill' ,
                         'interbank_interest_rate' , 'gold_close_price' , 'crude_close_price' ,
                           'usdgh_exchange_rate' ,
                         'net_international_reserve' , 'food_inf' , 'non_food_inf' ,
                          'future_max_depreciation','future_max_usdgh' ]

    df = df.drop( columns = features_to_drop )
    df  = df.dropna()
    df = slice_data( start_date = "2010-01-01" , end_date = "2025-12-31" , data = df )
    # Check for infinite values
    inf_counts = np.isinf(
      df.select_dtypes( include = np.number )
    ).sum()

    print( "Columns containing infinity:" )
    print( inf_counts[ inf_counts > 0 ] )
    save_to_csv( output_file = output_file , data = df , logger = logger , set_index = True )

    return df