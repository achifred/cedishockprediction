import pandas as pd

from src.config.config import MAX_LAG_RANGE , MIN_LAG_RANGE


def extract_feature_target( df:pd.DataFrame ):
  target_raw = df[ "target_shock_window" ]
  feature_raw = df.drop( columns = [ "target_shock_window" ] )

  lagged_columns = []
  for col in feature_raw.columns :
    for lag in range(MIN_LAG_RANGE, MAX_LAG_RANGE ):
      lag_series = feature_raw[ col ].shift( lag ).rename( f"{col}_lag_{lag}" )
      lagged_columns.append( lag_series )

  feature_lagged = pd.concat( lagged_columns , axis = 1 )
  feature_xgb = feature_lagged.dropna()
  target_xgb = target_raw.loc[ feature_xgb.index ]

  return feature_xgb, target_xgb
