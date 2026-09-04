import pandas as pd

def extract_feature_target( df:pd.DataFrame ):
  target_raw = df[ "target_shock_window" ]
  feature_raw = df.drop( columns = [ "target_shock_window" ] )

  feature_lagged = feature_raw.copy()
  for col in feature_raw.columns :
    feature_lagged[ f'{col}_lag1' ] = feature_raw[ col ].shift( 1 )
    feature_lagged[ f'{col}_lag2' ] = feature_raw[ col ].shift( 2 )
    feature_lagged[ f'{col}_lag3' ] = feature_raw[ col ].shift( 3 )
    feature_lagged[ f'{col}_lag4' ] = feature_raw[ col ].shift( 4 )
    feature_lagged[ f'{col}_lag5' ] = feature_raw[ col ].shift( 5 )

  feature_xgb = feature_lagged.dropna()
  target_xgb = target_raw.loc[ feature_xgb.index ]

  return feature_xgb, target_xgb
