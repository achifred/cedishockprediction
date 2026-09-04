from logging import Logger
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import  RobustScaler
import joblib

from src.config.config import FEATURE_NAMES, TIME_STEPS


def create_3d_sequence(scaled_feature, target, dates, logger:Logger):
  try:
    logger.info("Creating 3d sequence")

    assert len(scaled_feature) == len(target) == len(dates),(
            f"length mismatch: feature:{len(scaled_feature)}  target:{len(target)} sequence dates: {len(dates)}"
            )

    assert len(scaled_feature) > TIME_STEPS,(
      f"not enough rows {len(scaled_feature)} to build a single sequence length of {TIME_STEPS} "
    )

    feature_3d = [ ]
    target_3d = [ ]
    sequence_dates = [ ]

    for i in range( len( scaled_feature ) - TIME_STEPS ) :
      feature_3d.append(
        scaled_feature[ i :i + TIME_STEPS ]
      )

      target_3d.append(
        target[ i + TIME_STEPS ]
      )

      sequence_dates.append(
        dates[ i + TIME_STEPS ]
      )
    logger.info( "Creating 3d sequence done" )
    return (
      np.array( feature_3d ) ,
      np.array( target_3d ) ,
      np.array( sequence_dates )
    )


  except Exception as e:
    raise e



def create_scaler(data:pd.DataFrame, output_file:Path, logger:Logger):
  try:
    df = data.copy()
    logger.info("Creating scaler")

    target_train_raw =  df['target_shock_window'].values

    feature_train_raw = df.drop(columns = ['target_shock_window']).values

    scaler  = RobustScaler()
    feature_train_scaled = scaler.fit_transform(feature_train_raw)
    joblib.dump(scaler, output_file)
    logger.info("Creating scaler done")
    return feature_train_scaled, target_train_raw, scaler
  except Exception as e:
    raise e

def permutation_importance_lstm ( model , X , y , n_repeats = 10 , metric = roc_auc_score ) :
  """
  X: shape (n_samples, timesteps, n_features)
  y: shape (n_samples,) - true labels
  """

  baseline_preds = model.predict( X , verbose = 0 ).ravel()
  baseline_score = metric( y , baseline_preds )

  importances = { }
  for f_idx , fname in enumerate( FEATURE_NAMES ) :
    drops = [ ]
    for _ in range( n_repeats ) :
      X_permuted = X.copy()
      # Shuffle this feature across the sample axis, preserving timestep structure
      # per-sample shuffle (shuffles which sequence gets which feature values)
      perm_idx = np.random.permutation( X.shape[ 0 ] )
      X_permuted[ : , : , f_idx ] = X[ perm_idx , : , f_idx ]

      permuted_preds = model.predict( X_permuted , verbose = 0 ).ravel()
      permuted_score = metric( y , permuted_preds )
      drops.append( baseline_score - permuted_score )

    importances[ fname ] = {
      'mean_drop' : np.mean( drops ) ,
      'std_drop' : np.std( drops )
    }

  return baseline_score , importances