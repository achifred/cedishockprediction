from logging import Logger

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier

from src.data.feature_importance import generate_xgb_feature_importance
from src.config.config import BEST_XGBOOST_THRESHOLD_FILE , TRAINING_TIMELINE_LIMIT , VALIDATION_TIMELINE_END , \
  VALIDATION_TIMELINE_START , \
  XGBOOST_SAVED_COLUMNS_FILE , \
  XGB_RANDOM_SEED , \
  XGBOOST_MAX_DEPTH , XGBOOST_NUMBER_OF_ESTIMATORS , XGBOOST_EARLY_STOPPING_ROUNDS , XGBOOST_LEARNING_RATE , \
  XGB_VALIDATION_PREDICTION_RESULT_FILE
from src.utils.utils import get_path
from src.utils.utils import save_to_csv,get_best_threshold
from src.models.xgboost_model.helper import extract_feature_target


def train_xgboost_model(data:pd.DataFrame, logger:Logger):
  try:


        df  = data.copy()
        feature_xgb,target_xgb = extract_feature_target( df=df )

        # save processed feature columns for consistency
        logger.info(f"saving feature columns to {XGBOOST_SAVED_COLUMNS_FILE}")
        processed_columns = pd.Series(feature_xgb.columns)
        save_to_csv(data = processed_columns, output_file = XGBOOST_SAVED_COLUMNS_FILE,logger = logger)
        logger.info("feature columns saved")

        feature_train_xgb = feature_xgb.copy()
        target_train_xgb = target_xgb.copy()
        feature_validation_xgb = feature_xgb.copy()
        target_validation_xgb = target_xgb.copy()


        # define historical training timeline from 2010 to 2021
        logger.info("defining timeline for training data")
        train_mask = feature_train_xgb.index <= TRAINING_TIMELINE_LIMIT
        feature_train, target_train = feature_train_xgb[train_mask], target_train_xgb[train_mask]

        # defining validation timeline
        validation_mask = (feature_validation_xgb.index >= VALIDATION_TIMELINE_START) & (feature_validation_xgb.index <= VALIDATION_TIMELINE_END)
        feature_val, target_val = feature_validation_xgb[validation_mask], target_validation_xgb[validation_mask]

        # cater for class imbalance
        logger.info(f"calculating weight to handle class imbalance")
        num_negatives = np.sum(target_train == 0)
        num_positives = np.sum(target_train == 1)
        calculated_weight = num_negatives / num_positives

        logger.info(f"Training dataset balance: {num_negatives} safe days((0)no shocks) | {num_positives} shocks ((1)days with shocks)")
        logger.info(f"Calculated weight: {calculated_weight}")

        # model configuration
        logger.info("configuring xgboost model")
        model_xgb = xgb.XGBClassifier(
          objective = 'binary:logistic',
          eval_metric = 'aucpr',
          scale_pos_weight = calculated_weight,
          max_depth = XGBOOST_MAX_DEPTH,
          learning_rate = XGBOOST_LEARNING_RATE,
          n_estimators = XGBOOST_NUMBER_OF_ESTIMATORS,
          subsample = 0.9,
          colsample_bytree = 0.9,
          random_state = XGB_RANDOM_SEED,
          n_jobs =  1,
          tree_method = 'hist',
          min_child_weight =2,
          reg_alpha = 0.5 ,
          reg_lambda = 2 ,
          early_stopping_rounds = XGBOOST_EARLY_STOPPING_ROUNDS
        )
        logger.info(f"configuring xgboost model done")

        logger.info( f"Training xgboost model" )
        model_xgb.fit(
          feature_train, target_train,
          eval_set = [(feature_val, target_val)],
          verbose = True
          )

        logger.info( f"Training xgboost model done" )
        val_probability = model_xgb.predict_proba( feature_val )[ : , 1 ]
        val_prediction_result = pd.DataFrame({
          "actual_shock_window": target_val,
          "xgb_shock_probability": val_probability
        }, index =  pd.DatetimeIndex(feature_val.index, name = 'date'))

        threshold = get_best_threshold(target_val , val_probability)
        threshold_df = pd.DataFrame( { "threshold" : threshold } , index = [ 0 ] )
        logger.info( f"📈 Dynamically optimized XGBoost breakout threshold found at validation: {threshold:.4f}" )
        save_to_csv( data = threshold_df , output_file = BEST_XGBOOST_THRESHOLD_FILE , logger = logger )
        save_to_csv(data = val_prediction_result, output_file = XGB_VALIDATION_PREDICTION_RESULT_FILE,
                    logger = logger, set_index = True)

        generate_xgb_feature_importance( xgb_model = model_xgb , feature_names=processed_columns   )
        return model_xgb

  except Exception as e:
        raise e


def save_xgboost_model( output_file:str, model:XGBClassifier, logger:Logger ):
  try:
        file_path = get_path(output_file)
        model.save_model(file_path)
        logger.info(f"Saved xgboost model to {file_path}")
  except Exception as e:
    raise e


