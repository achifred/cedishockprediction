from logging import Logger

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier

from src.config.config import XGBOOST_SAVED_COLUMNS_FILE
from src.utils.utils import get_path
from src.utils.utils import save_to_csv
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


        # define historical training timeline from 2010 to 2021
        logger.info("defining timeline for training data")
        train_mask = feature_xgb.index <= '2021-12-31'
        feature_train, target_train = feature_xgb[train_mask], target_xgb[train_mask]

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
          eval_metric = 'logloss',
          scale_pos_weight = calculated_weight,
          max_depth = 5,
          learning_rate = 0.01,
          n_estimators = 300,
          subsample = 0.8,
          colsample_bytree = 0.8,
          random_state = 42,
          n_jobs =  1,
          tree_method = 'hist'
        )
        logger.info(f"configuring xgboost model done")

        logger.info( f"Training xgboost model" )
        model_xgb.fit(feature_train, target_train)
        logger.info( f"Training xgboost model done" )
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


