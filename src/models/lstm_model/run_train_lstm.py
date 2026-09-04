from xml.sax.handler import feature_validation

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import auc , classification_report , precision_recall_curve , roc_auc_score

from src.config.data_class import LSTMTrainingParameters
from src.config.looger_config import get_logger
from src.utils.utils import get_path
from src.models.lstm_model.train import train_lstm
from src.config.config import LSTM_MODEL_FILE , LSTM_SCALER_FILE , USDGH_DATASET_FILE
from src.models.lstm_model.lstm_helper import create_scaler, create_3d_sequence

logger = get_logger( name = __name__ , log_file = "lstm_model.log" )

def run_train_lstm(df: pd.DataFrame):

  try:
    scaler_path =  get_path(LSTM_SCALER_FILE)

    train_buffer_df = df.loc[ : '2021-12-31' ].copy()
    print( f"training shock window count: {train_buffer_df[ 'target_shock_window' ].sum()}" )
    feature_train_scaled, target_train_raw, scaler = create_scaler(data = train_buffer_df, logger = logger, output_file=scaler_path)
    # feature_cols = train_buffer_df.drop(columns = [ "target_shock_window" ])



    feature_train_3d , target_train,_ = create_3d_sequence(
      scaled_feature = feature_train_scaled,
      target = target_train_raw,
      dates = train_buffer_df.index,
      logger = logger
    )

    validation_buffer_df = df.loc[ '2022-01-01' :'2024-12-31' ].copy()
    print(f"validation shock window count: {validation_buffer_df['target_shock_window'].sum()}")
    target_validation_raw = validation_buffer_df[ 'target_shock_window' ].values
    feature_validation_raw = validation_buffer_df.drop( columns = [ 'target_shock_window' ] ).values
    feature_validation_scaled = scaler.transform(feature_validation_raw)
    feature_validation_3d, target_validation,validation_dates = create_3d_sequence(
      scaled_feature = feature_validation_scaled,
      target = target_validation_raw,
      dates = validation_buffer_df.index,
      logger = logger
    )

    evaluation_mask = (
                        validation_dates >= pd.Timestamp('2022-01-01')
                      ) & (
                        validation_dates <= pd.Timestamp('2024-12-31')
                      )

    feature_val_3d = feature_validation_3d[ evaluation_mask ]
    target_val = target_validation[ evaluation_mask ]
    # print(f"validation target count: {len(target_validation)}")


    lstm_train_param = LSTMTrainingParameters(
      feature_train_3d = feature_train_3d,
      target_train = target_train,
      feature_validation_3d = feature_val_3d,
      target_validation = target_val,
      model_output_path = get_path(LSTM_MODEL_FILE),
    )
    train_lstm(train_param = lstm_train_param,logger = logger)


  except Exception as e:
    logger.error(f"something went wrong: {e}")


if __name__ == '__main__':

  file_path = get_path( USDGH_DATASET_FILE )
  if not file_path.exists() :
    logger.error(f"file not found: {file_path}")
    exit(1)

  data = pd.read_csv( file_path , parse_dates = [ 'date' ] ).set_index( 'date' )
  run_train_lstm(df = data)


