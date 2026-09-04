from logging import Logger
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import auc , classification_report , precision_recall_curve , roc_auc_score

from src.config.config import LSTM_CLASSIFICATION_REPORTS_FILE , LSTM_PREDICTION_FILE
from src.config.data_class import LstmEvaluationParameters
from src.utils.utils import save_to_csv
from src.models.lstm_model.lstm_helper import create_3d_sequence, permutation_importance_lstm




def predict_lstm_model(data:pd.DataFrame, model_path:Path, scaler_path:Path, logger:Logger):
  try:
    logger.info( "Predicting LSTM model" )
    df = data.copy()

    # define timeline for test data
    logger.info("defining test data timeline")
    test_df_buffer =  df.loc['2024-11-15':'2025-12-31']
    print( f"test shock window count: {test_df_buffer[ 'target_shock_window' ].sum()}" )
    target_test_raw = test_df_buffer['target_shock_window'].values
    feature_test_raw = test_df_buffer.drop(columns=['target_shock_window']).values

    logger.info("defining test data timeline done")


    logger.info("loading saved scaler")
    scaler = joblib.load(scaler_path)
    feature_test_scaled = scaler.transform(feature_test_raw)

    feature_test_3d, target_test, test_dates = create_3d_sequence(
      scaled_feature = feature_test_scaled,
      target = target_test_raw,
      dates =  test_df_buffer.index,
      logger = logger
    )

    # test_dates_aligned = test_df_buffer.index[30:]
    evaluation_mask = test_dates >= pd.Timestamp('2025-01-01')
    feature_test_final = feature_test_3d[evaluation_mask]
    target_test_final = target_test[evaluation_mask]
    dates_final = test_dates[evaluation_mask]


    logger.info("loading saved LSTM model")
    loaded_lstm_model = tf.keras.models.load_model(model_path)
    logger.info( "loading saved LSTM model done" )


    logger.info("predicting LSTM model")
    target_probabilities = loaded_lstm_model.predict(feature_test_final).flatten()
    target_predictions = (target_probabilities >= 0.50).astype(int)


    logger.info("predicting LSTM model done")

    return  LstmEvaluationParameters(
      target_test_final = target_test_final,
      target_probabilities = target_probabilities,
      target_predictions = target_predictions,
      dates_final = dates_final
    )

  except Exception as e:
    raise e


def evaluate_lstm_model(evaluation_param:LstmEvaluationParameters, logger:Logger):
  try:
    logger.info("evaluating LSTM model")
    target_test_final = evaluation_param.target_test_final
    target_probabilities = evaluation_param.target_probabilities
    target_predictions = evaluation_param.target_predictions
    dates_final = evaluation_param.dates_final

    logger.info("generating model performance reports ")
    precision , recall , _ = precision_recall_curve( target_test_final , target_probabilities )
    pr_auc_score = auc( recall , precision )

    classification_reports = classification_report( target_test_final , target_predictions ,
                                                    target_names = [ 'safe window(0)' ,
                                                                     'shock warning(1)' , ] )
    roc_auc_scores = roc_auc_score( target_test_final , target_probabilities )
    logger.info(f"roc_auc_scores: {roc_auc_scores}")

    lstm_reports = pd.DataFrame( {
      "classification_reports" : [classification_reports] ,
      "roc_auc_scores" : [roc_auc_scores] ,
      "auc_score" : [pr_auc_score] ,
    } )

    logger.info( f"classification_reports: {classification_reports}" )
    logger.info( f"Receiver operating characteristic(ROC-AUC): {roc_auc_scores}" )
    logger.info( f"auc: {pr_auc_score:.4f}" )
    save_to_csv( data = lstm_reports , output_file = LSTM_CLASSIFICATION_REPORTS_FILE, logger = logger )

    logger.info("saving LSTM model prediction results ")
    prediction_result = pd.DataFrame( {
      "actual_shock_window" : target_test_final ,
      "predicted_shock_probability" : target_probabilities ,
      "ews_binary_alert" : target_predictions
    }, index = dates_final)

    save_to_csv( data = prediction_result , output_file = LSTM_PREDICTION_FILE ,
                 logger = logger )
    logger.info("saving LSTM model prediction results done")

  except Exception as e:
    raise e
