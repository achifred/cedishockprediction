import json
import os

import pandas as pd

from src.models.ensemble.ensemble import align_predictions , apply_ensemble_weight , apply_metrics , \
  calculate_ensemble_weight
from src.config.looger_config import get_logger
from src.utils.utils import get_path, save_to_csv
from src.config.config import ENSEMBLE_METRICS_FILE , LSTM_TEST_PREDICTION_FILE , \
  LSTM_VALIDATION_PREDICTION_RESULT_FILE , \
  XGBOOST_TEST_PREDICTION_FILE , \
  XGB_VALIDATION_PREDICTION_RESULT_FILE , ENSEMBLE_PREDICTION_FILE , ENSEMBLE_WEIGHTS_FILE


logger = get_logger(name = __name__, log_file = "ensemble.log")

if __name__ == "__main__":

  try:
    xgb_validation_prediction_file = get_path( XGB_VALIDATION_PREDICTION_RESULT_FILE )
    lstm_validation_prediction_file = get_path( LSTM_VALIDATION_PREDICTION_RESULT_FILE )
    xgb_pred_file = get_path( XGBOOST_TEST_PREDICTION_FILE )
    lstm_pred_file = get_path( LSTM_TEST_PREDICTION_FILE )


    aligned_validation_pred = align_predictions(xgb_validation_prediction_file, lstm_validation_prediction_file)
    best_result, result_df = calculate_ensemble_weight( aligned_eval_pred = aligned_validation_pred )

    print(f"Best Result: {best_result}")

    with open( get_path(ENSEMBLE_WEIGHTS_FILE) , 'w' ) as weights_file :
      json.dump( {
        'xgb_weight' : best_result[ 'xgb_weight' ] ,
        'lstm_weight' : best_result[ 'lstm_weight' ] ,
        'selection_metric' : 'pr_auc' ,
        'decision_threshold' : 0.5#best_result[ 'decision_threshold' ],
      } , weights_file , indent = 4 )


    aligned_test_predictions = align_predictions( xgb_pred_file , lstm_pred_file )
    with open( get_path(ENSEMBLE_WEIGHTS_FILE) ) as weights_file :
      locked_weights = json.load( weights_file )
    ensembled_test_predictions = apply_ensemble_weight( aligned_test_predictions , locked_weights[ 'xgb_weight' ], locked_weights[ 'decision_threshold' ] )
    save_to_csv(data = ensembled_test_predictions, output_file = get_path(ENSEMBLE_PREDICTION_FILE), logger = logger )

    ensemble_dev_metrics = apply_metrics(ensembled_predictions = ensembled_test_predictions)
    logger.info(f"Ensemble Model Metrics: {ensemble_dev_metrics}")
    metrics_df = pd.DataFrame([ensemble_dev_metrics])
    save_to_csv(data = metrics_df, output_file = get_path(ENSEMBLE_METRICS_FILE), logger = logger )

    # param = get_ensemble_model_prediction( xgb_prediction = xgb_preds , lstm_prediction = lstm_preds, logger = logger )
    # plot_ensemble_prediction_result( param = param , logger = logger )

  except Exception as e:
    logger.exception( f"Ensemble Prediction Error: {e}" )