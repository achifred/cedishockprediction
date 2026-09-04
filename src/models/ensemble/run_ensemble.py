import pandas as pd
from src.config.looger_config import get_logger
from src.utils.utils import get_path
from src.models.ensemble.ensemble import get_ensemble_model_prediction, plot_ensemble_prediction_result
from src.config.config import LSTM_PREDICTION_FILE , XGBOOST_PREDICTION_FILE


logger = get_logger(name = __name__, log_file = "ensemble.log")

if __name__ == "__main__":

  try:
    xgb_pred_file = get_path( XGBOOST_PREDICTION_FILE )
    lstm_pred_file = get_path( LSTM_PREDICTION_FILE )

    xgb_preds = pd.read_csv( xgb_pred_file , parse_dates = [ 'date' ] ).set_index( 'date' )
    lstm_preds = pd.read_csv( lstm_pred_file , parse_dates = [ 'date' ] ).set_index( 'date' )

    param = get_ensemble_model_prediction( xgb_prediction = xgb_preds , lstm_prediction = lstm_preds, logger = logger )
    plot_ensemble_prediction_result( param = param , logger = logger )

  except Exception as e:
    logger.error( f"Ensemble Prediction Error: {e}" )