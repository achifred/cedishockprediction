import pandas as pd
from src.config.looger_config import get_logger
from src.utils.utils import get_path
from src.models.lstm_model.evaluate import evaluate_lstm_model, predict_lstm_model
from src.config.config import LSTM_MODEL_FILE , LSTM_SCALER_FILE , USDGH_DATASET_FILE , USDGH_DATASET_LSTM_FILE

logger = get_logger (name = __name__, log_file = "lstm_evaluation.log ")

def run_evaluate_lstm(df: pd.DataFrame ):

  try:

    model_path = get_path( LSTM_MODEL_FILE )
    scaler_path = get_path( LSTM_SCALER_FILE )

    result = predict_lstm_model(
      data = df,
      model_path = model_path,
      scaler_path = scaler_path,
      logger=logger
    )

    evaluate_lstm_model(
      evaluation_param = result,
      logger=logger
    )

  except Exception as e:
    logger.exception( f"something went wrong: {e}" )


if __name__ == "__main__":
  data_path = get_path( USDGH_DATASET_LSTM_FILE )
  data = pd.read_csv( data_path , parse_dates = [ 'date' ] ).set_index( 'date' )
  run_evaluate_lstm(df = data)
