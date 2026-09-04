import pandas as pd
from src.utils.utils import get_path
from src.config.looger_config import get_logger
from src.models.xgboost_model.evaluate import predict_xgboost, evaluate_xgboost
from src.config.config import USDGH_DATASET_FILE, XGBOOST_MODEL_FILE, XGBOOST_SAVED_COLUMNS_FILE

logger = get_logger( name = __name__ , log_file = "xgboost_model.log" )

def run_evaluate_xgboost(df: pd.DataFrame):
  try:

    saved_col_path = get_path(XGBOOST_SAVED_COLUMNS_FILE)
    saved_columns = pd.read_csv(saved_col_path)
    model_path = get_path(XGBOOST_MODEL_FILE)

    param = predict_xgboost(data = df,save_cols = saved_columns,model_path = model_path, logger = logger)

    evaluate_xgboost(
      evaluation_param = param,
      logger = logger
    )

  except Exception as e:
    logger.error(f"something went wrong: {e}")

if __name__ == "__main__":

  try:
    source_data_path = get_path(USDGH_DATASET_FILE)
    data = pd.read_csv(source_data_path, parse_dates = ['date']).set_index('date')
    run_evaluate_xgboost(df = data)

  except Exception as e:
    logger.error(f"something went wrong: {e}")