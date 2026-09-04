import pandas as pd
from src.config.looger_config import get_logger
from src.utils.utils import get_path
from src.models.xgboost_model.train import train_xgboost_model, save_xgboost_model
from src.config.config import USDGH_DATASET_FILE , XGBOOST_MODEL_FILE

logger = get_logger( name = __name__ , log_file = "xgboost__train_model.log" )

def run_train_xgboost(df:pd.DataFrame):
  try :

    model = train_xgboost_model( data = df , logger = logger )
    save_xgboost_model( output_file = XGBOOST_MODEL_FILE , model = model , logger = logger )

  except Exception as e :
    logger.error( f"something went wrong : {e}" )

if __name__ == "__main__":

  try:
    logger.info(f"loading data from {USDGH_DATASET_FILE}")
    file_path = get_path( USDGH_DATASET_FILE )

    if not file_path.exists() :
      logger.info(f"loading data from {USDGH_DATASET_FILE} done")
      exit(1)

    data = pd.read_csv(file_path, parse_dates=['date']).set_index('date')

    run_train_xgboost(df = data)

  except Exception as e:
    logger.error(f"something went wrong : {e}")
