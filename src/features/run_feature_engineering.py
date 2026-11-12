import numpy as np
import pandas as pd


from src.config.looger_config import get_logger
from src.config.config import COMBINED_DATASET_FILE , USDGH_DATASET_FILE , USDGH_DATASET_LSTM_FILE
from src.features.feat import pipeline_feature_engineering
from src.features.feature_engineering_new import engineer_features
from src.features.feature_engineering_weekly import engineer_weekly_features
from src.utils.utils import get_correlation_matrix , get_path

logger = get_logger(name=__name__, log_file="feature_engineering.log")
if __name__ == "__main__":

  file_path  = get_path(COMBINED_DATASET_FILE)
  data = pd.read_csv(file_path)
  # df = feature_engineer_data(file_path = COMBINED_DATASET_FILE, output_file = USDGH_DATASET_FILE)
  df = engineer_features(df=data, output_file=USDGH_DATASET_FILE, logger=logger)
  df1 = engineer_weekly_features( df = data , output_file = USDGH_DATASET_LSTM_FILE , logger = logger )

  get_correlation_matrix(df = df)
  get_correlation_matrix(df = df1)