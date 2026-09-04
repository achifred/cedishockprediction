FORWARD_FILL_START = "2008-01-01"
FORWARD_FILL_END = "2026-01-31"
START_DATE = "2009-01-01"
END_DATE = "2026-01-31"
TIME_STEPS = 30

XGBOOST_PREDICTION_FILE = "artifact/result/xgboost_prediction_result.csv"
XGBOOST_CLASSIFICATION_REPORT_FILE = "artifact/result/xgboost_classification_report.csv"
XGBOOST_FEATURE_IMPORTANCE_FILE = "artifact/result/xgboost_feature_importance.csv"
XGBOOST_SAVED_COLUMNS_FILE = "artifact/result/xgb_feature_columns_list.csv"
XGBOOST_MODEL_FILE = "artifact/model/xgb_ews_model.json"

LSTM_CLASSIFICATION_REPORTS_FILE = "artifact/result/lstm_classification_report.csv"
LSTM_PREDICTION_FILE = "artifact/result/lstm_prediction_result.csv"
LSTM_MODEL_FILE = "artifact/model/lstm_model.keras"
LSTM_SCALER_FILE = "artifact/model/lstm_scaler.pkl"

ENSEMBLE_CLASSIFICATION_REPORTS_FILE = "artifact/result/ensemble_classification_report.csv"
ENSEMBLE_PREDICTION_FILE = "artifact/result/ensemble_prediction_result.csv"
ENSEMBLE_PRECISION_RECALL_COMPARISON_PLOT_FILE =  "artifact/result/ensemble_precision_recall_comparison_plot.png"


USDGH_DATASET_FILE = "data/processed/usdgh_shock_window_dataset.csv"
COMBINED_DATASET_FILE = "data/interim/combined_data.csv"

FEATURE_NAMES = [
  'net_reserve_momentum' ,
  'gold_return' ,
  'crude_return' ,
  'food_inf_change' ,
  'non_food_inf_change' ,
  'crude_log_volume' ,
  'gold_log_volume' ,
  'tbill_mpr_spread_change' ,
  'global_rate_gap'
]