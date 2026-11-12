
FORWARD_FILL_START = "2008-01-01"
FORWARD_FILL_END = "2026-01-31"
START_DATE = "2009-01-01"
END_DATE = "2026-01-31"
FINAL_DATASET_START_DATE = "2010-01-01"
FINAL_DATASET_END_DATE = "2025-12-31"

TRAINING_TIMELINE_START = "2010-01-01"
TRAINING_TIMELINE_LIMIT = "2019-12-31"
VALIDATION_TIMELINE_START = "2020-01-01"
VALIDATION_TIMELINE_END = "2023-12-31"
TEST_TIMELINE_START = "2024-01-01"
TEST_TIMELINE_END = "2025-12-31"

XGB_RANDOM_SEED =42 # DEFAULT = 42
XGBOOST_MAX_DEPTH = 3 #DEFAULT = 3
XGBOOST_NUMBER_OF_ESTIMATORS = 50 #DEFAULT = 50
XGBOOST_EARLY_STOPPING_ROUNDS = 5 #DEFAULT = 5
XGBOOST_LEARNING_RATE = 0.01 #DEFAULT = 0.02
MIN_LAG_RANGE = 1 #DEFAULT = 1
MAX_LAG_RANGE = 9#DEFAULT = 9


TIME_STEPS =8 # 8
LSTM_RANDOM_SEED = 42 # DEFAULT = 42
LSTM_UNITS = 12 #DEFAULT = 12
LSTM_DROPOUT_RATE = 0.1 #DEFAULT = 0.4
LSTM_RELU_DENSE_UNITS = 4 #DEFAULT = 4
LSTM_RELU_DENSE_DROPOUT_RATE = 0.1 #DEFAULT = 0.2
LSTM_LEARNING_RATE = 0.001 #DEFAULT = 0.001 0.0008, 0.0005
LSTM_EPOCHS = 50 #DEFAULT = 50
LSTM_BATCH_SIZE = 16 # DEFAULT = 16
EARLY_STOP_PATIENCE = 5 # DEFAULT = 10
REDUCE_LEARNING_PATIENCE = 5 # DEFAULT = 4


XGBOOST_TEST_PREDICTION_FILE = "artifact/result/xgboost_prediction_result.csv"
XGBOOST_CLASSIFICATION_REPORT_FILE = "artifact/result/xgboost_classification_report.csv"
XGBOOST_FEATURE_IMPORTANCE_FILE = "artifact/result/xgboost_feature_importance.csv"
XGBOOST_SAVED_COLUMNS_FILE = "artifact/result/xgb_feature_columns_list.csv"
XGBOOST_MODEL_FILE = "artifact/model/xgb_ews_model.json"
XGB_VALIDATION_PREDICTION_RESULT_FILE = "artifact/result/xgb_val_prediction_result.csv"
XGB_FEATURE_IMPORTANCE_FILE = "artifact/result/xgb_feature_importance.csv"
XGB_FEATURE_IMPORTANCE_PLOT_FILE = "artifact/plots/xgb_feature_importance.png"

LSTM_CLASSIFICATION_REPORTS_FILE = "artifact/result/lstm_classification_report.csv"
LSTM_TEST_PREDICTION_FILE = "artifact/result/lstm_prediction_result.csv"
LSTM_MODEL_FILE = "artifact/model/lstm_model.keras"
LSTM_SCALER_FILE = "artifact/model/lstm_scaler.pkl"
LSTM_VALIDATION_PREDICTION_RESULT_FILE = "artifact/result/lstm_val_prediction_result.csv"
LSTM_FEATURE_IMPORTANCE_FILE = "artifact/result/lstm_feature_importance.csv"
LSTM_FEATURE_IMPORTANCE_PLOT_FILE = "artifact/plots/lstm_feature_importance.png"

ENSEMBLE_CLASSIFICATION_REPORTS_FILE = "artifact/result/ensemble_classification_report.csv"
ENSEMBLE_PREDICTION_FILE = "artifact/result/ensemble_prediction_result.csv"
ENSEMBLE_PRECISION_RECALL_COMPARISON_PLOT_FILE =  "artifact/result/ensemble_precision_recall_comparison_plot.png"
ENSEMBLE_WEIGHTS_FILE = "artifact/result/ensemble_weights.json"
ENSEMBLE_METRICS_FILE = "artifact/result/ensemble_metrics.csv"


USDGH_DATASET_FILE = "data/processed/usdgh_shock_window_dataset.csv"
USDGH_DATASET_LSTM_FILE = "data/processed/usdgh_shock_window_dataset_lstm.csv"
COMBINED_DATASET_FILE = "data/interim/combined_data.csv"

BEST_LSTM_THRESHOLD_FILE = "artifact/result/best_lstm_threshold.csv"
BEST_XGBOOST_THRESHOLD_FILE = "artifact/result/best_xgboost_threshold.csv"

RAW_DEBT_TO_GDP_DATA_FILE = "data/raw/debt_to_gdp_data.csv"

DATA_SUMMARY_OUTPUT_DIR = "artifact/summary"
PRECISION_RECALL_CURVE_PLOT_FILE = "artifact/plots/precision_recall_curves.png"
CONFUSION_MATRIX_PLOT_FILE = "artifact/plots"
PREDICTED_PROBABILITY_DISTRIBUTION_PLOT_FILE = "artifact/plots/predicted_probability_distribution.png"
EXCHANGE_RATE_PLOT_FILE = "artifact/plots/usdgh_exchange_rate_evolution_plot.png"
VISUALIZE_DATA_SPLITS_PLOT_FILE = "artifact/plots/data_splits_and_sequence.png"

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