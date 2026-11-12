"""
Run the label-shuffle test of predictive skill for XGBoost and the LSTM.

Each run RETRAINS the model once on the true labels and once per shuffle on
permuted training labels, so it is slow, especially for the LSTM. Run from
the project root:

    python -m scripts.run_label_shuffle_test --model xgboost --shuffles 99
    python -m scripts.run_label_shuffle_test --model lstm --shuffles 49
    python -m scripts.run_label_shuffle_test --model both --shuffles 49

Reporting the empirical p-value, (k + 1) / (B + 1), needs B >= 19 for a
0.05 level and B >= 99 for a 0.01 level; smaller B cannot reach those levels.

Results are printed and saved as JSON under artifacts/ for use in the thesis.

NOTE ON THE RETRAINED MODELS: to keep cost manageable, the retrained models
are simplified versions of the reported ones (no early stopping or
learning-rate reduction for the LSTM; no early stopping for XGBoost). Their
observed scores therefore differ somewhat from the main results in the
thesis, and the comparison that matters is observed versus permuted-label
scores for the SAME simplified model.
"""

import argparse
import json
import random
from dataclasses import asdict

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.utils import compute_class_weight
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential


from src.config.config import (
  EARLY_STOP_PATIENCE , LSTM_BATCH_SIZE ,
  LSTM_DROPOUT_RATE ,
  LSTM_LEARNING_RATE ,
  LSTM_RELU_DENSE_DROPOUT_RATE ,
  LSTM_RELU_DENSE_UNITS ,
  LSTM_UNITS ,
  REDUCE_LEARNING_PATIENCE , VALIDATION_TIMELINE_END , VALIDATION_TIMELINE_START , XGBOOST_EARLY_STOPPING_ROUNDS ,
  XGB_RANDOM_SEED ,
  LSTM_RANDOM_SEED ,
  TIME_STEPS ,
  USDGH_DATASET_FILE ,
  USDGH_DATASET_LSTM_FILE ,
  TEST_TIMELINE_START ,
  TEST_TIMELINE_END ,
  TRAINING_TIMELINE_LIMIT ,
  XGBOOST_LEARNING_RATE ,
  XGBOOST_MAX_DEPTH ,
  XGBOOST_NUMBER_OF_ESTIMATORS ,
)
from src.config.looger_config import get_logger
from src.models.hypothesis.label_shuffle_test import run_label_shuffle_test
from src.models.lstm_model.lstm_helper import create_3d_sequence, create_scaler
from src.models.xgboost_model.helper import extract_feature_target
from src.utils.utils import get_path

logger = get_logger(name=__name__, log_file="label_shuffle_test.log")



def load_raw_dataset(file_str) -> pd.DataFrame:
  return pd.read_csv(get_path(file_str), parse_dates=["date"]).set_index("date")


# ----------------------------------------------------------------------
# XGBoost
# ----------------------------------------------------------------------
def xgboost_fit_predict(feature_train, target_train, feature_val, target_val, feature_test):
  negative_count = (target_train == 0).sum()
  positive_count = max((target_train == 1).sum(), 1)

  model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=negative_count / positive_count,
    max_depth=XGBOOST_MAX_DEPTH,
    learning_rate=XGBOOST_LEARNING_RATE,
    n_estimators=XGBOOST_NUMBER_OF_ESTIMATORS,
    early_stopping_rounds = XGBOOST_EARLY_STOPPING_ROUNDS,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=XGB_RANDOM_SEED,
    n_jobs=1,
    tree_method="hist",
    min_child_weight = 1 ,
    reg_alpha = 0.5 ,
    reg_lambda = 2 ,
  )
  model.fit(feature_train, target_train, eval_set = [(feature_val, target_val)])
  return model.predict_proba(feature_test)[:, 1]


def run_xgboost_shuffle_test(df: pd.DataFrame, number_of_shuffles: int, random_seed: int, alpha: float):
  feature_xgb, target_xgb = extract_feature_target(df=df)
  feature_validation_xgb = feature_xgb.copy()
  target_validation_xgb = target_xgb.copy()

  train_mask = feature_xgb.index <= TRAINING_TIMELINE_LIMIT
  test_mask = (feature_xgb.index >= TEST_TIMELINE_START) & (feature_xgb.index <= TEST_TIMELINE_END)

  feature_train = feature_xgb[train_mask].values
  target_train = target_xgb[train_mask].values
  feature_test = feature_xgb[test_mask].values
  target_test = target_xgb[test_mask].values

  validation_mask = (feature_validation_xgb.index >= VALIDATION_TIMELINE_START) & (
      feature_validation_xgb.index <= VALIDATION_TIMELINE_END)
  feature_val , target_val = feature_validation_xgb[ validation_mask ] , target_validation_xgb[ validation_mask ]

  logger.info(
    f"XGBoost label-shuffle test: train n={len(target_train)}, test n={len(target_test)}, "
    f"test shock weeks={int(target_test.sum())}"
  )

  return run_label_shuffle_test(
    fit_predict_fn=xgboost_fit_predict,
    feature_train=feature_train,
    target_train=target_train,
    feature_test=feature_test,
    target_test=target_test,
    feature_val=feature_val.values,
    target_val = target_val.values,
    logger=logger,
    number_of_shuffles=number_of_shuffles,
    random_seed=random_seed,
    significance_level=alpha,
  )


# ----------------------------------------------------------------------
# LSTM
# ----------------------------------------------------------------------
def build_fresh_lstm_model(time_steps: int, feature_count: int) -> Sequential:
  model = Sequential([
    Input(shape=(time_steps, feature_count)),
    LSTM(units=LSTM_UNITS, return_sequences=False),
    Dropout(0.4),
    Dense(units=LSTM_RELU_DENSE_UNITS, activation="relu"),
    Dropout(0.2),
    Dense(units=1, activation="sigmoid"),
  ])
  model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LSTM_LEARNING_RATE),
    loss="binary_crossentropy",
    metrics = [
      tf.keras.metrics.AUC( name = 'auc' ) ,
      tf.keras.metrics.AUC( curve = 'PR' , name = 'auc_pr' ) ,
      'accuracy' ,
      tf.keras.metrics.Precision() ,
      tf.keras.metrics.Recall() ,
    ]
  )
  return model


def make_lstm_fit_predict(time_steps: int, feature_count: int, epochs: int):
  def lstm_fit_predict(feature_train, target_train, feature_val, target_val, feature_test):
    # Same seed for every fit, so the only thing that varies between the
    # observed and the permuted-label runs is the training labels.
    random.seed( LSTM_RANDOM_SEED )
    np.random.seed( LSTM_RANDOM_SEED )
    tf.random.set_seed( LSTM_RANDOM_SEED )
    # tf.keras.utils.set_random_seed(LSTM_RANDOM_SEED)

    early_stop = tf.keras.callbacks.EarlyStopping(
      monitor = 'val_auc_pr' , mode = 'max' , patience = EARLY_STOP_PATIENCE , restore_best_weights = True
    )
    # shrink the learning rate before early stopping gives up entirely,
    # rather than jumping straight from "improving" to "stopped"
    reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
      monitor = 'val_auc_pr' , mode = 'max' , factor = 0.5 , patience = REDUCE_LEARNING_PATIENCE , min_lr = 1e-5
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
      get_path("artifact/model/lstm_test.keras") , monitor = 'val_auc_pr' , mode = 'max' , save_best_only = True , save_weights_only = False
    )
    print(f"time stamps: {time_steps}")
    model = build_fresh_lstm_model(time_steps=time_steps, feature_count=feature_count)

    classes = np.unique( target_train )
    weights = compute_class_weight( class_weight = 'balanced' , classes = classes , y = target_train )
    class_weight_dict = { classes[ 0 ] : weights[ 0 ] , classes[ 1 ] : weights[ 1 ] }
    model.fit(
      feature_train, target_train,
      # validation_data = (feature_val, target_val),
      epochs=50,
      batch_size=LSTM_BATCH_SIZE,
      class_weight=class_weight_dict, #{0: class_zero_weight, 1: class_one_weight},
      # callbacks = [ early_stop , reduce_learning_rate ],
      verbose=0,
    )
    return model.predict(feature_test, verbose=0).flatten()



  return lstm_fit_predict


def build_test_sequences(df: pd.DataFrame, scaler):
  """
  Build test sequences whose labelled weeks cover the WHOLE test period.

  The TIME_STEPS weeks before TEST_START_DATE are used as look-back context
  (past information only), so the first labelled test sequence falls on the
  first test week. Building the sequences from the test window alone would
  consume the first TIME_STEPS test weeks as look-back and silently drop the
  earliest test labels (including shock weeks).
  """
  window = df.loc["2023-08-31":TEST_TIMELINE_END]

  # first_test_position = int(history_until_end.index.searchsorted(pd.Timestamp("2023-11-01")))
  # lookback_start_position = max(0, first_test_position - TIME_STEPS)
  # window = history_until_end.iloc[lookback_start_position:]

  target_raw = window["target_shock_window"].values
  feature_scaled = scaler.transform(window.drop(columns=["target_shock_window"]).values)

  feature_3d, target_3d, sequence_dates = create_3d_sequence(
    scaled_feature=feature_scaled, target=target_raw, dates=window.index, logger=logger
  )

  within_test_period = (
    (sequence_dates >= pd.Timestamp(TEST_TIMELINE_START)) & (sequence_dates <= pd.Timestamp(TEST_TIMELINE_END))
  )
  return feature_3d[within_test_period], target_3d[within_test_period]


def run_lstm_shuffle_test(
  df: pd.DataFrame, number_of_shuffles: int, random_seed: int, alpha: float, epochs: int
):
  train_df = df.loc[:TRAINING_TIMELINE_LIMIT].copy()

  # Scaler fitted on the training partition only, as in the main pipeline.
  feature_train_scaled, target_train_raw, scaler = create_scaler(
    data=train_df, logger=logger, output_file=get_path("artifact/model/label_shuffle_scaler.joblib")
  )
  feature_train_3d, target_train_3d, _ = create_3d_sequence(
    scaled_feature=feature_train_scaled, target=target_train_raw, dates=train_df.index, logger=logger
  )

  feature_test_3d, target_test_3d = build_test_sequences(df, scaler)

  validation_buffer_df = df.loc[ VALIDATION_TIMELINE_START :VALIDATION_TIMELINE_END ].copy()

  target_validation_raw = validation_buffer_df[ 'target_shock_window' ].values

  feature_validation_raw = validation_buffer_df.drop( columns = [ 'target_shock_window' ] ).values
  feature_validation_scaled = scaler.transform( feature_validation_raw )
  feature_validation_3d , target_validation , validation_dates = create_3d_sequence(
    scaled_feature = feature_validation_scaled ,
    target = target_validation_raw ,
    dates = validation_buffer_df.index ,
    logger = logger
  )

  print( f"validation shock window count after sequence: {target_validation.sum()}" )

  validation_mask = (
                      validation_dates >= pd.Timestamp( VALIDATION_TIMELINE_START )
                    ) & (
                      validation_dates <= pd.Timestamp( VALIDATION_TIMELINE_END )
                    )

  feature_val_3d = feature_validation_3d[ validation_mask ]
  target_val = target_validation[ validation_mask ]

  logger.info(
    f"LSTM label-shuffle test: train sequences={len(target_train_3d)}, test sequences={len(target_test_3d)}, "
    f"test shock weeks={int(target_test_3d.sum())}"
  )

  return run_label_shuffle_test(
    fit_predict_fn=make_lstm_fit_predict(
      time_steps=12, feature_count=feature_train_3d.shape[2], epochs=epochs
    ),
    feature_train=feature_train_3d,
    target_train=target_train_3d,
    feature_test=feature_test_3d,
    target_test=target_test_3d,
    feature_val = feature_val_3d,
    target_val = target_val,
    logger=logger,
    number_of_shuffles=number_of_shuffles,
    random_seed=random_seed,
    significance_level=alpha,
  )


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def print_result(model_name: str, result) -> None:
  print(f"\n{model_name} label-shuffle test ({result.number_of_shuffles} shuffles)")
  print(f"  Observed PR-AUC:   {result.observed_pr_auc:.4f}  (no-skill = {result.no_skill_pr_auc:.4f})")
  print(f"  Observed ROC-AUC:  {result.observed_roc_auc:.4f}  (no-skill = {result.no_skill_roc_auc:.4f})")
  print(f"  Shuffled PR-AUC:   mean={result.shuffled_pr_auc_mean:.4f}  std={result.shuffled_pr_auc_std:.4f}")
  print(f"  Shuffled ROC-AUC:  mean={result.shuffled_roc_auc_mean:.4f}  std={result.shuffled_roc_auc_std:.4f}")
  print(
    f"  Shuffles matching/beating observed PR-AUC:  "
    f"{result.shuffled_replicates_matching_or_beating_observed_pr_auc}/{result.number_of_shuffles}"
    f"   empirical p = {result.pr_auc_empirical_p_value:.4f}"
  )
  print(
    f"  Shuffles matching/beating observed ROC-AUC: "
    f"{result.shuffled_replicates_matching_or_beating_observed_roc_auc}/{result.number_of_shuffles}"
    f"   empirical p = {result.roc_auc_empirical_p_value:.4f}"
  )
  print(
    f"  Smallest attainable p with {result.number_of_shuffles} shuffles: {result.minimum_attainable_p_value:.4f} "
    f"(need >= {result.minimum_shuffles_for_significance_level} shuffles for alpha = {result.significance_level})"
  )
  print(f"  PR-AUC separated from noise at alpha = {result.significance_level}:  {result.pr_auc_separated_from_noise}")
  print(f"  ROC-AUC separated from noise at alpha = {result.significance_level}: {result.roc_auc_separated_from_noise}")
  print(f"  {result.interpretation}")


def save_result(model_key: str, result) -> None:
  output_path = get_path(f"artifact/summary/label_shuffle_{model_key}.json")
  with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(asdict(result), output_file, indent=2)
  logger.info(f"saved label-shuffle result to {output_path}")


if __name__ == "__main__":

  raw_data = load_raw_dataset(USDGH_DATASET_FILE)
  raw_data_lstm = load_raw_dataset( USDGH_DATASET_LSTM_FILE )


  xgboost_result = run_xgboost_shuffle_test(raw_data, 10, 42, 0.09)
  print_result("XGBoost", xgboost_result)
  save_result("xgboost", xgboost_result)


  lstm_result = run_lstm_shuffle_test(raw_data_lstm, 10, 42, 0.1, 15)
  print_result("LSTM", lstm_result)
  save_result("lstm", lstm_result)