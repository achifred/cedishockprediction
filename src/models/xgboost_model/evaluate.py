from logging import Logger
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, classification_report, roc_auc_score, auc

from src.config.config import  XGBOOST_CLASSIFICATION_REPORT_FILE, XGBOOST_FEATURE_IMPORTANCE_FILE, XGBOOST_PREDICTION_FILE

from src.utils.utils import save_to_csv
from src.config.data_class import XGBoostEvaluationParameters
from src.models.xgboost_model.helper import extract_feature_target


def predict_xgboost(data:pd.DataFrame, save_cols:pd.DataFrame, model_path:Path, logger:Logger):
  df  = data.copy()

  logger.info("extracting feature and target data")
  feature_xgb, target_xgb = extract_feature_target( df = df )
  logger.info("extracting feature and target data done")

  # define testing timeline from data(2022-2025)
  logger.info("defining timeline for training data")
  test_mask = feature_xgb.index >= '2022-01-01'
  feature_test, target_test = feature_xgb[test_mask], target_xgb[test_mask]

  # check if columns are the same to saved columns
  saved_columns = save_cols.iloc[:, 0].tolist()
  feature_test = feature_test[saved_columns]

  logger.info("loading saved xgboost model")
  xgb_model = xgb.XGBClassifier()

  xgb_model.load_model(model_path)

  logger.info("predicting xgboost model")
  target_prediction = xgb_model.predict(feature_test)
  target_probability = xgb_model.predict_proba(feature_test)[:,1]
  logger.info("model prediction done")

  return XGBoostEvaluationParameters(

    target_prediction = target_prediction,
    target_probability = target_probability,
    target_test = target_test,
    feature_importance = xgb_model.feature_importances_,
    columns = feature_test.columns,
  )



def evaluate_xgboost(evaluation_param:XGBoostEvaluationParameters, logger:Logger):
  try:

    logger.info("beginning xgboost evaluation")

    target_test = evaluation_param.target_test
    target_prediction = evaluation_param.target_prediction
    feature_importance = evaluation_param.feature_importance
    target_probability = evaluation_param.target_probability
    columns = evaluation_param.columns

    precision, recall,_ =  precision_recall_curve(target_test, target_probability)
    pr_auc_score = auc(recall, precision)

    classification_reports = classification_report(target_test, target_prediction, target_names = ['safe window(0)',
                                                                                                   'shock warning(1)',])
    roc_auc_scores = roc_auc_score(target_test, target_probability)

    logger.info(f"classification_reports: {classification_reports}")
    logger.info(f"Receiver operating characteristic(ROC-AUC): {roc_auc_scores}")
    logger.info(f"auc: {pr_auc_score:.4f}")


    report = pd.DataFrame({
      "classification_reports": [classification_reports],
      "roc_auc_scores": [roc_auc_scores],
      "auc_score": [pr_auc_score]
    })

    save_to_csv(data = report, output_file = XGBOOST_CLASSIFICATION_REPORT_FILE, logger=logger)

    prediction_result = pd.DataFrame({
      "actual_shock_window": target_test,
      "predicted_shock_probability": target_probability,
      "ews_binary_alert": target_prediction
    }, index = target_test.index)

    save_to_csv(data= prediction_result, output_file = XGBOOST_PREDICTION_FILE,logger = logger)

    # compute feature importance
    feature_importance_report = pd.DataFrame({
      "feature": columns,
      "gain_importance": feature_importance,
    }).sort_values("gain_importance", ascending = False)

    save_to_csv(data = feature_importance_report, output_file = XGBOOST_FEATURE_IMPORTANCE_FILE,logger = logger)
    logger.info("evaluation of xgboost model done")
  except Exception as e:
    raise e






