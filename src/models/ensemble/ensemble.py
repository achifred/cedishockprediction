from logging import Logger

import pandas as pd
import numpy as np

from sklearn.metrics import (average_precision_score , brier_score_loss , classification_report ,
                             f1_score , \
  precision_score , \
  recall_score , \
  roc_auc_score , \
  precision_recall_curve , \
  auc)
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.config import BEST_LSTM_THRESHOLD_FILE , BEST_XGBOOST_THRESHOLD_FILE , \
  ENSEMBLE_CLASSIFICATION_REPORTS_FILE , \
  ENSEMBLE_PRECISION_RECALL_COMPARISON_PLOT_FILE , ENSEMBLE_PREDICTION_FILE
from src.config.data_class import EnsemblePlotParameters
from src.utils.utils import get_path , save_to_csv


def load_predictions(file_path:str):
  prediction_path = get_path(file_path)
  predictions = pd.read_csv(prediction_path)
  predictions.date = pd.to_datetime(predictions.date)
  return predictions


def align_predictions(xgboost_predictions_path:str, lstm_predictions_path:str):
  xgboost_predictions = load_predictions(xgboost_predictions_path)
  lstm_predictions = load_predictions(lstm_predictions_path)

  merged_predictions = xgboost_predictions.merge(
    lstm_predictions, on='date', how='inner', suffixes=('_xgb', '_lstm')
  )

  if not np.array_equal(merged_predictions.actual_shock_window_xgb.values, merged_predictions.actual_shock_window_lstm.values):
    raise ValueError('Actual targets differ between model prediction files.')

  return pd.DataFrame({
    'date': merged_predictions.date,
    'actual_shock_window': merged_predictions.actual_shock_window_xgb,
    'xgb_shock_probability': merged_predictions.xgb_shock_probability,
    'lstm_shock_probability': merged_predictions.lstm_shock_probability,
  }).sort_values('date').reset_index(drop=True)

def compute_ensemble_probability ( predictions , xgboost_weight ) :

  return xgboost_weight * predictions.xgb_shock_probability + (1 - xgboost_weight) * predictions.lstm_shock_probability


def apply_ensemble_weight(aligned_predictions:pd.DataFrame, xgboost_weight:float, threshold:float):
  aligned_predictions = aligned_predictions.copy()
  aligned_predictions['ensemble_probability'] = compute_ensemble_probability(aligned_predictions, xgboost_weight)
  aligned_predictions['ensemble_prediction'] = (
    aligned_predictions.ensemble_probability >= threshold
  ).astype(int)
  return aligned_predictions


def apply_metrics(ensembled_predictions:pd.DataFrame):
  actual_labels = ensembled_predictions.actual_shock_window.astype( int ).values
  predicted_probabilities = ensembled_predictions.ensemble_probability.values
  predicted_labels = ensembled_predictions.ensemble_prediction.values
  return {
    'ROC_AUC' : roc_auc_score( actual_labels , predicted_probabilities ) ,
    'PR_AUC' : average_precision_score( actual_labels , predicted_probabilities ) ,
    'Precision' : precision_score( actual_labels , predicted_labels , zero_division = 0 ) ,
    'Recall' : recall_score( actual_labels , predicted_labels , zero_division = 0 ) ,
    'F1' : f1_score( actual_labels , predicted_labels , zero_division = 0 ) ,
    'Brier' : brier_score_loss( actual_labels , predicted_probabilities ) ,
  }

def calculate_ensemble_weight(aligned_eval_pred:pd.DataFrame):
  actual_labels = aligned_eval_pred.actual_shock_window.values
  best_result = None
  search_results = [ ]
  for xgb_weight in np.arange(0.0, 1.01, 0.05):

    ensembled_probabilities = compute_ensemble_probability(aligned_eval_pred, xgb_weight)

    precision , recall , thresholds = precision_recall_curve( actual_labels , ensembled_probabilities )
    f1_scores = np.divide(
      2 * (precision * recall) , (precision + recall) ,
      out = np.zeros_like( precision * recall ) , where = (precision + recall) != 0
    )
    if len( thresholds ) == 0 :
      continue

    best_index_for_weight = int( np.argmax( f1_scores[ :-1 ] ) )
    candidate = {
      'xgb_weight' : float( xgb_weight ) ,
      'lstm_weight' : float( 1 - xgb_weight ) ,
      'decision_threshold' : float( thresholds[ best_index_for_weight ] ) ,
      'f1' : float( f1_scores[ best_index_for_weight ] ) ,
      'precision' : float( precision[ best_index_for_weight ] ) ,
      'recall' : float( recall[ best_index_for_weight ] ) ,
      # PR-AUC reported alongside for reference/discussion -- this is
      # the threshold-free ranking metric the original script used
      # for weight selection; worth reporting even though F1 now
      # drives the actual selection.
      'pr_auc' : float( average_precision_score( actual_labels , ensembled_probabilities ) ) ,
    }
    search_results.append( candidate )
    if best_result is None or candidate[ 'f1' ] > best_result[ 'f1' ] :
      best_result = candidate

  return best_result , pd.DataFrame( search_results )






















def get_ensemble_model_prediction(xgb_prediction:pd.DataFrame, lstm_prediction:pd.DataFrame, logger:Logger):
  xgb_preds = xgb_prediction.copy()
  lstm_preds = lstm_prediction.copy()


  ensemble_df = xgb_preds[['actual_shock_window','xgb_shock_probability']].join(lstm_preds['lstm_shock_probability'],
                                                                          how = 'inner')

  ensemble_df['ensemble_probability'] = (ensemble_df['xgb_shock_probability'] + ensemble_df[
    'lstm_shock_probability'])/2.0


  precision , recall , thresholds = precision_recall_curve( ensemble_df[ 'actual_shock_window' ] ,
                                                            ensemble_df[ 'ensemble_probability' ] )



  best_lstm_threshold = pd.read_csv( get_path(BEST_LSTM_THRESHOLD_FILE) )["threshold"].iloc[0]
  best_xbg_threshold = pd.read_csv( get_path(BEST_XGBOOST_THRESHOLD_FILE) )["threshold"].iloc[0]
  best_threshold =  (best_lstm_threshold + best_xbg_threshold) / 2.0
  logger.info( f"📈 Mathematically optimized Ensemble threshold found at: {best_threshold:.4f}" )

  ensemble_df['ews_alert'] = (ensemble_df['ensemble_probability'] >= best_threshold).astype(int)

  target_true = ensemble_df['actual_shock_window']
  target_predictions = ensemble_df['ews_alert']
  target_probabilities = ensemble_df['ensemble_probability']

  pr_auc_score = auc(recall, precision)

  prediction_result = pd.DataFrame( {
    "actual_shock_window" : target_true ,
    "lstm_shock_probability" : target_probabilities ,
    "lstm_binary_alert" : target_predictions
  } , index = target_true.index )



  classification_reports = classification_report( target_true , target_predictions ,
                                                  target_names = [ 'safe window(0)' , 'shock warning(1)' , ] )
  roc_auc_scores = roc_auc_score( target_true , target_probabilities )

  logger.info( f"classification_reports: {classification_reports}" )
  logger.info( f"Receiver operating characteristic(ROC-AUC): {roc_auc_scores}" )
  logger.info( f"auc: {pr_auc_score:.4f}" )

  report = pd.DataFrame( {
    "classification_reports" : [classification_reports ],
    "roc_auc_scores" : [roc_auc_scores] ,
    "auc_score" : [pr_auc_score] ,
  } )

  save_to_csv( data = report, output_file = ENSEMBLE_CLASSIFICATION_REPORTS_FILE, logger = logger )
  save_to_csv(data = prediction_result, output_file = ENSEMBLE_PREDICTION_FILE, logger = logger )
  return EnsemblePlotParameters(
    precision = precision,
    recall = recall,
    pr_auc_score = pr_auc_score,
  )




def plot_ensemble_prediction_result(param:EnsemblePlotParameters, logger:Logger):
  sns.set_theme(style="whitegrid")
  plt.figure(figsize=(10.0, 8.0), dpi=200)

  recall =  param.recall
  precision = param.precision
  pr_auc_score = param.pr_auc_score

  # plot precision-recall curves for ensemble model

  plt.plot(recall,precision, color='#8e44ad', linewidth=2.5, label=f'Ensemble Stacking (PR-AUC = {pr_auc_score:.2f})')

  plt.xlabel('Recall/ Sensitivity (percentage of shocks caught)', fontsize=12, fontweight='bold')
  plt.ylabel('Precision (Accuracy of issued warning alerts)', fontsize=12, fontweight='bold')

  plt.title("Early Warning System Profile\n Precision-Recall for GHS/USD shock forecasting", fontsize=14,
            fontweight='bold',pad=12)

  plt.legend(loc='lower right', frameon=True)
  plt.tight_layout()

  plt.savefig( fname = get_path( ENSEMBLE_PRECISION_RECALL_COMPARISON_PLOT_FILE ), bbox_inches='tight')
  logger.info("ensemble precision recall plot saved successfully")
  plt.show()

