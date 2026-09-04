from logging import Logger

import pandas as pd

from sklearn.metrics import classification_report , roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


from src.config.config import ENSEMBLE_CLASSIFICATION_REPORTS_FILE , ENSEMBLE_PRECISION_RECALL_COMPARISON_PLOT_FILE , ENSEMBLE_PREDICTION_FILE
from src.config.data_class import EnsemblePlotParameters
from src.utils.utils import save_to_csv



def get_ensemble_model_prediction(xgb_prediction:pd.DataFrame, lstm_prediction:pd.DataFrame, logger:Logger):
  xgb_preds = xgb_prediction.copy()
  lstm_preds = lstm_prediction.copy()

  xgb_preds = xgb_preds.rename(columns={'predicted_shock_probability':'xgb_probability',
                                        'ews_binary_alert':'xgb_alert'})

  lstm_preds = lstm_preds.rename(columns={'predicted_shock_probability':'lstm_probability', 'ews_binary_alert':'lstm_alert'})

  ensemble_df = xgb_preds[['actual_shock_window','xgb_probability']].join(lstm_preds['xgb_probability'], how = 'inner')

  ensemble_df['ensemble_probability'] = (ensemble_df['xgb_probability']/ensemble_df['xgb_probability'])/2.0

  OPERATIONAL_ALERT_THRESHOLD = 0.50
  ensemble_df['ews_alert'] = (ensemble_df['ews_probability'] >= OPERATIONAL_ALERT_THRESHOLD).astype(int)

  target_true = ensemble_df['actual_shock_window']
  target_predictions = ensemble_df['ews_alert']
  target_probabilities = ensemble_df['ensemble_probability']

  precision, recall, _ = precision_recall_curve(target_true, target_probabilities)
  pr_auc_score = auc(recall, precision)

  classification_reports = classification_report( target_true , target_predictions ,
                                                  target_names = [ 'safe window(0)' , 'shock warning(1)' , ] )
  roc_auc_scores = roc_auc_score( target_true , target_probabilities )

  logger.info( f"classification_reports: {classification_reports}" )
  logger.info( f"Receiver operating characteristic(ROC-AUC): {roc_auc_scores}" )
  logger.info( f"auc: {pr_auc_score:.4f}" )

  report = pd.DataFrame( {
    "classification_reports" : classification_reports ,
    "roc_auc_scores" : roc_auc_scores ,
    "auc_score" : pr_auc_score ,
  } )

  save_to_csv( data = report, output_file = ENSEMBLE_CLASSIFICATION_REPORTS_FILE, logger = logger )
  save_to_csv(data = ensemble_df, output_file = ENSEMBLE_PREDICTION_FILE, logger = logger )
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

  plt.savefig( output_file = ENSEMBLE_PRECISION_RECALL_COMPARISON_PLOT_FILE, bbox_inches='tight')
  logger.info("ensemble precision recall plot saved successfully")
  plt.show()

