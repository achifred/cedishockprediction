"""
Run the pairwise block-bootstrap PR-AUC comparisons directly from the
project's existing test-set prediction CSVs.

Requires NO retraining -- only the files already produced by the evaluation
scripts for XGBoost, the LSTM and the ensemble.

Each comparison is run at a primary block length (DEFAULT_BLOCK_LENGTH_WEEKS)
and at shorter and longer block lengths as a sensitivity check; conclusions
should hold across them. Run from the project root:

    python -m scripts.run_block_bootstrap

Results are printed and saved to artifacts/block_bootstrap_results.json.
"""

import json
from dataclasses import asdict

import pandas as pd

from src.config.config import ENSEMBLE_PREDICTION_FILE, LSTM_TEST_PREDICTION_FILE, XGBOOST_TEST_PREDICTION_FILE
from src.config.looger_config import get_logger
from src.models.hypothesis.block_bootstrap_pr_auc_comparison import (
  DEFAULT_BLOCK_LENGTH_WEEKS,
  run_block_bootstrap_pr_auc_comparison,
)
from src.utils.utils import get_path

logger = get_logger(name=__name__, log_file="block_bootstrap.log")

SENSITIVITY_BLOCK_LENGTHS = (4, DEFAULT_BLOCK_LENGTH_WEEKS, 12)


def load_prediction_file(file_name) -> pd.DataFrame:
  return pd.read_csv(get_path(file_name), index_col="date", parse_dates=["date"])


def build_aligned_predictions() -> pd.DataFrame:
  xgboost_predictions = load_prediction_file(XGBOOST_TEST_PREDICTION_FILE)
  lstm_predictions = load_prediction_file(LSTM_TEST_PREDICTION_FILE)
  ensemble_predictions = load_prediction_file(ENSEMBLE_PREDICTION_FILE)

  # Align the three on the shared dates (inner join), keeping each file's own
  # copy of the labels so that they can be checked against one another.
  merged = (
    xgboost_predictions[["actual_shock_window", "xgb_shock_probability"]]
    .rename(columns={"actual_shock_window": "label_xgboost"})
    .join(
      lstm_predictions[["actual_shock_window", "lstm_shock_probability"]]
      .rename(columns={"actual_shock_window": "label_lstm"}),
      how="inner",
    )
    .join(
      ensemble_predictions[["actual_shock_window", "ensemble_probability"]]
      .rename(columns={"actual_shock_window": "label_ensemble"}),
      how="inner",
    )
    .sort_index()
  )

  labels_agree = (
    (merged["label_xgboost"] == merged["label_lstm"]) & (merged["label_xgboost"] == merged["label_ensemble"])
  )
  if not labels_agree.all():
    raise ValueError(
      f"target labels differ across the prediction files on {int((~labels_agree).sum())} dates "
      f"(first: {merged.index[~labels_agree][0]})"
    )

  logger.info(
    f"aligned rows for bootstrap comparison: {len(merged)} "
    f"({merged.index.min().date()} to {merged.index.max().date()}), "
    f"shock weeks: {int(merged['label_xgboost'].sum())}"
  )
  return merged


def print_result(result) -> None:
  significant = not (result.bootstrap_difference_ci_lower <= 0.0 <= result.bootstrap_difference_ci_upper)
  print(
    f"  block = {result.block_length:>2} weeks | "
    f"{result.model_a_name} {result.model_a_observed_pr_auc:.4f} vs {result.model_b_name} "
    f"{result.model_b_observed_pr_auc:.4f} | diff {result.observed_pr_auc_difference:+.4f} | "
    f"{int(result.confidence_level * 100)}% CI [{result.bootstrap_difference_ci_lower:+.4f}, "
    f"{result.bootstrap_difference_ci_upper:+.4f}] | p = {result.two_sided_p_value:.4f} | "
    f"{result.model_b_name} better in {result.proportion_of_replicates_model_b_better:.1%} of resamples | "
    f"distinguishable: {significant}"
  )


if __name__ == "__main__":

  merged = build_aligned_predictions()

  target_test = merged["label_xgboost"].values
  probabilities = {
    "XGBoost": merged["xgb_shock_probability"].values,
    "LSTM": merged["lstm_shock_probability"].values,
    "Ensemble": merged["ensemble_probability"].values,
  }

  comparisons = [
    ("XGBoost", "Ensemble"),
    ("LSTM", "Ensemble"),
    ("XGBoost", "LSTM"),
  ]

  saved_results = []
  for model_a_name, model_b_name in comparisons:
    print(f"\n{model_a_name} vs {model_b_name}")
    for block_length in SENSITIVITY_BLOCK_LENGTHS:
      result = run_block_bootstrap_pr_auc_comparison(
        target_test=target_test,
        probability_model_a=probabilities[model_a_name],
        probability_model_b=probabilities[model_b_name],
        logger=logger,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        block_length=block_length,
      )
      print_result(result)

      summary = asdict(result)
      summary.pop("bootstrap_differences")  # keep the saved file small
      saved_results.append(summary)

  output_path = get_path("artifact/summary/block_bootstrap_results.json")
  with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(saved_results, output_file, indent=2)
  logger.info(f"saved block-bootstrap results to {output_path}")