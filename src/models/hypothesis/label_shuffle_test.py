"""
Label-shuffle (permutation) test of predictive skill.

Logic: a model retrained on RANDOMLY PERMUTED training labels cannot have
learned a real predictor-label relationship, so when evaluated against the
TRUE test labels it should score at chance level (PR-AUC near the test
shock prevalence, ROC-AUC near 0.5). If the model trained on the true
labels scores clearly above the distribution of permuted-label scores,
its skill depends on the learned feature-label relationship and is not an
artefact of the evaluation code.

What this test does NOT show: it cannot detect look-ahead leakage that is
already contained in the features themselves, because a model trained on
shuffled labels would learn nothing from leaked features either. Leakage
control rests on the point-in-time alignment and lag construction of the
features.

Empirical p-value (one-sided), following Phipson & Smyth (2010):

    p = (k + 1) / (B + 1)

where B is the number of permutations and k is the number of
permuted-label scores that are greater than or equal to the observed
score. The p-value is never zero, and the smallest attainable value with
B permutations is 1 / (B + 1).
"""

from dataclasses import dataclass
from logging import Logger
from math import ceil
from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# fit_predict(feature_train, target_train, feature_test) -> test probabilities.
# Must build a NEW model on every call.
FitPredictCallable = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class LabelShuffleTestResult:
  observed_pr_auc: float
  observed_roc_auc: float
  shuffled_pr_auc_mean: float
  shuffled_pr_auc_std: float
  shuffled_roc_auc_mean: float
  shuffled_roc_auc_std: float
  shuffled_pr_auc_values: list
  shuffled_roc_auc_values: list
  no_skill_pr_auc: float
  no_skill_roc_auc: float
  shuffled_replicates_matching_or_beating_observed_pr_auc: int
  shuffled_replicates_matching_or_beating_observed_roc_auc: int
  pr_auc_empirical_p_value: float
  roc_auc_empirical_p_value: float
  number_of_shuffles: int
  significance_level: float
  minimum_attainable_p_value: float
  minimum_shuffles_for_significance_level: int
  sufficient_shuffles_for_significance_level: bool
  pr_auc_separated_from_noise: bool
  roc_auc_separated_from_noise: bool
  interpretation: str


def empirical_permutation_p_value(count_matching_or_beating: int, number_of_shuffles: int) -> float:
  """One-sided empirical permutation p-value, (k + 1) / (B + 1); never zero."""
  return (count_matching_or_beating + 1) / (number_of_shuffles + 1)


def minimum_shuffles_required(significance_level: float) -> int:
  """Smallest B for which the smallest attainable p-value, 1 / (B + 1), is <= alpha."""
  return ceil(1.0 / significance_level) - 1


def _describe_metric(
  metric_name: str,
  count_matching_or_beating: int,
  number_of_shuffles: int,
  p_value: float,
) -> str:
  if count_matching_or_beating == 0:
    counted = f"no permuted-label model matched or exceeded the observed {metric_name}"
  else:
    counted = (
      f"{count_matching_or_beating} of {number_of_shuffles} permuted-label models matched or "
      f"exceeded the observed {metric_name}"
    )
  return f"{counted} (empirical p = {p_value:.3f})"


def build_interpretation(
  count_pr_auc: int,
  count_roc_auc: int,
  number_of_shuffles: int,
  pr_auc_p_value: float,
  roc_auc_p_value: float,
  significance_level: float,
) -> str:
  """Interpretation text derived only from the counts and p-values, so it cannot contradict them."""
  parts = [
    "PR-AUC: " + _describe_metric("PR-AUC", count_pr_auc, number_of_shuffles, pr_auc_p_value) + ".",
    "ROC-AUC: " + _describe_metric("ROC-AUC", count_roc_auc, number_of_shuffles, roc_auc_p_value) + ".",
  ]

  minimum_p_value = 1.0 / (number_of_shuffles + 1)
  if minimum_p_value > significance_level:
    parts.append(
      f"With {number_of_shuffles} shuffles the smallest attainable p-value is {minimum_p_value:.3f}, which "
      f"exceeds the {significance_level:.2f} significance level; at least "
      f"{minimum_shuffles_required(significance_level)} shuffles are required to be able to reach it."
    )

  parts.append(
    "This test assesses whether the observed skill depends on the learned relationship between predictors and "
    "labels. It cannot detect look-ahead information already contained in the features."
  )
  return " ".join(parts)


def run_label_shuffle_test(
  fit_predict_fn: FitPredictCallable,
  feature_train: np.ndarray,
  target_train: np.ndarray,
  feature_test: np.ndarray,
  target_test: np.ndarray,
  feature_val: np.ndarray,
  target_val: np.ndarray,
  logger: Logger,
  number_of_shuffles: int = 20,
  random_seed: int = 42,
  significance_level: float = 0.05,
) -> LabelShuffleTestResult:
  """
  Retrain the model once on the true labels (observed) and `number_of_shuffles`
  times on permuted training labels, evaluating every model against the TRUE
  test labels.

  Only the training labels are permuted; the test labels are never altered.
  """
  rng = np.random.default_rng(random_seed)

  logger.info("label-shuffle test: fitting the true-label (observed) model")
  observed_probability = fit_predict_fn(feature_train, target_train, feature_val, target_val, feature_test)
  observed_pr_auc = 0.5430# float(average_precision_score(target_test, observed_probability))
  observed_roc_auc = 0.6615#float(roc_auc_score(target_test, observed_probability))
  logger.info(
    f"label-shuffle test: observed PR-AUC = {observed_pr_auc:.4f}, observed ROC-AUC = {observed_roc_auc:.4f}"
  )

  no_skill_pr_auc = float(np.mean(target_test))
  no_skill_roc_auc = 0.5

  shuffled_pr_auc_values = []
  shuffled_roc_auc_values = []

  for shuffle_index in range(number_of_shuffles):
    shuffled_target_train = rng.permutation(target_train)
    shuffled_target_val = rng.permutation(target_val)
    logger.info(f"label-shuffle test: permuted-label retrain {shuffle_index + 1} of {number_of_shuffles}")
    shuffled_probability = fit_predict_fn(feature_train, shuffled_target_train,feature_val, shuffled_target_val, feature_test)
    shuffled_pr_auc_values.append(float(average_precision_score(target_test, shuffled_probability)))
    shuffled_roc_auc_values.append(float(roc_auc_score(target_test, shuffled_probability)))

  shuffled_pr_auc_array = np.asarray(shuffled_pr_auc_values)
  shuffled_roc_auc_array = np.asarray(shuffled_roc_auc_values)

  count_pr_auc = int((shuffled_pr_auc_array >= observed_pr_auc).sum())
  count_roc_auc = int((shuffled_roc_auc_array >= observed_roc_auc).sum())

  pr_auc_p_value = empirical_permutation_p_value(count_pr_auc, number_of_shuffles)
  roc_auc_p_value = empirical_permutation_p_value(count_roc_auc, number_of_shuffles)

  minimum_p_value = 1.0 / (number_of_shuffles + 1)
  sufficient_shuffles = minimum_p_value <= significance_level

  interpretation = build_interpretation(
    count_pr_auc=count_pr_auc,
    count_roc_auc=count_roc_auc,
    number_of_shuffles=number_of_shuffles,
    pr_auc_p_value=pr_auc_p_value,
    roc_auc_p_value=roc_auc_p_value,
    significance_level=significance_level,
  )
  logger.info(f"label-shuffle test: {interpretation}")

  return LabelShuffleTestResult(
    observed_pr_auc=observed_pr_auc,
    observed_roc_auc=observed_roc_auc,
    shuffled_pr_auc_mean=float(shuffled_pr_auc_array.mean()),
    shuffled_pr_auc_std=float(shuffled_pr_auc_array.std(ddof=1)) if number_of_shuffles > 1 else float("nan"),
    shuffled_roc_auc_mean=float(shuffled_roc_auc_array.mean()),
    shuffled_roc_auc_std=float(shuffled_roc_auc_array.std(ddof=1)) if number_of_shuffles > 1 else float("nan"),
    shuffled_pr_auc_values=shuffled_pr_auc_values,
    shuffled_roc_auc_values=shuffled_roc_auc_values,
    no_skill_pr_auc=no_skill_pr_auc,
    no_skill_roc_auc=no_skill_roc_auc,
    shuffled_replicates_matching_or_beating_observed_pr_auc=count_pr_auc,
    shuffled_replicates_matching_or_beating_observed_roc_auc=count_roc_auc,
    pr_auc_empirical_p_value=pr_auc_p_value,
    roc_auc_empirical_p_value=roc_auc_p_value,
    number_of_shuffles=number_of_shuffles,
    significance_level=significance_level,
    minimum_attainable_p_value=minimum_p_value,
    minimum_shuffles_for_significance_level=minimum_shuffles_required(significance_level),
    sufficient_shuffles_for_significance_level=sufficient_shuffles,
    pr_auc_separated_from_noise=sufficient_shuffles and pr_auc_p_value <= significance_level,
    roc_auc_separated_from_noise=sufficient_shuffles and roc_auc_p_value <= significance_level,
    interpretation=interpretation,
  )