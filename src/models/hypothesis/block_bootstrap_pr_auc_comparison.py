"""
Block-bootstrap PR-AUC comparison test.

Logic: PR-AUC has no closed-form test for comparing two models on the same
test set, so uncertainty is estimated by resampling. An ordinary i.i.d.
(row-wise) bootstrap would understate that uncertainty here, because
consecutive weekly observations are not independent: each label looks two to
four weeks ahead, so neighbouring labels share part of their forward window,
shock weeks occur in runs, and the models' probabilities are smooth because
they are built from rolling and look-back windows. This module therefore uses a
CIRCULAR BLOCK BOOTSTRAP (Politis & Romano, 1992), a variant of the moving block
bootstrap (Kunsch, 1989): blocks of consecutive weeks are resampled with
replacement, which preserves the dependence within each block. Blocks wrap
around the end of the sample, so every observation is equally likely to be
drawn; without wrap-around, observations near the start and end of the sample
are drawn less often than those in the middle.

The same resampled rows are applied to both models in every replicate (a
paired bootstrap), so the comparison reflects the difference between the two
models on identical resampled weeks.

This module works directly on already-computed, held-out probability arrays,
so no retraining is required.
"""

from dataclasses import dataclass
from logging import Logger
from math import ceil
from typing import Optional

import numpy as np
from sklearn.metrics import average_precision_score

# Default block length, in weeks. Consecutive labels overlap for two weeks
# (horizon of weeks t+2 to t+4), but dependence in the data extends further:
# shock weeks occur in runs (the longest run in the test sample is 10 weeks) and
# the models' inputs use windows of up to 12 weeks. A block of 8 weeks is a
# compromise; results should be checked at shorter and longer blocks (for
# example 4 and 12 weeks), as the driver script does.
DEFAULT_BLOCK_LENGTH_WEEKS = 8

DEFAULT_NUMBER_OF_BOOTSTRAP_REPLICATES = 2000
MAX_REDRAW_ATTEMPTS = 100


@dataclass(frozen=True)
class BlockBootstrapComparisonResult:
  model_a_name: str
  model_b_name: str
  model_a_observed_pr_auc: float
  model_b_observed_pr_auc: float
  observed_pr_auc_difference: float
  bootstrap_difference_mean: float
  bootstrap_difference_std: float
  bootstrap_difference_ci_lower: float
  bootstrap_difference_ci_upper: float
  proportion_of_replicates_model_b_better: float
  confidence_level: float
  two_sided_p_value: float
  block_length: int
  number_of_bootstrap_replicates: int
  bootstrap_differences: list


def _build_circular_block_bootstrap_indices(
  sample_count: int,
  block_length: int,
  rng: np.random.Generator,
) -> np.ndarray:
  """
  Build one circular-block-bootstrap resample of row indices.

  Block start positions are drawn uniformly from all `sample_count` positions;
  each block takes `block_length` consecutive indices, wrapping around the end
  of the sample. Blocks are concatenated and truncated to `sample_count`, so
  every replicate has the original sample size.
  """
  if block_length < 1 or block_length >= sample_count:
    raise ValueError(
      f"block_length ({block_length}) must be at least 1 and smaller than the sample count ({sample_count})"
    )

  number_of_blocks_needed = ceil(sample_count / block_length)
  block_start_positions = rng.integers(low=0, high=sample_count, size=number_of_blocks_needed)
  within_block_offsets = np.arange(block_length)

  block_indices = (block_start_positions[:, None] + within_block_offsets[None, :]) % sample_count
  return block_indices.ravel()[:sample_count]


def run_block_bootstrap_pr_auc_comparison(
  target_test: np.ndarray,
  probability_model_a: np.ndarray,
  probability_model_b: np.ndarray,
  logger: Logger,
  model_a_name: str = "Model A",
  model_b_name: str = "Model B",
  block_length: Optional[int] = None,
  number_of_bootstrap_replicates: int = DEFAULT_NUMBER_OF_BOOTSTRAP_REPLICATES,
  confidence_level: float = 0.95,
  random_seed: int = 42,
) -> BlockBootstrapComparisonResult:
  """
  Compare two models' PR-AUC (average precision) on the same test partition via
  a paired circular block bootstrap, reporting a percentile confidence interval
  and a two-sided empirical p-value for the difference (model_b - model_a).

  Parameters
  ----------
  target_test
      True binary shock labels for the test partition, in chronological order
      (the block structure relies on this ordering).
  probability_model_a, probability_model_b
      Predicted shock probabilities from the two models, aligned to the SAME
      dates and order as target_test.
  logger
      Project logger.
  model_a_name, model_b_name
      Labels used for logging and the returned result.
  block_length
      Number of consecutive weeks per resampled block. If None, uses
      DEFAULT_BLOCK_LENGTH_WEEKS (8).
  number_of_bootstrap_replicates
      Number of bootstrap resamples. The default of 2000 gives reasonably
      stable 95% interval endpoints; fewer than about 1000 is only suitable for
      checking that the code runs.
  confidence_level
      Confidence level of the reported interval.
  random_seed
      Seed for the bootstrap random number generator.

  Returns
  -------
  BlockBootstrapComparisonResult

  Notes on the p-value
  --------------------
  The two-sided p-value is twice the smaller of the two tail proportions of the
  bootstrap distribution of the difference on either side of zero, each
  computed as (count + 1) / (B + 1) so that it is never exactly zero (the
  smallest value with B replicates is 2 / (B + 1)). This is the usual
  percentile-bootstrap convention and is approximate; the confidence interval
  is the primary summary.
  """
  target_test = np.asarray(target_test).astype(int)
  probability_model_a = np.asarray(probability_model_a, dtype=float)
  probability_model_b = np.asarray(probability_model_b, dtype=float)

  sample_count = len(target_test)
  if not (len(probability_model_a) == len(probability_model_b) == sample_count):
    raise ValueError(
      "target_test, probability_model_a and probability_model_b must be the same length "
      "and in the same (chronological) row order"
    )
  if not set(np.unique(target_test)).issubset({0, 1}):
    raise ValueError("target_test must contain only the binary labels 0 and 1")
  if not 0.0 < confidence_level < 1.0:
    raise ValueError("confidence_level must be strictly between 0 and 1")

  if block_length is None:
    block_length = DEFAULT_BLOCK_LENGTH_WEEKS

  if number_of_bootstrap_replicates < 1000:
    logger.warning(
      f"block bootstrap: only {number_of_bootstrap_replicates} replicates; percentile interval "
      f"endpoints and p-values are unreliable below about 1000"
    )

  model_a_observed_pr_auc = float(average_precision_score(target_test, probability_model_a))
  model_b_observed_pr_auc = float(average_precision_score(target_test, probability_model_b))
  observed_pr_auc_difference = model_b_observed_pr_auc - model_a_observed_pr_auc

  logger.info(
    f"block bootstrap: {model_a_name} PR-AUC = {model_a_observed_pr_auc:.4f}, "
    f"{model_b_name} PR-AUC = {model_b_observed_pr_auc:.4f}, "
    f"observed difference ({model_b_name} - {model_a_name}) = {observed_pr_auc_difference:.4f}"
  )
  logger.info(
    f"block bootstrap: block_length = {block_length} weeks, replicates = {number_of_bootstrap_replicates}"
  )

  rng = np.random.default_rng(random_seed)
  bootstrap_differences = np.empty(number_of_bootstrap_replicates, dtype=float)

  for replicate_index in range(number_of_bootstrap_replicates):
    # A resample can contain only one class when positives are rare; redraw
    # until both classes are present (average precision is undefined otherwise).
    for _ in range(MAX_REDRAW_ATTEMPTS):
      resampled_row_indices = _build_circular_block_bootstrap_indices(
        sample_count=sample_count, block_length=block_length, rng=rng
      )
      resampled_target = target_test[resampled_row_indices]
      positive_count = int(resampled_target.sum())
      if 0 < positive_count < sample_count:
        break
    else:
      raise RuntimeError(
        f"could not draw a resample containing both classes in {MAX_REDRAW_ATTEMPTS} attempts"
      )

    replicate_pr_auc_a = average_precision_score(resampled_target, probability_model_a[resampled_row_indices])
    replicate_pr_auc_b = average_precision_score(resampled_target, probability_model_b[resampled_row_indices])
    bootstrap_differences[replicate_index] = replicate_pr_auc_b - replicate_pr_auc_a

  lower_percentile = (1.0 - confidence_level) / 2.0 * 100.0
  upper_percentile = 100.0 - lower_percentile
  confidence_interval_lower = float(np.percentile(bootstrap_differences, lower_percentile))
  confidence_interval_upper = float(np.percentile(bootstrap_differences, upper_percentile))

  count_at_or_below_zero = int((bootstrap_differences <= 0.0).sum())
  count_at_or_above_zero = int((bootstrap_differences >= 0.0).sum())
  lower_tail_proportion = (count_at_or_below_zero + 1) / (number_of_bootstrap_replicates + 1)
  upper_tail_proportion = (count_at_or_above_zero + 1) / (number_of_bootstrap_replicates + 1)
  two_sided_p_value = float(min(1.0, 2.0 * min(lower_tail_proportion, upper_tail_proportion)))

  proportion_model_b_better = float((bootstrap_differences > 0.0).mean())

  logger.info(
    f"block bootstrap: {confidence_level * 100:.0f}% CI for the PR-AUC difference = "
    f"[{confidence_interval_lower:.4f}, {confidence_interval_upper:.4f}], two-sided p = {two_sided_p_value:.4f}"
  )
  if confidence_interval_lower <= 0.0 <= confidence_interval_upper:
    logger.info(
      f"block bootstrap: the interval includes zero, so the PR-AUC difference between {model_a_name} and "
      f"{model_b_name} is not statistically distinguishable from zero at the {confidence_level * 100:.0f}% level"
    )
  else:
    better_model_name = model_b_name if observed_pr_auc_difference > 0 else model_a_name
    logger.info(
      f"block bootstrap: the interval excludes zero; {better_model_name} has a statistically "
      f"distinguishable PR-AUC advantage at the {confidence_level * 100:.0f}% level"
    )

  return BlockBootstrapComparisonResult(
    model_a_name=model_a_name,
    model_b_name=model_b_name,
    model_a_observed_pr_auc=model_a_observed_pr_auc,
    model_b_observed_pr_auc=model_b_observed_pr_auc,
    observed_pr_auc_difference=observed_pr_auc_difference,
    bootstrap_difference_mean=float(bootstrap_differences.mean()),
    bootstrap_difference_std=float(bootstrap_differences.std(ddof=1)),
    bootstrap_difference_ci_lower=confidence_interval_lower,
    bootstrap_difference_ci_upper=confidence_interval_upper,
    proportion_of_replicates_model_b_better=proportion_model_b_better,
    confidence_level=confidence_level,
    two_sided_p_value=two_sided_p_value,
    block_length=block_length,
    number_of_bootstrap_replicates=number_of_bootstrap_replicates,
    bootstrap_differences=bootstrap_differences.tolist(),
  )