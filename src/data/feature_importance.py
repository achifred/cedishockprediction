import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score

from src.config.config import LSTM_FEATURE_IMPORTANCE_FILE , LSTM_FEATURE_IMPORTANCE_PLOT_FILE , \
  XGB_FEATURE_IMPORTANCE_FILE , XGB_FEATURE_IMPORTANCE_PLOT_FILE
from src.utils.utils import get_path


# ============================================================
# XGBOOST FEATURE IMPORTANCE
# ============================================================

def extract_xgb_feature_importance(
    model,
    feature_names,
):
    """
    Extract gain-based feature importance from a trained
    XGBoost model.

    Returns
    -------
    pd.DataFrame
        Feature importance ranked from highest to lowest.
    """

    gain_importance = model.feature_importances_

    importance_df = pd.DataFrame( {
      "feature" : feature_names ,
      "importance" : gain_importance ,
    } ).sort_values( "importance" , ascending = False )

    total_importance = importance_df["importance"].sum()

    if total_importance > 0:

        importance_df["importance_pct"] = (
            importance_df["importance"]
            / total_importance
            * 100
        )

    else:

        importance_df["importance_pct"] = 0.0

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return importance_df


# ============================================================
# LSTM PREDICTION - TENSORFLOW / KERAS
# ============================================================

def lstm_predict(
    model,
    X,
):
    """
    Generate probability predictions from a trained
    TensorFlow/Keras LSTM model.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Trained LSTM model.

    X : np.ndarray
        LSTM input with shape:

            (samples, lookback, features)

    Returns
    -------
    np.ndarray
        Predicted probabilities.
    """

    probabilities = model.predict(
        X,
        verbose=0
    )

    return np.asarray(
        probabilities
    ).ravel()


# ============================================================
# LSTM PERMUTATION IMPORTANCE
# ============================================================

def extract_lstm_permutation_importance(
    model,
    X,
    y,
    feature_names,
    n_repeats=10,
    random_state=42,
):
    """
    Calculate LSTM feature importance using permutation
    importance.

    Importance is measured as the reduction in PR-AUC after
    permuting one feature across samples.

    The LSTM input is assumed to have shape:

        (samples, lookback, features)

    For each feature, the complete temporal sequence belonging
    to that feature is shuffled across samples.

    Example:

        X.shape = (800, 12, 10)

    If feature_idx = 3, then the complete 12-week history of
    feature 3 is shuffled between samples.

    This preserves the temporal ordering within each sample.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Trained LSTM model.

    X : np.ndarray
        LSTM input.

    y : np.ndarray
        Binary target.

    feature_names : list
        Feature names in the same order as X's final dimension.

    n_repeats : int
        Number of permutations for each feature.

    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Feature importance ranked from highest to lowest.
    """

    rng = np.random.default_rng(
        random_state
    )

    # --------------------------------------------------------
    # Baseline performance
    # --------------------------------------------------------

    baseline_probabilities = lstm_predict(
        model=model,
        X=X
    )

    baseline_pr_auc = average_precision_score(
        y,
        baseline_probabilities
    )

    results = []

    # --------------------------------------------------------
    # Permute each feature
    # --------------------------------------------------------

    for feature_idx, feature_name in enumerate(
        feature_names
    ):

        score_drops = []

        for _ in range(n_repeats):

            X_permuted = X.copy()

            # Shuffle complete 12-week feature histories
            permutation = rng.permutation(
                X.shape[0]
            )

            X_permuted[:, :, feature_idx] = (
                X[
                    permutation,
                    :,
                    feature_idx
                ]
            )

            # Predict using the permuted data
            permuted_probabilities = lstm_predict(
                model=model,
                X=X_permuted
            )

            permuted_pr_auc = average_precision_score(
                y,
                permuted_probabilities
            )

            score_drop = (
                baseline_pr_auc
                - permuted_pr_auc
            )

            score_drops.append(
                score_drop
            )

        results.append({
            "feature": feature_name,
            "importance": np.mean(score_drops),
            "std": np.std(score_drops),
        })

    importance_df = pd.DataFrame(
        results
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return importance_df


# ============================================================
# XGBOOST GRAPH
# ============================================================

def plot_xgb_feature_importance(
    importance_df,
    top_n=10,
):
    """
    Plot the top XGBoost features using gain-based importance.
    """

    plot_df = (
        importance_df
        .head(top_n)
        .sort_values("importance_pct")
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        plot_df["feature"],
        plot_df["importance_pct"]
    )

    plt.xlabel(
        "Gain-based importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"XGBoost Feature Importance "
        f"(Top {top_n})"
    )

    plt.grid(
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(
        get_path(XGB_FEATURE_IMPORTANCE_PLOT_FILE),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# LSTM GRAPH
# ============================================================

def plot_lstm_feature_importance(
    importance_df,
    top_n=10,
):
    """
    Plot the top LSTM features using permutation importance.
    """

    plot_df = (
        importance_df
        .head(top_n)
        .sort_values("importance")
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        plot_df["feature"],
        plot_df["importance"]
    )

    plt.xlabel(
        "Decrease in PR-AUC after permutation (PR-AUC_baseline−PR-AUC_permuted)"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"Feature Importance "
        f"(Top {top_n})"
    )

    plt.axvline(
        0,
        linewidth=1
    )

    plt.grid(
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()



    plt.savefig(
        get_path(LSTM_FEATURE_IMPORTANCE_PLOT_FILE),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# COMPLETE FEATURE-IMPORTANCE PIPELINE
# ============================================================

def generate_xgb_feature_importance(
    xgb_model,
    feature_names,
    top_n=10,

):
    """
    Generate feature-importance tables and graphs for
    both XGBoost.

    Outputs
    -------
    xgb_feature_importance.csv
    xgb_feature_importance.png

    """


    # ========================================================
    # XGBOOST
    # ========================================================

    print(
        "Extracting XGBoost feature importance..."
    )

    xgb_importance = (
        extract_xgb_feature_importance(
            model=xgb_model,
            feature_names=feature_names
        )
    )

    xgb_importance.to_csv(
        get_path(XGB_FEATURE_IMPORTANCE_FILE),
        index=False
    )

    plot_xgb_feature_importance(
        importance_df=xgb_importance,
        top_n=top_n
    )




def generate_lstm_feature_importance(lstm_model, X_lstm, y_lstm, feature_names, top_n=10, n_repeats=10):
  lstm_importance = (
    extract_lstm_permutation_importance(
      model = lstm_model ,
      X = X_lstm ,
      y = y_lstm ,
      feature_names = feature_names ,
      n_repeats = n_repeats ,
      random_state = 42
    )
  )

  lstm_importance.to_csv(
    get_path(LSTM_FEATURE_IMPORTANCE_FILE) ,
    index = False
  )

  plot_lstm_feature_importance(
    importance_df = lstm_importance ,
    top_n = top_n
  )

  print(
    "\nFeature importance analysis complete."
  )