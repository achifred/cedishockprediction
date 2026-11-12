import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)

from src.config.config import CONFUSION_MATRIX_PLOT_FILE , ENSEMBLE_PREDICTION_FILE , LSTM_TEST_PREDICTION_FILE , \
  PRECISION_RECALL_CURVE_PLOT_FILE , \
  PREDICTED_PROBABILITY_DISTRIBUTION_PLOT_FILE , XGBOOST_TEST_PREDICTION_FILE
from src.utils.utils import get_path


# ============================================================
# 1. PRECISION–RECALL CURVE
# ============================================================

def plot_precision_recall_curves(
    prediction_files,
    output_path,
    model_names=None,
):
    """
    Plot Precision–Recall curves for one or more models.

    Parameters
    ----------
    prediction_files : dict
        Dictionary mapping model names to prediction CSV paths.

        Example:
        {
            "XGBoost": "outputs/test/xgb_test_predictions.csv",
            "LSTM": "outputs/test/lstm_test_predictions.csv",
            "Ensemble": "outputs/test/ensemble_test_predictions.csv"
        }

    output_path : str or Path
        Path where the figure will be saved.

    model_names : list, optional
        Models to include. If None, all models in prediction_files
        are plotted.

    Required CSV columns
    --------------------
    y_true
    predicted_probability
    """

    plt.figure(figsize=(9, 7))

    if model_names is None:
        model_names = list(prediction_files.keys())

    for model_name in model_names:

        df = pd.read_csv(prediction_files[model_name]["file_name"])

        y_true = df[prediction_files[model_name]["y_true_col"]].astype(int)
        y_prob = df[prediction_files[model_name]["y_prob_col"]].astype(float)

        precision, recall, _ = precision_recall_curve(
            y_true,
            y_prob
        )

        average_precision = average_precision_score(
            y_true,
            y_prob
        )

        plt.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{model_name} (AP = {average_precision:.3f})"
        )

    # No-skill baseline = proportion of positive observations
    first_df = pd.read_csv(prediction_files[model_names[0]]["file_name"])
    positive_rate = first_df[prediction_files[model_names[0]]["y_true_col"]].mean()

    plt.axhline(
        y=positive_rate,
        linestyle="--",
        linewidth=1.5,
        label=f"No-skill baseline ({positive_rate:.3f})"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Precision–Recall curve saved to: {output_path}")







# ============================================================
# 2. CONFUSION MATRICES
# ============================================================
def plot_confusion_matrices (
  prediction_files ,
  output_path ,
  model_names = None ,
) :
  """
  Plot and save a separate confusion matrix for each model.

  Parameters
  ----------
  prediction_files : dict
      Dictionary mapping model names to prediction CSV information.

      Example:
      {
          "LSTM": {
              "file_name": "lstm_predictions.csv",
              "y_true_col": "y_true",
              "y_pred_col": "predicted_class"
          },
          "XGBoost": {
              "file_name": "xgboost_predictions.csv",
              "y_true_col": "y_true",
              "y_pred_col": "predicted_class"
          }
      }

  output_path : str or Path
      Directory where the confusion matrix figures will be saved.

  model_names : list, optional
      Models to include. If None, all models in prediction_files
      are used.

  Required CSV columns
  --------------------
  The columns specified by y_true_col and y_pred_col.
  """

  if model_names is None :
    model_names = list( prediction_files.keys() )

  # Create output directory
  # output_path = Path( output_path )
  # output_path.mkdir( parents = True , exist_ok = True )

  for model_name in model_names :

    # Read predictions
    df = pd.read_csv(
      prediction_files[ model_name ][ "file_name" ]
    )

    y_true = df[
      prediction_files[ model_name ][ "y_true_col" ]
    ].astype( int )

    y_pred = df[
      prediction_files[ model_name ][ "y_pred_col" ]
    ].astype( int )

    # Calculate confusion matrix
    cm = confusion_matrix(
      y_true ,
      y_pred ,
      labels = [ 0 , 1 ]
    )

    # Create figure
    fig , ax = plt.subplots(
      figsize = (5 , 5)
    )

    # Display confusion matrix
    ax.imshow( cm )

    # Add values to cells
    for i in range( 2 ) :
      for j in range( 2 ) :
        ax.text(
          j ,
          i ,
          str( cm[ i , j ] ) ,
          ha = "center" ,
          va = "center" ,
          fontsize = 14
        )

    # Axis labels
    ax.set_xticks( [ 0 , 1 ] )
    ax.set_yticks( [ 0 , 1 ] )

    ax.set_xticklabels( [
      "Safe (0)" ,
      "Shock (1)"
    ] )

    ax.set_yticklabels( [
      "Safe (0)" ,
      "Shock (1)"
    ] )

    ax.set_xlabel( "Predicted Class" )
    ax.set_ylabel( "Actual Class" )

    ax.set_title(
      f"{model_name}\nConfusion Matrix"
    )

    # Make layout neat
    plt.tight_layout()

    # Create a safe filename from the model name
    safe_model_name = (
      model_name.lower()
      .replace( " " , "_" )
      .replace( "-" , "_" )
    )

    output_file = (
      output_path /
      f"{safe_model_name}_confusion_matrix.png"
    )

    # Save figure
    plt.savefig(
      output_file ,
      dpi = 300 ,
      bbox_inches = "tight"
    )

    plt.close()

    print(
      f"{model_name} confusion matrix saved to: "
      f"{output_file}"
    )





# ============================================================
# 3. PREDICTED PROBABILITIES WITH FORWARD SHOCK WINDOWS
# ============================================================

def plot_predicted_probabilities(
    prediction_files,
    output_path,
    model_names=None,
    min_horizon=2,
    max_horizon=4,
):
    """
    Plot predicted shock probabilities over the test period while
    explicitly representing the forward shock-window definition.

    Target definition:

        P*_t = max(P_{t+2}, P_{t+3}, P_{t+4})

        D_t = P*_t / P_t - 1

        Y_t = 1 if D_t >= 0.03

    Therefore, a shock-warning observation at date t means that
    a qualifying depreciation occurs sometime during the forward
    window [t+2, t+4].

    Parameters
    ----------
    prediction_files : dict
        Dictionary mapping model names to prediction CSV paths.

        Example:
        {
            "XGBoost": "outputs/test/xgb_test_predictions.csv",
            "LSTM": "outputs/test/lstm_test_predictions.csv",
            "Ensemble": "outputs/test/ensemble_test_predictions.csv"
        }

    output_path : str or Path
        Location where the figure will be saved.

    model_names : list, optional
        Models to plot. If None, all models are plotted.

    min_horizon : int
        Minimum forward horizon used by the shock definition.

    max_horizon : int
        Maximum forward horizon used by the shock definition.

    Required CSV columns
    --------------------
    date
    y_true
    predicted_probability

    Optional CSV column
    -------------------
    threshold
    """

    if model_names is None:
        model_names = list(prediction_files.keys())

    # ========================================================
    # Load prediction data
    # ========================================================

    predictions = {}

    for model_name in model_names:

        df = pd.read_csv(
            prediction_files[model_name]["file_name"]
        )

        df["date"] = pd.to_datetime(
            df["date"]
        )

        df = df.sort_values("date").reset_index(drop=True)

        predictions[model_name] = df

    # ========================================================
    # Use the first model as the reference for actual labels
    # ========================================================

    reference_df = predictions[model_names[0]].copy()

    # ========================================================
    # Determine decision threshold
    # ========================================================

    threshold = 0.50

    if "threshold" in reference_df.columns:

        threshold_values = (
            reference_df["threshold"]
            .dropna()
            .unique()
        )

        if len(threshold_values) > 0:
            threshold = float(threshold_values[0])

    # ========================================================
    # Create figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(15, 8)
    )

    # ========================================================
    # Identify contiguous shock-warning periods
    # ========================================================

    shock_dates = reference_df.loc[
        reference_df[prediction_files[model_names[0]]["y_pred_col"]] == 1,
        "date"
    ].tolist()

    # --------------------------------------------------------
    # Highlight the forward target window.
    #
    # If date t is labelled as a shock warning, the actual
    # qualifying depreciation occurs somewhere between t+2
    # and t+4 weeks.
    # --------------------------------------------------------

    for prediction_date in shock_dates:

        forward_start = (
            prediction_date
            + pd.Timedelta(weeks=min_horizon)
        )

        forward_end = (
            prediction_date
            + pd.Timedelta(weeks=max_horizon)
        )

        ax.axvspan(
            forward_start,
            forward_end,
            alpha=0.15
        )

        # Mark the prediction date itself
        ax.axvline(
            prediction_date,
            linestyle="--",
            linewidth=0.8,
            alpha=0.35
        )

    # ========================================================
    # Plot model probabilities
    # ========================================================

    for model_name in model_names:

        df = predictions[model_name]

        ax.plot(
            df["date"],
            df[prediction_files[model_name]["y_prob_col"]],
            linewidth=1.8,
            label=f"{model_name} probability"
        )

    # ========================================================
    # Plot decision threshold
    # ========================================================

    ax.axhline(
        threshold,
        linestyle=":",
        linewidth=2,
        label=f"Warning threshold ({threshold:.2f})"
    )

    # ========================================================
    # Formatting
    # ========================================================

    ax.set_xlabel(
        "Prediction date (t)"
    )

    ax.set_ylabel(
        "Predicted probability of shock"
    )

    ax.set_title(
        "Predicted Shock Probabilities and Forward Shock Windows"
    )

    ax.set_ylim(
        0,
        1
    )

    ax.grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Add explanatory text
    # --------------------------------------------------------

    ax.text(
        0.01,
        0.02,
        (
            "Shaded regions: forward shock window "
            f"(t+{min_horizon} to t+{max_horizon}) "
            "following an observation labelled as a shock warning"
        ),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom"
    )

    ax.legend(
        loc="upper left"
    )

    plt.tight_layout()

    # ========================================================
    # Save figure
    # ========================================================

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Predicted probability plot saved to: {output_path}"
    )



def main():
    # Example usage
    prediction_files = {
        "XGBoost": {
          "file_name":get_path(XGBOOST_TEST_PREDICTION_FILE),
          "y_true_col":"actual_shock_window",
          "y_prob_col":"xgb_shock_probability",
          "y_pred_col":"xgb_binary_alert"
        },
        "LSTM": {
          "file_name":get_path(LSTM_TEST_PREDICTION_FILE),
          "y_true_col":"actual_shock_window",
          "y_prob_col":"lstm_shock_probability",
          "y_pred_col":"lstm_binary_alert"
        },
        "Ensemble": {
          "file_name":get_path(ENSEMBLE_PREDICTION_FILE),
          "y_true_col":"actual_shock_window",
          "y_prob_col":"ensemble_probability",
          "y_pred_col":"ensemble_prediction"
        }
    }

    plot_precision_recall_curves(
        prediction_files=prediction_files,
        output_path=get_path(PRECISION_RECALL_CURVE_PLOT_FILE)
    )

    plot_confusion_matrices(
        prediction_files=prediction_files,
        output_path=get_path(CONFUSION_MATRIX_PLOT_FILE)
    )

    plot_predicted_probabilities(
        prediction_files=prediction_files,
        output_path=get_path(PREDICTED_PROBABILITY_DISTRIBUTION_PLOT_FILE)
    )


if __name__ == "__main__":
    main()