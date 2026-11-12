
import json

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from src.config.config import EXCHANGE_RATE_PLOT_FILE , FINAL_DATASET_END_DATE , FINAL_DATASET_START_DATE , \
  USDGH_DATASET_FILE , \
  DATA_SUMMARY_OUTPUT_DIR,COMBINED_DATASET_FILE
from src.config.looger_config import get_logger
from src.utils.utils import get_path
from src.data.visualize_data_splits import visualize_data_splits

DATE_COL = "date"
TARGET_COL = "target_shock_window"
SHOCK = 1
NON_SHOCK = 0

logger = get_logger(name=__name__, log_file="dataset_summary.log")


def load_dataset(input_path: Path, date_col: str, target_col: str) -> pd.DataFrame:
    """Load and validate the dataset without dropping rows with missing features."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path.resolve()}. Pass its path using --input."
        )

    df = pd.read_csv(input_path)
    missing_columns = [c for c in (date_col, target_col) if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required column(s): {missing_columns}. Available columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    unexpected = set(df[target_col].dropna().unique()) - {NON_SHOCK, SHOCK}
    if unexpected:
        raise ValueError(f"Unexpected target labels: {sorted(unexpected)}; expected 0, 1, or missing.")

    df = df.sort_values(date_col).reset_index(drop=True)
    if df[date_col].isna().any():
        logger.warning("%d row(s) have invalid/missing dates.", int(df[date_col].isna().sum()))
    return df


def build_dataset_summary(df: pd.DataFrame, date_col: str, target_col: str) -> dict:
    """Calculate overall dataset size, date coverage, labels, and predictor missingness."""
    n_rows = len(df)
    n_labelled = int(df[target_col].notna().sum())
    n_shocks = int((df[target_col] == SHOCK).sum())
    n_non_shocks = int((df[target_col] == NON_SHOCK).sum())
    valid_dates = df[date_col].dropna()
    predictors = [c for c in df.columns if c not in {date_col, target_col}]

    return {
        "input_rows": int(n_rows),
        "date_start": valid_dates.min().date().isoformat() if not valid_dates.empty else None,
        "date_end": valid_dates.max().date().isoformat() if not valid_dates.empty else None,
        "rows_with_invalid_or_missing_date": int(df[date_col].isna().sum()),
        "number_of_columns_including_date_and_target": int(df.shape[1]),
        "number_of_predictor_columns": len(predictors),
        "labelled_rows": n_labelled,
        "incomplete_or_missing_target_rows": int(df[target_col].isna().sum()),
        "shock_count": n_shocks,
        "non_shock_count": n_non_shocks,
        "shock_rate_among_labelled_pct": round(100 * n_shocks / n_labelled, 2) if n_labelled else None,
        "non_shock_rate_among_labelled_pct": round(100 * n_non_shocks / n_labelled, 2) if n_labelled else None,
        "missing_values_by_predictor": {c: int(df[c].isna().sum()) for c in predictors},
        "missing_percentage_by_predictor": {c: round(float(df[c].isna().mean() * 100), 2) for c in predictors},
    }


def build_shock_distribution(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Create counts and percentages for both classes and unlabelled rows."""
    total = len(df)
    labelled = int(df[target_col].notna().sum())
    rows = []
    for name, value in (("Non-shock", NON_SHOCK), ("Shock", SHOCK)):
        count = int((df[target_col] == value).sum())
        rows.append({
            "class": name,
            "target_value": value,
            "count": count,
            "percentage_of_labelled_rows": round(100 * count / labelled, 2) if labelled else 0.0,
            "percentage_of_all_rows": round(100 * count / total, 2) if total else 0.0,
        })

    missing = int(df[target_col].isna().sum())
    rows.append({
        "class": "Incomplete / missing target",
        "target_value": None,
        "count": missing,
        "percentage_of_labelled_rows": None,
        "percentage_of_all_rows": round(100 * missing / total, 2) if total else 0.0,
    })
    return pd.DataFrame(rows)


def save_distribution_plot(distribution: pd.DataFrame, output_path: Path) -> None:
    """Save a bar chart for the labelled shock and non-shock classes."""
    plot_df = distribution[distribution["class"].isin(["Non-shock", "Shock"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(plot_df["class"], plot_df["count"])
    ax.set_title("USD/GHS Shock and Non-Shock Distribution")
    ax.set_xlabel("Target class")
    ax.set_ylabel("Number of labelled weekly observations")
    ax.grid(axis="y", alpha=0.25)
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        ax.annotate(
            f'{int(row["count"])} ({row["percentage_of_labelled_rows"]:.1f}%)',
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_outputs(summary: dict, distribution: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_summary.json"
    csv_path = output_dir / "dataset_summary.csv"
    distribution_path = output_dir / "shock_distribution.csv"
    plot_path = output_dir / "shock_distribution.png"

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    flat_summary = {k: v for k, v in summary.items() if not isinstance(v, dict)}
    pd.DataFrame([{"metric": k, "value": v} for k, v in flat_summary.items()]).to_csv(csv_path, index=False)
    distribution.to_csv(distribution_path, index=False)
    save_distribution_plot(distribution, plot_path)

    for path in (json_path, csv_path, distribution_path, plot_path):
        logger.info("Saved: %s", path.resolve())


def plot_usd_ghs_exchange_rate (
  file_path: Path,
) :
  """
  Plot the USD/GHS exchange rate over the study period.

  Parameters


  Returns
  -------
  pd.DataFrame
      Filtered exchange-rate data used in the plot.
  """

  df = pd.read_csv( file_path, parse_dates = [ 'date' ] )
  start_year = pd.to_datetime( FINAL_DATASET_START_DATE ).year
  end_year = pd.to_datetime( FINAL_DATASET_END_DATE ).year

  # --------------------------------------------------------
  # Validate required columns
  # --------------------------------------------------------

  required_columns = [
    "date" ,
    "usdgh_exchange_rate"
  ]

  missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
  ]

  if missing_columns :
    raise ValueError(
      f"Missing required columns: {missing_columns}"
    )

  # --------------------------------------------------------
  # Prepare dates and exchange rate
  # --------------------------------------------------------

  df[ "date" ] = pd.to_datetime(
    df[ "date" ] ,
    errors = "coerce"
  )

  df[ "usdgh_exchange_rate" ] = pd.to_numeric(
    df[ "usdgh_exchange_rate" ] ,
    errors = "coerce"
  )

  df = df.dropna(
    subset = [
      "date" ,
      "usdgh_exchange_rate"
    ]
  )

  # --------------------------------------------------------
  # Restrict to study period
  # --------------------------------------------------------

  start_date = pd.to_datetime( FINAL_DATASET_START_DATE )
  end_date = pd.to_datetime( FINAL_DATASET_END_DATE )

  df = df[
    (df[ "date" ] >= start_date)
    & (df[ "date" ] <= end_date)
    ].copy()

  if df.empty :
    raise ValueError(
      "No exchange-rate observations found "
      "within the specified study period."
    )

  # Sort chronologically
  df = df.sort_values( "date" )
  df = df.set_index("date").resample("W-SUN")["usdgh_exchange_rate"].last().reset_index()

  df[ "year" ] = df[ "date" ].dt.year

  # Calculate annual maximum
  annual_max = (
    df.groupby( "year" )[ "usdgh_exchange_rate" ]
    .max()
    .reset_index()
  )

  # --------------------------------------------------------
  # Plot
  # --------------------------------------------------------

  plt.figure( figsize = (14 , 7) )

  plt.plot(
    # df[ "date" ] ,
    # df[ "usdgh_exchange_rate" ] ,
    annual_max[ "year" ] ,
    annual_max[ "usdgh_exchange_rate" ] ,
    marker = "o" ,
    linewidth = 1.5
  )

  plt.xlabel(
    "Year"
  )

  plt.ylabel(
    "USD/GHS exchange rate"
  )

  plt.title(
    f"USD/GHS Exchange Rate Evolution, {start_year}–{end_year}"
  )

  # ax = plt.gca()
  #
  # ax.xaxis.set_major_locator(
  #   mdates.YearLocator( 1 )
  # )
  #
  # ax.xaxis.set_major_formatter(
  #   mdates.DateFormatter( "%Y" )
  # )
  plt.xticks(
    range( start_year , end_year + 1 ) ,
    rotation = 45
  )

  plt.grid(
    True ,
    alpha = 0.3
  )

  plt.tight_layout()

  # --------------------------------------------------------
  # Save figure
  # --------------------------------------------------------



  plt.savefig(
    get_path(EXCHANGE_RATE_PLOT_FILE) ,
    dpi = 300 ,
    bbox_inches = "tight"
  )

  plt.close()

  print(
    f"Figure saved to: {EXCHANGE_RATE_PLOT_FILE}"
  )

  return df

def summarize_shocks_by_split (
  df: pd.DataFrame ,
  date_col: str = "date" ,
  target_col: str = "target_shock_window" ,
) -> pd.DataFrame :
  """
  Group weekly observations by chronological dataset split
  and shock/non-shock class.

  Train:      2010-01-01 to 2019-12-31
  Validation: 2020-01-01 to 2023-12-31
  Test:       2024-01-01 to 2025-12-31

  Returns counts and percentages for each split.
  """

  data = df.copy()
  data[ date_col ] = pd.to_datetime( data[ date_col ] )

  # Define chronological splits
  conditions = [
    data[ date_col ].between( "2010-01-01" , "2019-12-31" ) ,
    data[ date_col ].between( "2020-01-01" , "2023-12-31" ) ,
    data[ date_col ].between( "2024-01-01" , "2025-12-31" ) ,
  ]

  labels = [ "Train" , "Validation" , "Test" ]

  data[ "split" ] = pd.Series( index = data.index , dtype = "object" )

  for condition , label in zip( conditions , labels ) :
    data.loc[ condition , "split" ] = label

  # Keep observations within the defined study period
  data = data.dropna( subset = [ "split" , target_col ] )

  # Count shock and non-shock observations
  summary = (
    data.groupby( [ "split" , target_col ] , observed = True )
    .size()
    .unstack( fill_value = 0 )
    .reindex( labels , fill_value = 0 )
  )

  # Ensure both target classes exist as columns
  summary = summary.reindex( columns = [ 0 , 1 ] , fill_value = 0 )
  summary.columns = [ "Non-shock" , "Shock" ]

  summary[ "Total" ] = summary[ "Non-shock" ] + summary[ "Shock" ]

  # Calculate percentages within each split
  summary[ "Non-shock (%)" ] = (
    summary[ "Non-shock" ] / summary[ "Total" ].replace( 0 , pd.NA ) * 100
  )
  summary[ "Shock (%)" ] = (
    summary[ "Shock" ] / summary[ "Total" ].replace( 0 , pd.NA ) * 100
  )

  summary.index.name = "Dataset split"
  summary.reset_index()
  summary.to_csv(
    get_path("artifact/summary/shock_distribution_by_split.csv") ,
    index = False ,
  )

  return summary.reset_index()







def main() -> None:

    file_input =  get_path(USDGH_DATASET_FILE)
    output_dir = get_path(DATA_SUMMARY_OUTPUT_DIR)

    df = load_dataset(input_path = file_input, date_col = DATE_COL, target_col = TARGET_COL)
    summary = build_dataset_summary(df, date_col = DATE_COL, target_col = TARGET_COL)
    distribution = build_shock_distribution(df, target_col = TARGET_COL)
    save_outputs(summary, distribution, output_dir = output_dir)
    logger.info(
        "Rows=%d | labelled=%d | shocks=%d | non-shocks=%d | incomplete targets=%d",
        summary["input_rows"], summary["labelled_rows"], summary["shock_count"],
        summary["non_shock_count"], summary["incomplete_or_missing_target_rows"],
    )

    plot_usd_ghs_exchange_rate(get_path(COMBINED_DATASET_FILE))
    visualize_data_splits()
    summarize_shocks_by_split(df)



if __name__ == "__main__":
    main()
