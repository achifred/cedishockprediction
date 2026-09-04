"""
Diagnostic Leave-One-Episode-Out (LOEO) validation for the cedi-shock LSTM.

This version keeps the original model/training design but adds:
1. Non-overlapping train/validation/test ranges.
2. Per-episode predictions and probability-separation diagnostics.
3. Episode characteristics before and during the prediction window.
4. Good/moderate/poor episode grouping.
5. Feature-level comparisons between good and poor episodes.
6. Episode duration / feature-change analysis.
7. CSV outputs and diagnostic plots.

Important:
- ROC-AUC and PR-AUC are calculated on the fold test window.
- The threshold is selected from validation only.
- Feature characteristics are descriptive. They are NOT used to train the model.
- "prediction_window" means 20 to 10 calendar days before the episode start,
  matching the intended 10-20 day-ahead forecasting idea.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc as sk_auc,
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense

from src.config.config import USDGH_DATASET_FILE
from src.utils.utils import get_path

warnings.filterwarnings("ignore", category=RuntimeWarning)

RANDOM_SEED = 42
TIME_STEPS = 30
BUFFER_DAYS = 45

FEATURE_COLS = [
    "net_reserve_momentum",
    "tbill_mpr_spread_change",
    "global_rate_gap_change",
    "gold_return",
    "crude_return",
    "food_inf_change",
    "non_food_inf_change",
    "crude_log_volume",
    "gold_log_volume",
  "interbank_mpr_spread_change",
  "net_reserve_momentum_30d",
  "net_reserve_momentum_60d",
  "severe_reserve_depletion"
]


# ---------------------------------------------------------------------------
# 1. Episode identification
# ---------------------------------------------------------------------------
def identify_shock_episodes(
    df: pd.DataFrame,
    target_col: str = "target_shock_window",
    gap_days: int = 14,
) -> pd.DataFrame:
    """Group positive target dates into distinct shock episodes."""
    pos_dates = df.index[df[target_col] == 1].sort_values()

    if len(pos_dates) == 0:
        raise ValueError("No positive shock windows found in target column.")

    gaps = pos_dates.to_series().diff().dt.days
    episode_id = (gaps > gap_days).cumsum()

    episodes = (
        pos_dates.to_series()
        .groupby(episode_id)
        .agg(start="min", end="max", n_days="count")
        .reset_index(drop=True)
    )

    episodes["episode_id"] = np.arange(len(episodes))
    return episodes[
        ["episode_id", "start", "end", "n_days"]
    ]


# ---------------------------------------------------------------------------
# 2. Build non-overlapping validation/test ranges
# ---------------------------------------------------------------------------
def _initial_range(start, end, buffer_days):
    return [
        start - pd.Timedelta(days=buffer_days),
        end + pd.Timedelta(days=buffer_days),
    ]


def build_loeo_folds(
    episodes: pd.DataFrame,
    buffer_days: int = BUFFER_DAYS,
):
    """
    For each held-out episode:
      - test_range surrounds the test episode.
      - validation uses the chronologically nearest other episode.
      - overlapping test/validation buffers are automatically shrunk.

    This prevents validation and test windows from sharing dates.
    """
    folds = []
    n = len(episodes)

    for i in range(n):
        test_episode_start = episodes.loc[i, "start"]
        test_episode_end = episodes.loc[i, "end"]

        other_idx = [j for j in range(n) if j != i]
        distances = [
            abs(
                (
                    episodes.loc[j, "start"]
                    - episodes.loc[i, "start"]
                ).days
            )
            for j in other_idx
        ]
        val_idx = other_idx[int(np.argmin(distances))]

        val_episode_start = episodes.loc[val_idx, "start"]
        val_episode_end = episodes.loc[val_idx, "end"]

        test_start, test_end = _initial_range(
            test_episode_start,
            test_episode_end,
            buffer_days,
        )
        val_start, val_end = _initial_range(
            val_episode_start,
            val_episode_end,
            buffer_days,
        )

        # If ranges overlap, shrink the buffers around the two episode cores.
        if test_start <= val_end and val_start <= test_end:
            if test_episode_end < val_episode_start:
                gap = (val_episode_start - test_episode_end).days
            elif val_episode_end < test_episode_start:
                gap = (test_episode_start - val_episode_end).days
            else:
                raise ValueError(
                    f"Episodes {i} and {val_idx} overlap; "
                    "check episode identification."
                )

            safe_buffer = max(0, gap // 2)

            test_start, test_end = _initial_range(
                test_episode_start,
                test_episode_end,
                safe_buffer,
            )
            val_start, val_end = _initial_range(
                val_episode_start,
                val_episode_end,
                safe_buffer,
            )

            # Final one-day guard against touching/overlapping ranges.
            if test_episode_end < val_episode_start:
                test_end = min(
                    test_end,
                    val_episode_start - pd.Timedelta(days=1),
                )
                val_start = max(
                    val_start,
                    test_episode_end + pd.Timedelta(days=1),
                )
            else:
                val_end = min(
                    val_end,
                    test_episode_start - pd.Timedelta(days=1),
                )
                test_start = max(
                    test_start,
                    val_episode_end + pd.Timedelta(days=1),
                )

        folds.append(
            {
                "fold_id": i,
                "test_episode_id": int(episodes.loc[i, "episode_id"]),
                "test_episode_start": test_episode_start,
                "test_episode_end": test_episode_end,
                "test_range": (test_start, test_end),
                "val_episode_id": int(
                    episodes.loc[val_idx, "episode_id"]
                ),
                "val_episode_start": val_episode_start,
                "val_episode_end": val_episode_end,
                "val_range": (val_start, val_end),
            }
        )

    return folds


# ---------------------------------------------------------------------------
# 3. Train segments
# ---------------------------------------------------------------------------
def get_train_segments(df: pd.DataFrame, exclude_ranges: list):
    """Return contiguous segments after removing validation/test ranges."""
    exclude_ranges = sorted(
        exclude_ranges,
        key=lambda r: r[0],
    )

    segments = []
    cursor = df.index.min()

    for ex_start, ex_end in exclude_ranges:
        if cursor < ex_start:
            seg = df.loc[
                cursor : ex_start - pd.Timedelta(days=1)
            ]
            if len(seg) > 0:
                segments.append(seg)

        cursor = max(
            cursor,
            ex_end + pd.Timedelta(days=1),
        )

    tail = df.loc[cursor:]
    if len(tail) > 0:
        segments.append(tail)

    return [s for s in segments if len(s) > 0]


# ---------------------------------------------------------------------------
# 4. Sequence creation
# ---------------------------------------------------------------------------
def create_3d_sequence(
    scaled_feature,
    target,
    dates,
    time_steps=TIME_STEPS,
):
    """Create 30-step sequences without crossing data gaps."""
    assert len(scaled_feature) == len(target) == len(dates)

    if len(scaled_feature) <= time_steps:
        return (
            np.empty(
                (0, time_steps, scaled_feature.shape[1])
            ),
            np.empty((0,)),
            np.empty((0,), dtype="datetime64[ns]"),
        )

    feature_3d = []
    target_3d = []
    sequence_dates = []

    for i in range(len(scaled_feature) - time_steps):
        feature_3d.append(
            scaled_feature[i : i + time_steps]
        )
        target_3d.append(target[i + time_steps])
        sequence_dates.append(dates[i + time_steps])

    return (
        np.array(feature_3d),
        np.array(target_3d),
        np.array(sequence_dates),
    )


def build_sequences_from_segments(
    segments,
    scaler,
    target_col,
    time_steps=TIME_STEPS,
):
    """Scale and sequence-build each contiguous segment independently."""
    all_X = []
    all_y = []
    all_dates = []
    n_features = None

    for seg in segments:
        target_raw = seg[target_col].values
        feature_raw = seg.drop(columns=[target_col]).values
        n_features = feature_raw.shape[1]

        feature_scaled = scaler.transform(feature_raw)

        X, y, d = create_3d_sequence(
            feature_scaled,
            target_raw,
            seg.index.values,
            time_steps=time_steps,
        )

        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            all_dates.append(d)

    if not all_X:
        n_features = n_features or 0
        return (
            np.empty(
                (0, time_steps, n_features)
            ),
            np.empty((0,)),
            np.empty((0,)),
        )

    return (
        np.concatenate(all_X),
        np.concatenate(all_y),
        np.concatenate(all_dates),
    )


# ---------------------------------------------------------------------------
# 5. Model
# ---------------------------------------------------------------------------
def build_lstm_model(input_shape):
    model = Sequential(
        [
            Input(shape=input_shape),
            Bidirectional(
                LSTM(
                    units=64,
                    return_sequences=True,
                )
            ),
            Dropout(0.2),
            LSTM(
                units=32,
                return_sequences=False,
            ),
            Dropout(0.2),
            Dense(
                units=16,
                activation="relu",
            ),
            Dense(
                units=1,
                activation="sigmoid",
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(
                curve="PR",
                name="auc_pr",
            ),
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
        ],
    )

    return model


# ---------------------------------------------------------------------------
# 6. Episode characteristics
# ---------------------------------------------------------------------------
def calculate_episode_characteristics(
    df: pd.DataFrame,
    episodes: pd.DataFrame,
    pre_days: int = 45,
    prediction_start: int = 20,
    prediction_end: int = 10,
):
    """
    Describe each episode using:
      - pre-shock period
      - 20-to-10-day prediction/precursor window
      - episode itself

    These statistics are calculated independently of the model.
    """
    records = []

    usable_features = [
        f for f in FEATURE_COLS
        if f in df.columns
    ]

    for _, episode in episodes.iterrows():
        start = episode["start"]
        end = episode["end"]

        pre_start = start - pd.Timedelta(days=pre_days)
        pre_end = start - pd.Timedelta(days=1)

        prediction_window_start = (
            start - pd.Timedelta(days=prediction_start)
        )
        prediction_window_end = (
            start - pd.Timedelta(days=prediction_end)
        )

        pre_df = df.loc[pre_start:pre_end]
        prediction_df = df.loc[
            prediction_window_start:prediction_window_end
        ]
        episode_df = df.loc[start:end]

        record = {
            "episode_id": int(episode["episode_id"]),
            "episode_start": start,
            "episode_end": end,
            "episode_duration": int(episode["n_days"]),
            "pre_rows": len(pre_df),
            "prediction_window_rows": len(prediction_df),
            "episode_rows": len(episode_df),
        }

        for feature in usable_features:
            pre_mean = pre_df[feature].mean()
            prediction_mean = prediction_df[feature].mean()

            record[
                f"{feature}_pre_mean"
            ] = pre_mean

            record[
                f"{feature}_pre_std"
            ] = pre_df[feature].std()

            record[
                f"{feature}_prediction_mean"
            ] = prediction_mean

            record[
                f"{feature}_prediction_std"
            ] = prediction_df[feature].std()

            record[
                f"{feature}_episode_mean"
            ] = episode_df[feature].mean()

            # Change from the general pre-shock state
            # to the 20-to-10-day prediction window.
            record[
                f"{feature}_prediction_change"
            ] = prediction_mean - pre_mean

            # Range in the prediction window.
            if len(prediction_df) > 0:
                record[
                    f"{feature}_prediction_range"
                ] = (
                    prediction_df[feature].max()
                    - prediction_df[feature].min()
                )
            else:
                record[
                    f"{feature}_prediction_range"
                ] = np.nan

        records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 7. Run one fold
# ---------------------------------------------------------------------------
def run_fold(
    df: pd.DataFrame,
    fold: dict,
    target_col: str = "target_shock_window",
    time_steps: int = TIME_STEPS,
    epochs: int = 50,
    batch_size: int = 32,
    seed: int = RANDOM_SEED,
    verbose: int = 0,
):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    test_start, test_end = fold["test_range"]
    val_start, val_end = fold["val_range"]

    test_df = df.loc[test_start:test_end]
    val_df = df.loc[val_start:val_end]

    train_segments = get_train_segments(
        df,
        exclude_ranges=[
            fold["test_range"],
            fold["val_range"],
        ],
    )

    if (
        not train_segments
        or len(test_df) <= time_steps
        or len(val_df) <= time_steps
    ):
        print(
            f"[fold {fold['fold_id']}] WARNING: "
            "not enough data -- skipping."
        )
        return None, None

    # Fit scaler ONLY on training observations.
    train_concat = pd.concat(train_segments)

    scaler = RobustScaler()
    scaler.fit(
        train_concat.drop(
            columns=[target_col]
        ).values
    )

    X_train, y_train, _ = build_sequences_from_segments(
        train_segments,
        scaler,
        target_col,
        time_steps,
    )

    X_val, y_val, val_dates = build_sequences_from_segments(
        [val_df],
        scaler,
        target_col,
        time_steps,
    )

    X_test, y_test, test_dates = build_sequences_from_segments(
        [test_df],
        scaler,
        target_col,
        time_steps,
    )

    if (
        len(y_train) == 0
        or y_train.sum() == 0
        or y_val.sum() == 0
        or y_test.sum() == 0
    ):
        print(
            f"[fold {fold['fold_id']}] WARNING: "
            f"zero positives "
            f"(train={y_train.sum():.0f}, "
            f"val={y_val.sum():.0f}, "
            f"test={y_test.sum():.0f}) -- skipping."
        )
        return None, None

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    class_weight_dict = {
        classes[0]: weights[0],
        classes[1]: weights[1],
    }

    model = build_lstm_model(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2],
        )
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc_pr",
        mode="max",
        patience=5,
        restore_best_weights=True,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc_pr",
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=[
            early_stop,
            reduce_lr,
        ],
        verbose=verbose,
    )

    # Validation threshold selection.
    val_probs = model.predict(
        X_val,
        verbose=0,
    ).ravel()

    precisions, recalls, thresholds = (
        precision_recall_curve(
            y_val,
            val_probs,
        )
    )

    f1s = (
        2 * (precisions * recalls)
        / (precisions + recalls + 1e-8)
    )

    best_idx = (
        int(np.argmax(f1s[:-1]))
        if len(thresholds) > 0
        else 0
    )

    best_threshold = (
        float(thresholds[best_idx])
        if len(thresholds) > 0
        else 0.5
    )

    # Test prediction.
    test_probs = model.predict(
        X_test,
        verbose=0,
    ).ravel()

    test_preds = (
        test_probs >= best_threshold
    ).astype(int)

    test_roc_auc = (
        roc_auc_score(y_test, test_probs)
        if len(np.unique(y_test)) > 1
        else np.nan
    )

    p, r, _ = precision_recall_curve(
        y_test,
        test_probs,
    )

    test_pr_auc = sk_auc(r, p)

    report = classification_report(
        y_test,
        test_preds,
        target_names=[
            "safe window(0)",
            "shock warning(1)",
        ],
        output_dict=True,
        zero_division=0,
    )

    positive_probs = test_probs[y_test == 1]
    negative_probs = test_probs[y_test == 0]

    mean_positive = (
        float(positive_probs.mean())
        if len(positive_probs)
        else np.nan
    )
    mean_negative = (
        float(negative_probs.mean())
        if len(negative_probs)
        else np.nan
    )

    probability_separation = (
        mean_positive - mean_negative
    )

    # AUC < 0.5 can indicate reversed ranking.
    ranking_direction = (
        "correct"
        if test_roc_auc >= 0.5
        else "reversed"
    )

    result = {
        "fold_id": fold["fold_id"],
        "test_episode_id": fold["test_episode_id"],
        "test_episode_start": fold["test_episode_start"],
        "test_episode_end": fold["test_episode_end"],
        "test_range_start": test_start,
        "test_range_end": test_end,
        "val_episode_id": fold["val_episode_id"],
        "val_episode_start": fold["val_episode_start"],
        "val_episode_end": fold["val_episode_end"],
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "n_test_positive": int(y_test.sum()),
        "threshold_used": best_threshold,
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
        "test_precision_shock": report[
            "shock warning(1)"
        ]["precision"],
        "test_recall_shock": report[
            "shock warning(1)"
        ]["recall"],
        "test_f1_shock": report[
            "shock warning(1)"
        ]["f1-score"],
        "mean_positive_probability": mean_positive,
        "mean_negative_probability": mean_negative,
        "probability_separation": probability_separation,
        "ranking_direction": ranking_direction,
    }

    # Keep predictions for episode-level diagnostics.
    predictions = pd.DataFrame(
        {
            "fold_id": fold["fold_id"],
            "episode_id": fold["test_episode_id"],
            "date": pd.to_datetime(test_dates),
            "actual": y_test.astype(int),
            "probability": test_probs,
            "prediction": test_preds,
        }
    )

    return result, predictions


# ---------------------------------------------------------------------------
# 8. Feature comparison
# ---------------------------------------------------------------------------
def compare_good_vs_poor(
    analysis_df: pd.DataFrame,
    good_threshold: float = 0.70,
    poor_threshold: float = 0.50,
):
    """
    Compare mean feature characteristics of good and poor episodes.

    Good: ROC-AUC >= 0.70
    Poor: ROC-AUC < 0.50
    Moderate: everything between them.
    """
    df = analysis_df.copy()

    df["performance_group"] = np.select(
        [
            df["test_roc_auc"] >= good_threshold,
            df["test_roc_auc"] < poor_threshold,
        ],
        [
            "good",
            "poor",
        ],
        default="moderate",
    )

    rows = []

    for feature in FEATURE_COLS:
        prediction_col = (
            f"{feature}_prediction_mean"
        )
        change_col = (
            f"{feature}_prediction_change"
        )

        if (
            prediction_col not in df.columns
            or change_col not in df.columns
        ):
            continue

        good_values = df.loc[
            df["performance_group"] == "good",
            prediction_col,
        ].dropna()

        poor_values = df.loc[
            df["performance_group"] == "poor",
            prediction_col,
        ].dropna()

        good_changes = df.loc[
            df["performance_group"] == "good",
            change_col,
        ].dropna()

        poor_changes = df.loc[
            df["performance_group"] == "poor",
            change_col,
        ].dropna()

        rows.append(
            {
                "feature": feature,
                "good_n": len(good_values),
                "poor_n": len(poor_values),
                "good_prediction_mean": (
                    good_values.mean()
                    if len(good_values)
                    else np.nan
                ),
                "poor_prediction_mean": (
                    poor_values.mean()
                    if len(poor_values)
                    else np.nan
                ),
                "prediction_mean_difference": (
                    good_values.mean()
                    - poor_values.mean()
                    if len(good_values)
                    and len(poor_values)
                    else np.nan
                ),
                "good_prediction_change_mean": (
                    good_changes.mean()
                    if len(good_changes)
                    else np.nan
                ),
                "poor_prediction_change_mean": (
                    poor_changes.mean()
                    if len(poor_changes)
                    else np.nan
                ),
                "prediction_change_difference": (
                    good_changes.mean()
                    - poor_changes.mean()
                    if len(good_changes)
                    and len(poor_changes)
                    else np.nan
                ),
            }
        )

    comparison = pd.DataFrame(rows)

    if len(comparison):
        comparison["abs_prediction_difference"] = (
            comparison[
                "prediction_mean_difference"
            ].abs()
        )

        comparison = comparison.sort_values(
            "abs_prediction_difference",
            ascending=False,
        )

    return df, comparison


# ---------------------------------------------------------------------------
# 9. Main LOEO experiment
# ---------------------------------------------------------------------------
def run_leave_one_episode_out(
    df: pd.DataFrame,
    target_col: str = "target_shock_window",
    buffer_days: int = BUFFER_DAYS,
    time_steps: int = TIME_STEPS,
    epochs: int = 50,
    max_folds: int = None,
    output_dir: str = "artifact/result/loeo_diagnostic",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    episodes = identify_shock_episodes(
        df,
        target_col=target_col,
    )

    print(
        f"Identified {len(episodes)} shock episodes:"
    )
    print(episodes)
    print()

    folds = build_loeo_folds(
        episodes,
        buffer_days=buffer_days,
    )

    if max_folds is not None:
        folds = folds[:max_folds]

    results = []
    prediction_frames = []

    for fold in folds:
        print(
            f"--- Fold {fold['fold_id']}: "
            f"held-out episode "
            f"{fold['test_episode_start'].date()} "
            f"to "
            f"{fold['test_episode_end'].date()} ---"
        )

        result, predictions = run_fold(
            df,
            fold,
            target_col=target_col,
            time_steps=time_steps,
            epochs=epochs,
        )

        if result is not None:
            results.append(result)
            prediction_frames.append(predictions)

            print(
                f"  ROC-AUC={result['test_roc_auc']:.3f} "
                f"PR-AUC={result['test_pr_auc']:.3f} "
                f"Precision={result['test_precision_shock']:.3f} "
                f"Recall={result['test_recall_shock']:.3f} "
                f"F1={result['test_f1_shock']:.3f} "
                f"PosProb={result['mean_positive_probability']:.4f} "
                f"NegProb={result['mean_negative_probability']:.4f}"
            )

        print()

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("No usable folds.")
        return results_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    # Episode characteristics.
    characteristics_df = (
        calculate_episode_characteristics(
            df,
            episodes,
        )
    )

    # Merge model performance with characteristics.
    analysis_df = results_df.merge(
        characteristics_df,
        left_on="test_episode_id",
        right_on="episode_id",
        how="left",
    )

    analysis_df, comparison_df = (
        compare_good_vs_poor(
            analysis_df
        )
    )

    # Save everything.
    episodes.to_csv(
        output_dir / "episodes.csv",
        index=False,
    )

    results_df.to_csv(
        output_dir / "loeo_metrics.csv",
        index=False,
    )

    predictions_df.to_csv(
        output_dir / "loeo_predictions.csv",
        index=False,
    )

    characteristics_df.to_csv(
        output_dir / "episode_characteristics.csv",
        index=False,
    )

    analysis_df.to_csv(
        output_dir / "episode_analysis_master.csv",
        index=False,
    )

    comparison_df.to_csv(
        output_dir / "good_vs_poor_feature_comparison.csv",
        index=False,
    )

    # Summary.
    summary = {
        "n_folds": int(len(results_df)),
        "mean_roc_auc": float(
            results_df["test_roc_auc"].mean()
        ),
        "std_roc_auc": float(
            results_df["test_roc_auc"].std()
        ),
        "min_roc_auc": float(
            results_df["test_roc_auc"].min()
        ),
        "max_roc_auc": float(
            results_df["test_roc_auc"].max()
        ),
        "mean_pr_auc": float(
            results_df["test_pr_auc"].mean()
        ),
        "std_pr_auc": float(
            results_df["test_pr_auc"].std()
        ),
    }

    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            default=str,
        )

    print("=" * 80)
    print(
        f"LOEO SUMMARY "
        f"({len(results_df)} usable folds)"
    )
    print("=" * 80)

    print(
        results_df[
            [
                "test_episode_id",
                "test_episode_start",
                "test_episode_end",
                "test_roc_auc",
                "test_pr_auc",
                "test_precision_shock",
                "test_recall_shock",
                "test_f1_shock",
                "mean_positive_probability",
                "mean_negative_probability",
                "probability_separation",
                "ranking_direction",
            ]
        ].sort_values(
            "test_roc_auc"
        ).to_string(index=False)
    )

    print()
    print("GOOD / MODERATE / POOR EPISODES")
    print(
        analysis_df[
            [
                "episode_id",
                "episode_start",
                "episode_end",
                "episode_duration",
                "test_roc_auc",
                "test_pr_auc",
                "performance_group",
            ]
        ].sort_values(
            "test_roc_auc"
        ).to_string(index=False)
    )

    print()
    print("TOP FEATURE DIFFERENCES: GOOD vs POOR")
    print(
        comparison_df[
            [
                "feature",
                "good_prediction_mean",
                "poor_prediction_mean",
                "prediction_mean_difference",
                "good_prediction_change_mean",
                "poor_prediction_change_mean",
                "prediction_change_difference",
            ]
        ].head(10).to_string(index=False)
    )

    return (
        results_df,
        predictions_df,
        analysis_df,
        comparison_df,
    )


# ---------------------------------------------------------------------------
# 10. Plotting
# ---------------------------------------------------------------------------
def create_diagnostic_plots(
    analysis_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    output_dir: str = "artifact/result/loeo_diagnostic",
):
    """
    Create simple diagnostic plots.

    Matplotlib is imported here so the core LOEO run does not require it.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. AUC by episode.
    ordered = analysis_df.sort_values(
        "test_roc_auc"
    )

    plt.figure(figsize=(12, 6))
    plt.bar(
        ordered["episode_id"].astype(str),
        ordered["test_roc_auc"],
    )
    plt.axhline(
        0.5,
        linestyle="--",
    )
    plt.xlabel("Episode ID")
    plt.ylabel("ROC-AUC")
    plt.title("LOEO ROC-AUC by Held-Out Episode")
    plt.tight_layout()
    plt.savefig(
        output_dir / "01_auc_by_episode.png",
        dpi=150,
    )
    plt.close()

    # 2. Episode duration vs AUC.
    plt.figure(figsize=(8, 6))
    plt.scatter(
        analysis_df["episode_duration"],
        analysis_df["test_roc_auc"],
    )
    plt.axhline(
        0.5,
        linestyle="--",
    )
    plt.xlabel("Episode duration (days)")
    plt.ylabel("ROC-AUC")
    plt.title(
        "Episode Duration vs LOEO ROC-AUC"
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "02_duration_vs_auc.png",
        dpi=150,
    )
    plt.close()

    # 3. Probability separation.
    ordered = analysis_df.sort_values(
        "probability_separation"
    )

    plt.figure(figsize=(12, 6))
    plt.bar(
        ordered["episode_id"].astype(str),
        ordered["probability_separation"],
    )
    plt.axhline(
        0,
        linestyle="--",
    )
    plt.xlabel("Episode ID")
    plt.ylabel(
        "Mean positive probability - "
        "mean negative probability"
    )
    plt.title(
        "Prediction Probability Separation by Episode"
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "03_probability_separation.png",
        dpi=150,
    )
    plt.close()

    # 4. Top feature differences.
    top = comparison_df.head(10).copy()

    if len(top):
        top = top.sort_values(
            "prediction_mean_difference"
        )

        plt.figure(figsize=(10, 6))
        plt.barh(
            top["feature"],
            top["prediction_mean_difference"],
        )
        plt.axvline(
            0,
            linestyle="--",
        )
        plt.xlabel(
            "Good episode mean - poor episode mean"
        )
        plt.ylabel("Feature")
        plt.title(
            "Largest Feature Differences: "
            "Good vs Poor Episodes"
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / "04_good_vs_poor_features.png",
            dpi=150,
        )
        plt.close()

    # 5. Prediction probability distributions by episode.
    if len(predictions_df):
        for episode_id, group in predictions_df.groupby(
            "episode_id"
        ):
            plt.figure(figsize=(10, 5))
            plt.hist(
                group.loc[
                    group["actual"] == 0,
                    "probability",
                ],
                bins=30,
                alpha=0.6,
                label="Actual safe",
            )
            plt.hist(
                group.loc[
                    group["actual"] == 1,
                    "probability",
                ],
                bins=30,
                alpha=0.6,
                label="Actual shock",
            )
            plt.xlabel("Predicted probability")
            plt.ylabel("Count")
            plt.title(
                f"Episode {episode_id}: "
                "Probability Distributions"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                output_dir
                / f"05_probability_distribution_episode_{episode_id}.png",
                dpi=150,
            )
            plt.close()


# ---------------------------------------------------------------------------
# 11. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    file_path = get_path(USDGH_DATASET_FILE)

    df = pd.read_csv(
        file_path,
        parse_dates=["date"],
        index_col="date",
    )

    (
        results_df,
        predictions_df,
        analysis_df,
        comparison_df,
    ) = run_leave_one_episode_out(
        df,
        buffer_days=BUFFER_DAYS,
        time_steps=TIME_STEPS,
        epochs=50,
    )

    if len(analysis_df):
        create_diagnostic_plots(
            analysis_df,
            predictions_df,
            comparison_df,
        )
