"""
Leave-one-episode-out (LOEO) validation for the cedi shock LSTM model.

Rationale: shock events are rare and highly clustered (11 distinct episodes
across the full 2010-2025 dataset). A single calendar-based train/val/test
split can, by chance, contain zero or one holdout episode, making any single
test score unreliable and highly sensitive to which episode happened to fall
in the test window. This script instead rotates the held-out test episode
across all identified episodes, retrains a fresh model per fold, and
aggregates metrics across folds for a much more honest estimate of
generalization -- and shows which specific episodes the model struggles with.

Expects a DataFrame with a DatetimeIndex and a 'target_shock_window' column
(0/1), plus feature columns -- i.e. the output of your feature_engineer_data
pipeline.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, auc as sk_auc
)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense

from src.config.config import USDGH_DATASET_FILE
from src.utils.utils import get_path


# ---------------------------------------------------------------------------
# 1. Episode identification
# ---------------------------------------------------------------------------
def identify_shock_episodes(df: pd.DataFrame, target_col: str = 'target_shock_window',
                             gap_days: int = 14) -> pd.DataFrame:
    """
    Groups consecutive shock-flagged dates into distinct episodes.
    Dates flagged 1 that are within `gap_days` of each other are treated
    as the same episode; a gap larger than that starts a new episode.
    """
    pos_dates = df.index[df[target_col] == 1].sort_values()
    if len(pos_dates) == 0:
        raise ValueError("No positive shock windows found in target column.")

    gaps = pos_dates.to_series().diff().dt.days
    episode_id = (gaps > gap_days).cumsum()
    episodes = (
        pos_dates.to_series()
        .groupby(episode_id)
        .agg(start='min', end='max', n_days='count')
        .reset_index(drop=True)
    )
    return episodes


# ---------------------------------------------------------------------------
# 2. Building per-fold train / val / test date ranges
# ---------------------------------------------------------------------------
def build_loeo_folds(episodes: pd.DataFrame, buffer_days: int = 45):
    """
    For each episode, builds a fold definition:
      - test_range: [episode.start - buffer, episode.end + buffer]
      - val_range: the chronologically nearest *other* episode (+ buffer),
        used for early stopping / threshold selection, also excluded from training
      - train: everything else

    Returns a list of dicts, one per fold.
    """
    folds = []
    n = len(episodes)
    for i in range(n):
        test_start = episodes.loc[i, 'start'] - pd.Timedelta(days=buffer_days)
        test_end = episodes.loc[i, 'end'] + pd.Timedelta(days=buffer_days)

        other_idx = [j for j in range(n) if j != i]
        distances = [abs((episodes.loc[j, 'start'] - episodes.loc[i, 'start']).days)
                     for j in other_idx]
        val_idx = other_idx[int(np.argmin(distances))]
        val_start = episodes.loc[val_idx, 'start'] - pd.Timedelta(days=buffer_days)
        val_end = episodes.loc[val_idx, 'end'] + pd.Timedelta(days=buffer_days)

        folds.append({
            'fold_id': i,
            'test_episode_start': episodes.loc[i, 'start'],
            'test_episode_end': episodes.loc[i, 'end'],
            'test_range': (test_start, test_end),
            'val_episode_idx': val_idx,
            'val_range': (val_start, val_end),
        })
    return folds


# ---------------------------------------------------------------------------
# 3. Segmenting the dataframe into contiguous train chunks
#    (excluding the held-out test and validation windows)
# ---------------------------------------------------------------------------
def get_train_segments(df: pd.DataFrame, exclude_ranges: list):
    """
    Returns a list of contiguous dataframe segments covering everything in
    `df` EXCEPT the given exclude_ranges (list of (start, end) tuples).
    Sequences are built separately per segment later, so a gap never gets
    silently bridged into a false "continuous" window.
    """
    exclude_ranges = sorted(exclude_ranges, key=lambda r: r[0])
    segments = []
    cursor = df.index.min()

    for (ex_start, ex_end) in exclude_ranges:
        if cursor < ex_start:
            seg = df.loc[cursor:ex_start - pd.Timedelta(days=1)]
            if len(seg) > 0:
                segments.append(seg)
        cursor = max(cursor, ex_end + pd.Timedelta(days=1))

    tail = df.loc[cursor:]
    if len(tail) > 0:
        segments.append(tail)

    return [s for s in segments if len(s) > 0]


# ---------------------------------------------------------------------------
# 4. Sequence building per segment, then concatenation
# ---------------------------------------------------------------------------
def create_3d_sequence(scaled_feature, target, dates, time_steps=30):
    assert len(scaled_feature) == len(target) == len(dates), (
        f"Length mismatch: features={len(scaled_feature)}, "
        f"target={len(target)}, dates={len(dates)}"
    )
    if len(scaled_feature) <= time_steps:
        return (np.empty((0, time_steps, scaled_feature.shape[1])),
                np.empty((0,)), np.empty((0,), dtype='datetime64[ns]'))

    feature_3d, target_3d, sequence_dates = [], [], []
    for i in range(len(scaled_feature) - time_steps):
        feature_3d.append(scaled_feature[i:i + time_steps])
        target_3d.append(target[i + time_steps])
        sequence_dates.append(dates[i + time_steps])

    return np.array(feature_3d), np.array(target_3d), np.array(sequence_dates)


def build_sequences_from_segments(segments: list, scaler, target_col: str, time_steps: int = 30):
    """
    Scales and sequence-builds each contiguous segment separately, then
    concatenates. Building one sequence across a gap would silently stitch
    unrelated time periods into a single false "continuous" window -- this
    keeps each segment's windows honest.
    """
    all_X, all_y, all_dates = [], [], []
    n_features = None
    for seg in segments:
        target_raw = seg[target_col].values
        feature_raw = seg.drop(columns=[target_col]).values
        n_features = feature_raw.shape[1]
        feature_scaled = scaler.transform(feature_raw)

        X, y, d = create_3d_sequence(feature_scaled, target_raw, seg.index.values, time_steps=time_steps)
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            all_dates.append(d)

    if not all_X:
        n_features = n_features or 0
        return (np.empty((0, time_steps, n_features)), np.empty((0,)), np.empty((0,)))

    return np.concatenate(all_X), np.concatenate(all_y), np.concatenate(all_dates)


# ---------------------------------------------------------------------------
# 5. Model builder (mirrors the existing train_lstm architecture)
# ---------------------------------------------------------------------------
def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(units=64, return_sequences=True)),
        Dropout(0.2),
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(units=16, activation='relu'),
        Dense(units=1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(curve='PR', name='auc_pr'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# 6. One fold: train, evaluate, return metrics
# ---------------------------------------------------------------------------
def run_fold(df: pd.DataFrame, fold: dict, target_col: str = 'target_shock_window',
             time_steps: int = 30, epochs: int = 50, batch_size: int = 32, seed: int = 42,
             verbose: int = 0):

    tf.random.set_seed(seed)
    np.random.seed(seed)

    test_start, test_end = fold['test_range']
    val_start, val_end = fold['val_range']

    test_df = df.loc[test_start:test_end]
    val_df = df.loc[val_start:val_end]
    train_segments = get_train_segments(df, exclude_ranges=[fold['test_range'], fold['val_range']])

    if not train_segments or len(test_df) <= time_steps or len(val_df) <= time_steps:
        print(f"[fold {fold['fold_id']}] WARNING: not enough data to build sequences -- skipping.")
        return None

    # scaler fit on TRAINING data only
    train_concat = pd.concat(train_segments)
    scaler = RobustScaler()
    scaler.fit(train_concat.drop(columns=[target_col]).values)

    X_train, y_train, _ = build_sequences_from_segments(train_segments, scaler, target_col, time_steps)
    X_val, y_val, _ = build_sequences_from_segments([val_df], scaler, target_col, time_steps)
    X_test, y_test, test_dates = build_sequences_from_segments([test_df], scaler, target_col, time_steps)

    if len(y_train) == 0 or y_train.sum() == 0 or y_val.sum() == 0 or y_test.sum() == 0:
        print(f"[fold {fold['fold_id']}] WARNING: a split has zero positives "
              f"(train={y_train.sum():.0f}, val={y_val.sum():.0f}, test={y_test.sum():.0f}) "
              f"-- skipping this fold.")
        return None

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = {classes[0]: weights[0], classes[1]: weights[1]}

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc_pr', mode='max', patience=5, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc_pr', mode='max', factor=0.5, patience=2, min_lr=1e-5
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=verbose,
    )

    # threshold picked on VALIDATION, applied to TEST -- never derive the
    # threshold from the fold's own test data
    val_probs = model.predict(X_val, verbose=0).ravel()
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1s[:-1])) if len(thresholds) > 0 else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5

    test_probs = model.predict(X_test, verbose=0).ravel()
    test_preds = (test_probs >= best_threshold).astype(int)

    test_roc_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else np.nan
    p, r, _ = precision_recall_curve(y_test, test_probs)
    test_pr_auc = sk_auc(r, p)

    report = classification_report(
        y_test, test_preds, target_names=['safe window(0)', 'shock warning(1)'],
        output_dict=True, zero_division=0
    )

    return {
        'fold_id': fold['fold_id'],
        'test_episode_start': fold['test_episode_start'],
        'test_episode_end': fold['test_episode_end'],
        'n_train': len(y_train),
        'n_val': len(y_val),
        'n_test': len(y_test),
        'n_test_positive': int(y_test.sum()),
        'threshold_used': best_threshold,
        'test_roc_auc': test_roc_auc,
        'test_pr_auc': test_pr_auc,
        'test_precision_shock': report['shock warning(1)']['precision'],
        'test_recall_shock': report['shock warning(1)']['recall'],
        'test_f1_shock': report['shock warning(1)']['f1-score'],
    }


# ---------------------------------------------------------------------------
# 7. Full LOEO run across all episodes
# ---------------------------------------------------------------------------
def run_leave_one_episode_out(df: pd.DataFrame, target_col: str = 'target_shock_window',
                               buffer_days: int = 45, time_steps: int = 30,
                               epochs: int = 50, max_folds: int = None):
    episodes = identify_shock_episodes(df, target_col=target_col)
    print(f"Identified {len(episodes)} shock episodes:")
    print(episodes)
    print()

    folds = build_loeo_folds(episodes, buffer_days=buffer_days)
    if max_folds is not None:
        folds = folds[:max_folds]

    results = []
    for fold in folds:
        print(f"--- Fold {fold['fold_id']}: held-out episode "
              f"{fold['test_episode_start'].date()} to {fold['test_episode_end'].date()} ---")
        r = run_fold(df, fold, target_col=target_col, time_steps=time_steps, epochs=epochs)
        if r is not None:
            results.append(r)
            print(f"  ROC-AUC={r['test_roc_auc']:.3f}  PR-AUC={r['test_pr_auc']:.3f}  "
                  f"Precision={r['test_precision_shock']:.3f}  Recall={r['test_recall_shock']:.3f}  "
                  f"F1={r['test_f1_shock']:.3f}  (n_test_pos={r['n_test_positive']})")
        print()

    results_df = pd.DataFrame(results)
    if len(results_df) == 0:
        print("No usable folds -- every fold had a split with zero positives. "
              "Try a larger buffer_days or check episode identification.")
        return results_df

    print("=" * 70)
    print(f"LOEO SUMMARY (across {len(results_df)} usable folds out of {len(folds)} total)")
    print("=" * 70)
    summary_cols = ['test_roc_auc', 'test_pr_auc', 'test_precision_shock',
                     'test_recall_shock', 'test_f1_shock']
    print(results_df[summary_cols].agg(['mean', 'std', 'min', 'max']))

    return results_df


if __name__ == '__main__':
    # df must have a DatetimeIndex and a 'target_shock_window' column,
    # e.g. the output of your feature_engineer_data pipeline
    file_path = get_path( USDGH_DATASET_FILE )
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    results_df = run_leave_one_episode_out(df)
    results_df.to_csv('artifact/result/loeo_results.csv', index=False)