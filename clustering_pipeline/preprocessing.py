import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from constants import (
    DERIVED_FEATURE_COLUMNS,
    REQUIRED_SOURCE_COLUMNS,
    SKEWNESS_THRESHOLD,
)

SKEW_ANALYSIS_COLUMNS = [
    "foreground_app_duration_sum",
    "foreground_app_switch_per_hour",
]


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing_columns = [col for col in REQUIRED_SOURCE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing columns required for derived features: {missing_columns}"
        )


def _get_skew_direction(skew_value: float) -> str:
    if skew_value > 0: return "right"
    if skew_value < 0: return "left"
    return "balanced"


def _build_derived_features(
    source_df: pd.DataFrame,
    use_log_transform: bool,
) -> pd.DataFrame:
    duration_base = source_df["foreground_app_duration_sum"]
    switch_base = source_df["foreground_app_switch_per_hour"]

    if use_log_transform:
        duration_base = np.log1p(duration_base)
        switch_base = np.log1p(switch_base)

    derived = pd.DataFrame(index=source_df.index)
    derived["force_intensity"] = duration_base * source_df["concentration_ratio"]
    derived["switch_frequency"] = switch_base * (1 - source_df["concentration_ratio"])
    return derived


def load_and_preprocess(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    initial_rows = len(df)

    _validate_required_columns(df)

    source_df = df[REQUIRED_SOURCE_COLUMNS].copy()
    source_df = source_df.dropna(axis=0, how="any")

    source_skewness = {
        col: float(source_df[col].skew()) for col in SKEW_ANALYSIS_COLUMNS
    }
    skew_direction = {
        col: _get_skew_direction(source_skewness[col]) for col in SKEW_ANALYSIS_COLUMNS
    }
    right_skew_flags = {
        col: source_skewness[col] > SKEWNESS_THRESHOLD for col in SKEW_ANALYSIS_COLUMNS
    }

    non_negative_for_log = all(source_df[col].min() >= 0 for col in SKEW_ANALYSIS_COLUMNS)
    use_log_transform = all(right_skew_flags.values()) and non_negative_for_log

    log_transform_block_reason = None
    if not all(right_skew_flags.values()):
        log_transform_block_reason = (
            f"At least one source variable is not strongly right-skewed "
            f"(threshold={SKEWNESS_THRESHOLD})."
        )
    elif not non_negative_for_log:
        log_transform_block_reason = (
            "At least one source variable contains values below 0, "
            "so log1p would be invalid or unstable."
        )

    X = _build_derived_features(source_df, use_log_transform)

    metadata = {
        "initial_rows": initial_rows,
        "final_rows": len(source_df),
        "dropped_nan_rows": initial_rows - len(source_df),
        "used_columns": DERIVED_FEATURE_COLUMNS,
        "source_skewness": source_skewness,
        "skew_direction": skew_direction,
        "right_skew_flags": right_skew_flags,
        "log_transform_applied": use_log_transform,
        "log_transform_block_reason": log_transform_block_reason,
    }
    return X, source_df, metadata


def build_raw_pipeline(
    X: pd.DataFrame,
    scaler_cls: type = MinMaxScaler,
) -> tuple[np.ndarray, object]:
    scaler = scaler_cls()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def build_standard_pipeline(
    X: pd.DataFrame,
    scaler_cls: type = StandardScaler,
) -> tuple[np.ndarray, object]:
    scaler = scaler_cls()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler