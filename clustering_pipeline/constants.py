from pathlib import Path

RANDOM_STATE = 42
DEFAULT_N_CLUSTERS = 4
SKEWNESS_THRESHOLD = 1.0

REQUIRED_SOURCE_COLUMNS = [
    "foreground_app_duration_sum",
    "foreground_app_switch_per_hour",
    "concentration_ratio",
]

DERIVED_FEATURE_COLUMNS = ["force_intensity", "switch_frequency"]

DROP_COLUMNS = [
    "device_id",
    "screen_on_timestamp_unix",
    "screen_on_timestamp_dt",
    "screen_off_timestamp_unix",
    "screen_off_timestamp_dt",
    #############  default  ###############
    "screen_duration",
    "foreground_app_count",
    "foreground_app_switch_count",
    "foreground_app_duration_mean",
    "unique_foreground_app_duration_max",
    "foreground_apps_and_durations",
    "unique_foreground_apps_and_durations",
    "unique_foreground_app_count",
    "concentration_ratio",
]



"""
        # "screen_duration",
        # "unique_foreground_app_count", # 2. 목적성
        # "foreground_app_count",
        # "foreground_app_switch_count",
        "foreground_app_switch_per_hour", # 3. 주의력 분산
        "foreground_app_duration_sum", # 1. 절대적 사용량(screen_duration과 비슷하지만 파생변수 사용)
        # "foreground_app_duration_mean",
        # "unique_foreground_app_duration_max",
        "concentration_ratio" # 4. 몰입 소비형 분리용 변수
        # "foreground_apps_and_durations",
        # "unique_foreground_apps_and_durations"

"""