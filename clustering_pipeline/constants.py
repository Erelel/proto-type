from datetime import datetime
from pathlib import Path

RANDOM_STATE = 42
DEFAULT_N_CLUSTERS = 4
SKEWNESS_THRESHOLD = 1.0

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent
DEFAULT_RESULT_DIR = PROJECT_ROOT / "result"
RUN_DATE_TAG = datetime.now().strftime("%m%d")
DEFAULT_IMAGE_DIR = DEFAULT_RESULT_DIR / f"{RUN_DATE_TAG}_image"
DEFAULT_CSV_DIR = DEFAULT_RESULT_DIR / f"{RUN_DATE_TAG}_csv"
DEFAULT_MAIN_CSV = PROJECT_ROOT / "mendeley_v5.csv"

MINMAX_FEATURE_RANGE = (0.0, 1.0)


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path

REQUIRED_SOURCE_COLUMNS = [
    "foreground_app_duration_sum",
    "foreground_app_switch_per_hour",
    "concentration_ratio",
]

DERIVED_FEATURE_COLUMNS = ["focus_intensity", "switch_frequency"]

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