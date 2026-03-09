import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
MODIFIED_DATA_DIR = PROJECT_DIR / "modified_data"
MODELS_DIR = PROJECT_DIR / "models"
SUBMITS_DIR = PROJECT_DIR / "submits"
CATBOOST_INFO_DIR = PROJECT_DIR / "catboost_info"

TRAIN_MAIN_PATH = DATA_DIR / "train_main_features.parquet"
TRAIN_EXTRA_PATH = DATA_DIR / "train_extra_features.parquet"
TRAIN_TARGET_PATH = DATA_DIR / "train_target.parquet"
TEST_MAIN_PATH = DATA_DIR / "test_main_features.parquet"
TEST_EXTRA_PATH = DATA_DIR / "test_extra_features.parquet"
TRAIN_PROCESSED_PATH = MODIFIED_DATA_DIR / "train_no_nan_min_var.parquet"
SAMPLE_SUBMIT_PATH = SUBMITS_DIR / "sample_submit.parquet"

TARGET_PREFIX = "target_"
PREDICT_PREFIX = "predict_"
CAT_FEATURE_PREFIX = "cat_feature"
MISSING_VALUE = "__MISSING__"

LOSS_FUNCTION = "MultiLogloss"
ITERATIONS = 4000
OD_TYPE = "Iter"
OD_WAIT = 200
USE_BEST_MODEL = True
VERBOSE_PERIOD = 200
RANDOM_SEED = 42
TASK_TYPE = "GPU"
DEVICES = "0"
ALLOW_WRITING_FILES = False

LEARNING_RATE_MIN = 3e-6
LEARNING_RATE_MAX = 0.1
DEPTH_MIN = 4
DEPTH_MAX = 12
L2_MIN = 4.0
L2_MAX = 50.0
RANDOM_STRENGTH_MIN = 0.0
RANDOM_STRENGTH_MAX = 5.0
BAGGING_TEMPERATURE_MIN = 0.0
BAGGING_TEMPERATURE_MAX = 2.0

OPTUNA_TRIALS = 20
OPTUNA_DIRECTION = "minimize"

VAL_TEST_SIZE = 0.2
VAL_RANDOM_STATE = 42

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
