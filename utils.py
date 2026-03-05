
import logging
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import Pool
import config


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(config.LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_categorical_features(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col.startswith(config.CAT_FEATURE_PREFIX)]


def get_target_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col.startswith(config.TARGET_PREFIX)]


def get_predict_columns(target_columns: List[str]) -> List[str]:
    return [col.replace(config.TARGET_PREFIX, config.PREDICT_PREFIX) for col in target_columns]


def load_data(train_path: Path, target_path: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:

    try:
        logger.info(f"Загружаем данные из {train_path}")
        X = pd.read_parquet(train_path)

        logger.info(f"Загружаем целевые переменные из {target_path}")
        y = pd.read_parquet(target_path)

        logger.info(f"Размер X: {X.shape}, размер y: {y.shape}")
        return X, y
    except FileNotFoundError as e:
        logger.error(f"Файл не найден: {e}")
        raise


def prepare_categorical_features(
    X: pd.DataFrame,
    test: pd.DataFrame,
    cat_feature_names: List[str],
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    logger.info(f"Подготовка {len(cat_feature_names)} категориальных признаков")

    X = X.copy()
    test = test.copy()

    for col in cat_feature_names:
        X[col] = X[col].fillna(config.MISSING_VALUE).astype(str)
        test[col] = test[col].fillna(config.MISSING_VALUE).astype(str)

    for col in cat_feature_names:
        freq = X[col].value_counts(dropna=False)
        freq_col_name = f"{col}__freq"
        X[freq_col_name] = X[col].map(freq).fillna(0).astype("int32")
        test[freq_col_name] = test[col].map(freq).fillna(0).astype("int32")

    logger.info(f"Категориальные признаки подготовлены. Новая размер X: {X.shape}")
    return X, test


def create_pool(
    data: pd.DataFrame,
    target: Optional[pd.DataFrame] = None,
    cat_features: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> Pool:
    if logger:
        logger.info(f"Создаём Pool. Размер данных: {data.shape}")

    data_features = data.drop("customer_id", axis=1, errors='ignore')

    if target is not None:
        target_values = target.drop("customer_id", axis=1, errors='ignore')
        return Pool(
            data=data_features,
            label=target_values,
            cat_features=cat_features or []
        )
    else:
        return Pool(
            data=data_features,
            cat_features=cat_features or []
        )


def merge_test_data(
    test_main: pd.DataFrame,
    test_extra: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    logger.info(f"Объединяем тестовые данные. main: {test_main.shape}, extra: {test_extra.shape}")
    test = pd.merge(test_main, test_extra, on="customer_id", how="left")
    logger.info(f"Объединённые данные: {test.shape}")
    return test


def align_test_with_train(
    test: pd.DataFrame,
    train_columns: List[str],
    logger: logging.Logger
) -> pd.DataFrame:
    logger.info(f"Выравниваем признаки. Тренировка: {len(train_columns)}, тесты: {len(test.columns)}")
    test = test[train_columns]
    logger.info(f"После выравнивания: {test.shape}")
    return test


def save_model(model, path: Path, logger: logging.Logger) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(path))
        logger.info(f"Модель сохранена: {path}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении модели: {e}")
        raise


def save_predictions(
    predictions: pd.DataFrame,
    path: Path,
    logger: logging.Logger
) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(path, index=False)
        logger.info(f"Предсказания сохранены: {path}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении предсказаний: {e}")
        raise


def load_model(model_path: Path, logger: logging.Logger):
    try:
        from catboost import CatBoostClassifier
        logger.info(f"Загружаем модель из {model_path}")
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        logger.info("Модель успешно загружена")
        return model
    except FileNotFoundError:
        logger.error(f"Модель не найдена: {model_path}")
        raise
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        raise
