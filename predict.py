import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
import utils


logger = utils.setup_logger(__name__)


class ModelPredictor:

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.X_train: Optional[pd.DataFrame] = None
        self.test: Optional[pd.DataFrame] = None
        self.cat_features: list = []
        self.predictions: Optional[np.ndarray] = None

    def load_model(self) -> None:
        logger.info("ЗАГРУЗКА МОДЕЛИ")

        self.model = utils.load_model(self.model_path, logger)

    def load_and_prepare_data(self) -> None:
        logger.info("ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")

        self.X_train = pd.read_parquet(config.TRAIN_PROCESSED_PATH)
        logger.info(f"Тренировочные данные загружены: {self.X_train.shape}")

        test_main = pd.read_parquet(config.TEST_MAIN_PATH)
        test_extra = pd.read_parquet(config.TEST_EXTRA_PATH)
        logger.info(f"Основные признаки теста: {test_main.shape}")
        logger.info(f"Дополнительные признаки теста: {test_extra.shape}")

        self.test = utils.merge_test_data(test_main, test_extra, logger)

        self.cat_features = utils.get_categorical_features(self.X_train)
        logger.info(f"Категориальных признаков: {len(self.cat_features)}")

        self.X_train, self.test = utils.prepare_categorical_features(
            self.X_train, self.test, self.cat_features, logger
        )

        self.test = utils.align_test_with_train(
            self.test, self.X_train.columns.tolist(), logger
        )

    def make_predictions(self) -> None:
        logger.info("ГЕНЕРАЦИЯ ПРЕДСКАЗАНИЙ")

        if self.model is None:
            logger.error("Модель не загружена!")
            raise RuntimeError("Модель не загружена!")

        test_pool = utils.create_pool(
            self.test, target=None, cat_features=self.cat_features, logger=logger
        )

        logger.info("Генерируем предсказания...")
        self.predictions = self.model.predict(test_pool, prediction_type="RawFormulaVal")
        logger.info(f"Получено предсказаний: {self.predictions.shape}")

    def save_submission(self, output_path: Optional[Path] = None) -> Path:

        logger.info("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")

        if self.predictions is None:
            logger.error("Предсказания не сгенерированы!")
            raise RuntimeError("Предсказания не сгенерированы!")

        sample_submit = pd.read_parquet(config.SAMPLE_SUBMIT_PATH)
        logger.info(f"Sample submission загружен: {sample_submit.shape}")

        target_columns = utils.get_target_columns(sample_submit)
        predict_columns = utils.get_predict_columns(target_columns)
        logger.info(f"Целевых столбцов: {len(predict_columns)}")

        result_df = sample_submit.copy()
        result_df[predict_columns] = self.predictions
        result_df["customer_id"] = result_df["customer_id"].astype("int32")

        logger.info(f"Форма результата: {result_df.shape}")

        if output_path is None:
            timestamp = time.time()
            output_path = config.SUBMITS_DIR / f"submission_{timestamp:.0f}.parquet"

        utils.save_predictions(result_df, output_path, logger)

        return output_path

    def predict(self, model_path: Optional[Path] = None, output_path: Optional[Path] = None) -> Path:

        try:
            if model_path is not None:
                self.model_path = model_path

            self.load_model()
            self.load_and_prepare_data()
            self.make_predictions()
            result_path = self.save_submission(output_path)

            logger.info("ПРЕДСКАЗАНИЯ ЗАВЕРШЕНЫ УСПЕШНО")

            return result_path

        except Exception as e:
            logger.error(f"Критическая ошибка при генерации предсказаний: {e}", exc_info=True)
            raise


def get_latest_model_path() -> Path:
    model_dir = config.MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    model_files = list(model_dir.glob("model_*.cbm"))

    if not model_files:
        model_files = list(model_dir.glob("*.cbm"))

    if not model_files:
        raise FileNotFoundError(f"Модели не найдены в {model_dir}")

    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Найдена последняя модель: {latest_model}")

    return latest_model


def main():
    logger.info("Начало генерации предсказаний")
    logger.info(f"Дата/время: {datetime.now().isoformat()}")

    try:
        model_path = get_latest_model_path()
    except FileNotFoundError as e:
        logger.error(f"Ошибка: {e}")
        logger.error("Пожалуйста, сначала обучите модель, запустив train.py")
        raise

    predictor = ModelPredictor(model_path)
    result_path = predictor.predict()

    logger.info(f"Результаты сохранены: {result_path}")


if __name__ == "__main__":
    main()
