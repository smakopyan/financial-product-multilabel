import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.integration import CatBoostPruningCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

import config
import utils


logger = utils.setup_logger(__name__)


class ModelTrainer:

    def __init__(self):
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.DataFrame] = None
        self.test: Optional[pd.DataFrame] = None
        self.cat_features: list = []
        self.train_pool = None
        self.val_pool = None
        self.model: Optional[CatBoostClassifier] = None
        self.study: Optional[optuna.Study] = None

    def load_data(self) -> None:
        logger.info("=" * 60)
        logger.info("ЗАГРУЗКА ДАННЫХ")
        logger.info("=" * 60)

        self.X, self.y = utils.load_data(
            config.TRAIN_PROCESSED_PATH,
            config.TRAIN_TARGET_PATH,
            logger
        )

        test_main = pd.read_parquet(config.TEST_MAIN_PATH)
        test_extra = pd.read_parquet(config.TEST_EXTRA_PATH)
        self.test = utils.merge_test_data(test_main, test_extra, logger)

        logger.info(f"Размер X: {self.X.shape}")
        logger.info(f"Размер y: {self.y.shape}")
        logger.info(f"Размер test: {self.test.shape}")

    def prepare_features(self) -> None:
        logger.info("=" * 60)
        logger.info("ПОДГОТОВКА ПРИЗНАКОВ")
        logger.info("=" * 60)

        self.cat_features = utils.get_categorical_features(self.X)
        logger.info(f"Найдено {len(self.cat_features)} категориальных признаков")

        self.X, self.test = utils.prepare_categorical_features(
            self.X, self.test, self.cat_features, logger
        )

        self.test = utils.align_test_with_train(
            self.test, self.X.columns.tolist(), logger
        )

    def split_data(self) -> None:
        logger.info("РАЗДЕЛЕНИЕ ДАННЫХ")

        train, val, target, val_target = train_test_split(
            self.X, self.y,
            test_size=config.VAL_TEST_SIZE,
            random_state=config.VAL_RANDOM_STATE
        )

        logger.info(f"Тренировка: {train.shape}, валидация: {val.shape}")

        self.train_pool = utils.create_pool(
            train, target, self.cat_features, logger
        )
        self.val_pool = utils.create_pool(
            val, val_target, self.cat_features, logger
        )

    def get_model_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "loss_function": config.LOSS_FUNCTION,
            "iterations": config.ITERATIONS,
            "learning_rate": trial.suggest_float(
                "lr", config.LEARNING_RATE_MIN, config.LEARNING_RATE_MAX, log=True
            ),
            "depth": trial.suggest_int("depth", config.DEPTH_MIN, config.DEPTH_MAX),
            "l2_leaf_reg": trial.suggest_float(
                "l2", config.L2_MIN, config.L2_MAX, log=True
            ),
            "random_strength": trial.suggest_float(
                "random_strength", config.RANDOM_STRENGTH_MIN, config.RANDOM_STRENGTH_MAX
            ),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", config.BAGGING_TEMPERATURE_MIN, config.BAGGING_TEMPERATURE_MAX
            ),
            "od_type": config.OD_TYPE,
            "od_wait": config.OD_WAIT,
            "use_best_model": config.USE_BEST_MODEL,
            "random_seed": config.RANDOM_SEED,
            "task_type": config.TASK_TYPE,
            "allow_writing_files": config.ALLOW_WRITING_FILES,
            "verbose": config.VERBOSE_PERIOD,
            "devices": config.DEVICES,
        }
        return params

    def objective(self, trial: optuna.Trial) -> float:
        params = self.get_model_params(trial)

        self.model = CatBoostClassifier(**params)

        pruning_callback = CatBoostPruningCallback(trial, "Logloss")

        try:
            self.model.fit(
                self.train_pool,
                eval_set=self.val_pool,
                callbacks=[pruning_callback]
            )

            best_score = self.model.get_best_score()["validation"]["MultiLogloss"]
            logger.info(f"Trial {trial.number}: лучший скор = {best_score:.6f}")

            return best_score

        except Exception as e:
            logger.error(f"Ошибка в Trial {trial.number}: {e}")
            raise

    def optimize_hyperparameters(self) -> None:
        logger.info("=" * 60)
        logger.info("ПОИСК ОПТИМАЛЬНЫХ ГИПЕРПАРАМЕТРОВ")
        logger.info("=" * 60)

        self.study = optuna.create_study(
            direction=config.OPTUNA_DIRECTION,
            pruner=MedianPruner()
        )

        logger.info(f"Запуск оптимизации на {config.OPTUNA_TRIALS} попыток...")
        start_time = time.time()

        try:
            self.study.optimize(self.objective, n_trials=config.OPTUNA_TRIALS)
            elapsed = time.time() - start_time

            logger.info(f"Оптимизация завершена за {elapsed:.2f} секунд")
            logger.info(f"Лучший скор: {self.study.best_value:.6f}")
            logger.info(f"Лучшие параметры: {self.study.best_params}")

        except Exception as e:
            logger.error(f"Ошибка при оптимизации: {e}")

            if self.model is not None:
                self.save_training_info()
            raise

    def save_training_info(self) -> None:
        logger.info("Сохранение информации об обучении...")

        train_info = {
            "timestamp": datetime.now().isoformat(),
            "best_iteration": int(self.model.get_best_iteration()),
            "best_score": self.model.get_best_score(),
            "model_params": self.model.get_params(),
        }

        if self.study is not None:
            train_info["optuna_best_params"] = self.study.best_params
            train_info["optuna_best_value"] = self.study.best_value

        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        info_path = config.MODELS_DIR / "training_info.json"

        try:
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(train_info, f, ensure_ascii=False, indent=2)
            logger.info(f"Информация об обучении сохранена: {info_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении информации об обучении: {e}")

    def save_model_and_info(self) -> None:
        """Сохранить обученную модель и информацию об обучении."""
        logger.info("=" * 60)
        logger.info("СОХРАНЕНИЕ МОДЕЛИ")
        logger.info("=" * 60)

        if self.model is None:
            logger.error("Модель не обучена!")
            return

        timestamp = time.time()
        model_name = f"model_{timestamp:.0f}.cbm"
        model_path = config.MODELS_DIR / model_name

        utils.save_model(self.model, model_path, logger)

        self.save_training_info()

    def train(self) -> None:
        """Главный метод обучения."""
        try:
            self.load_data()
            self.prepare_features()
            self.split_data()
            self.optimize_hyperparameters()
            self.save_model_and_info()

            logger.info("=" * 60)
            logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Критическая ошибка при обучении: {e}", exc_info=True)
            raise


def main():
    logger.info("Начало обучения модели")
    logger.info(f"Дата/время: {datetime.now().isoformat()}")

    trainer = ModelTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
