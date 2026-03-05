#!/usr/bin/env python
# coding: utf-8

# In[1]:


import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import optuna

from catboost import Pool, CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from optuna.pruners import MedianPruner
from optuna.integration import CatBoostPruningCallback

try:
    import pyarrow
    print("PyArrow is installed. Version:", pyarrow.__version__)
except ModuleNotFoundError:
    print("PyArrow is NOT found in this environment. You installed it in a different one.")




X = pd.read_parquet('../modified_data/train_no_nan_min_var.parquet')
y = pd.read_parquet('../data/train_target.parquet')


test_main = pd.read_parquet('../data/test_main_features.parquet')
test_extra = pd.read_parquet('../data/test_extra_features.parquet')



print('Тренировочные данные:', X.shape)
print('Тестовые данные:', test_main.shape)
print('Тестовые данные:', test_extra.shape)









cat_feature_names = [
    col_name for col_name in X.columns 
    if col_name.startswith("cat_feature")
]




# train = pd.merge(train_main, train_extra, on="customer_id", how="left")
test  = pd.merge(test_main,  test_extra,  on="customer_id", how="left")




test[X.columns]




X[cat_feature_names] = X[cat_feature_names].astype(str)
test[cat_feature_names] = test[cat_feature_names].astype(str)




for c in cat_feature_names:
    X[c] = X[c].fillna("__MISSING__").astype(str)
    test[c]  = test[c].fillna("__MISSING__").astype(str)




for c in cat_feature_names:
    freq = X[c].value_counts(dropna=False)
    X[c + "__freq"] = X[c].map(freq).fillna(0).astype("int32")
    test[c + "__freq"]  = test[c].map(freq).fillna(0).astype("int32")




train, val, target, val_target = train_test_split(X, y, test_size = 0.2, random_state = 42)



train_pool = Pool(data = train.drop("customer_id", axis=1), 
                  label = target.drop("customer_id", axis=1), 
                  cat_features = cat_feature_names)




val_pool = Pool(data = val.drop("customer_id", axis=1), 
                  label = val_target.drop("customer_id", axis=1), 
                  cat_features = cat_feature_names)









def objective(trial: optuna.Trial):
    params = {
        "loss_function":"MultiLogloss",

        "iterations":4000,         
        "learning_rate":trial.suggest_float( "lr", 3e-6, 0.1, log=True),       # стабильнее, чем 0.1
        "depth":trial.suggest_int("depth", 4, 12),                  # лучше для 2200 фичей

        "l2_leaf_reg": trial.suggest_float("l2", 4, 50.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "od_type": "Iter",
        "od_wait": 200,
        "use_best_model": True,


        "random_seed":42,
        "task_type":"GPU",
        "allow_writing_files":False,
        "verbose":200,
        "devices":'0',
    }


    model = CatBoostClassifier(**params)

    model.fit(train_pool, eval_set=val_pool)

    return model.get_best_score()["validation"]["MultiLogloss"]


try:

	study = optuna.create_study(direction="minimize")
	study.optimize(objective, n_trials=20)
except Exception:
	train_info = {
    		"best_iteration": int(model.get_best_iteration()),
    		"best_score": model.get_best_score(),
    		"params": model.get_params(),
	}

	with open(train_itrain_info_.json, "w", encoding="utf-8") as f:
    		json.dump(train_info, f, ensure_ascii=False, indent=2)
    		json.dump(study.best_params, f, ensure_ascii=False, indent=2)

# with open(train_itrain_info_.json, "w", encoding="utf-8") as f:








test_pool = Pool(data = test.drop("customer_id", axis = 1), 
                 cat_features = cat_feature_names)




test_predict = model.predict(test_pool, prediction_type = "RawFormulaVal")

test_predict.shape




predict_schema = [col.replace("target_", "predict_") for col in target.columns if col.startswith("target_")]

catboost_predictions = pl.DataFrame(test_predict, schema = predict_schema)

catboost_predictions.head(n = 5)


# проверим roc_auc на тренировочной выборке


y_true = target.drop('customer_id', axis = 1)
y_pred = model.predict(train_pool, prediction_type = 'RawFormulaVal')
# roc_auc_score(y_true, y_pred, average="macro")

roc_auc_score(y_true, y_pred, average="macro")


# сохраняем сабмит для сдачи



sample_submit = pd.read_parquet('./submits/sample_submit.parquet')




timestamp = time.time()




result_df = sample_submit.copy()
result_df.iloc[:, 1:] = test_predict
result_df['customer_id'] = result_df['customer_id'].astype('int32')
current_type = result_df['customer_id'].dtype

result_df.to_parquet(f'./submits/exp_{timestamp}.parquet', index=False)


# сохраняем модельку))))))


model.save_model(f"./models/exp_{timestamp}.cbm")






