import mlflow
import os
import pandas as pd

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET = "MedHouseVal"

model_names = ["random_forest", "linear_regression", "desicion_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))

# Получим датасет California housing
housing = fetch_california_housing(as_frame=True)
# Объединим фичи и таргет в один np.array
data = pd.concat([housing["data"], pd.DataFrame(housing["target"])], axis=1)

# Сделать препроцессинг
# Разделить на фичи и таргет
X, y = data[FEATURES], data[TARGET]

# Разделить данные на обучение и тест
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучить стандартизатор на train
scaler = StandardScaler()
X_train_fitted = scaler.fit_transform(X_train)
X_test_fitted = scaler.transform(X_test)

# Создать новый эксперимент
exp_name = "Tatyana_Panyutina"
experiment_id = mlflow.create_experiment(exp_name)
mlflow.set_experiment(exp_name)

with mlflow.start_run(run_name="t_panyutina", experiment_id=experiment_id, description="parent") as parent_run:
    for model_name in models.keys():
        # Запустим child run на каждую модель.
        with mlflow.start_run(run_name=model_name, experiment_id=experiment_id, nested=True) as child_run:
            model = models[model_name]

            # Обучим модель.
            model.fit(pd.DataFrame(X_train_fitted), y_train)

            # Сделаем предсказание.
            prediction = model.predict(X_test_fitted)

            # Создадим валидационный датасет.
            eval_df = X_test.copy()
            eval_df["target"] = y_test

            # Сохраним результаты обучения с помощью MLFlow.
            signature = infer_signature(X_test, prediction)
            model_info = mlflow.sklearn.log_model(model, "logreg", signature=signature)
            mlflow.evaluate(
                model=model_info.model_uri,
                data=eval_df,
                targets="target",
                model_type="regressor",
                evaluators=["default"],
            )