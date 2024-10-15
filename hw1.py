import io
import json
import logging
import pandas as pd
import pickle

from airflow.models import DAG, Variable
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, Literal
from sklearn.datasets import fetch_california_housing
from datetime import datetime, timedelta

from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

from airflow.providers.amazon.aws.hooks.s3 import S3Hook

BUCKET = Variable.get("S3_BUCKET")

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

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

DEFAULT_ARGS = {
    "owner": "Tatyana Panyutina",
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
}

model_names = ["random_forest", "linear_regression", "desicion_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))


def create_dag(dag_id: str, m_name: Literal["random_forest", "linear_regression", "desicion_tree"]):
    ####### DAG STEPS #######

    def init() -> Dict[str, Any]:
        start_time = datetime.now().strftime('%H:%M:%S %d.%m.%y')
        _LOG.info("Train pipeline started.")
        return {'model_name': model_name,
                'start_time': start_time}

    def get_data(**kwargs) -> Dict[str, Any]:
        metrics = kwargs['ti'].xcom_pull(task_ids='init')
        metrics['start_get_data_time'] = datetime.now().strftime('%H:%M:%S %d.%m.%y')

        # Получим датасет California housing
        housing = fetch_california_housing(as_frame=True)
        metrics['end_get_data_time'] = datetime.now().strftime('%H:%M:%S %d.%m.%y')
        # Объединим фичи и таргет в один np.array
        data = pd.concat([housing["data"], pd.DataFrame(housing["target"])], axis=1)
        metrics['dataset_shape'] = data.shape

        # Использовать созданный ранее S3 connection
        s3_hook = S3Hook("s3_connection")
        filebuffer = io.BytesIO()
        data.to_pickle(filebuffer)
        filebuffer.seek(0)

        # Сохранить файл в формате pkl на S3
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f"2024/datasets/california_housing_{m_name}.pkl",
            bucket_name=BUCKET,
            replace=True,
        )

        _LOG.info("Data downloaded.")
        return metrics

    def prepare_data(**kwargs) -> Dict[str, Any]:
        metrics = kwargs['ti'].xcom_pull(task_ids='get_data')
        # Использовать созданный ранее S3 connection
        start_timestamp = datetime.now()
        s3_hook = S3Hook("s3_connection")
        file = s3_hook.download_file(key=f"2024/datasets/california_housing_{m_name}.pkl", bucket_name=BUCKET)
        data = pd.read_pickle(file)

        # Сделать препроцессинг
        # Разделить на фичи и таргет
        X, y = data[FEATURES], data[TARGET]

        # Разделить данные на обучение и тест
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Обучить стандартизатор на train
        scaler = StandardScaler()
        X_train_fitted = scaler.fit_transform(X_train)
        X_test_fitted = scaler.transform(X_test)

        # Обучить стандартизатор на train
        scaler = StandardScaler()
        X_train_fitted = scaler.fit_transform(X_train)
        X_test_fitted = scaler.transform(X_test)

        # Сохранить готовые данные на S3
        session = s3_hook.get_session("ru-central1")
        resource = session.resource("s3")

        for name, data in zip(
                [f"X_train_{m_name}", f"X_test_{m_name}", f"y_train_{m_name}", f"y_test_{m_name}"],
                [X_train_fitted, X_test_fitted, y_train, y_test],
        ):
            filebuffer = io.BytesIO()
            pickle.dump(data, filebuffer)
            filebuffer.seek(0)
            s3_hook.load_file_obj(
                file_obj=filebuffer,
                key=f"2024/datasets/{name}.pkl",
                bucket_name=BUCKET,
                replace=True,
            )

        end_timestamp = datetime.now()
        metrics['prepare_data_time'] = end_timestamp - start_timestamp

        _LOG.info("Data prepared.")
        return metrics

    def train_model(**kwargs) -> None:
        metrics = kwargs['ti'].xcom_pull(task_ids='get_data')
        start_timestamp = datetime.now()
        # Использовать созданный ранее S3 connection
        s3_hook = S3Hook("s3_connection")
        # Загрузить готовые данные с S3
        data = {}
        for name in ["X_train", "X_test", "y_train", "y_test"]:
            file = s3_hook.download_file(
                key=f"2024/datasets/{name}_{m_name}.pkl",
                bucket_name=BUCKET,
            )
            data[name] = pd.read_pickle(file)

        # Обучить модель
        model = models[m_name]
        model.fit(data["X_train"], data["y_train"])
        prediction = model.predict(data["X_test"])

        # Посчитать метрики
        metrics["r2_score"] = r2_score(data["y_test"], prediction)
        metrics["rmse"] = mean_squared_error(data["y_test"], prediction) ** 0.5
        metrics["mae"] = median_absolute_error(data["y_test"], prediction)

        end_timestamp = datetime.now()
        metrics['train_model_time'] = (end_timestamp - start_timestamp).total_seconds()

        # Сохранить результат на S3
        date = datetime.now().strftime("%Y_%m_%d_%H")
        filebuffer = io.BytesIO()
        filebuffer.write(json.dumps(metrics).encode())
        filebuffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f"2024/metrics/{m_name}_{date}.pkl",
            bucket_name=BUCKET,
            replace=True,
        )
        _LOG.info("Model trained.")

    def save_results() -> None:
        print('Files saved')

        ####### INIT DAG #######

    dag = DAG(dag_id=dag_id,
              schedule_interval="0 1 * * *",
              start_date=days_ago(1),
              catchup=False,
              tags=["mlops"],
              default_args=DEFAULT_ARGS,
              )

    with dag:
        # YOUR TASKS HERE
        task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)

        task_get_data = PythonOperator(task_id="get_data", python_callable=get_data, dag=dag)

        task_prepare_data = PythonOperator(task_id="data_preparation", python_callable=prepare_data, dag=dag)

        task_train_model = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag)

        task_save_results = PythonOperator(task_id="save_results", python_callable=save_results, dag=dag)

        task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results


for model_name in models.keys():
    create_dag(f"Tatyana_Panyutina_{model_name}", model_name)