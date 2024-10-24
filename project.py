import mlflow
import os
import io
import pickle
import pandas as pd
import json

from airflow.models import DAG, Variable
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from typing import Any, Dict, Literal
from sklearn.datasets import fetch_california_housing
from datetime import datetime, timedelta
from io import StringIO

from mlflow.models import infer_signature


BUCKET = Variable.get("S3_BUCKET")
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

s3_hook = S3Hook("s3_connection")

DEFAULT_ARGS = {
    'owner': 'Tatyana Panyutina',
    'retry': 3,
    'retry-delay': timedelta(minutes=1)
}
dag = DAG(
    dag_id='TatyanaPanyutina',
    schedule_interval='0 1 * * *',
    start_date=days_ago(2),
    catchup=False,
    tags=['mlops'],
    default_args=DEFAULT_ARGS
)
model_names = ["random_forest", "linear_regression", "decision_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))

def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)


def save_data_to_s3(data, title) -> None:
    buffer = io.BytesIO()
    pickle.dump(data, buffer)
    buffer.seek(0)
    s3_hook.load_file_obj(
        file_obj=buffer,
        key=title,
        bucket_name=BUCKET,
        replace=True,
    )

def load_data_from_s3(title) -> Any:
    s3_object = s3_hook.get_key(
        key=title,
        bucket_name=BUCKET
    )
    buffer = io.BytesIO(s3_object.get()['Body'].read())
    data = pickle.load(buffer)
    return data

def init() -> Dict[str, Any]:
    configure_mlflow()
    exp_name = "TatyanaPanyutina"
    experiment_id = mlflow.create_experiment(exp_name)
    mlflow.set_experiment(exp_name)
    parent_run = mlflow.start_run(
        run_name='t_panyutina',
        experiment_id=experiment_id,
        description="parent",
    )
    metrics = {
            "exp_name": exp_name,
            "exp_id": experiment_id,
            "parent_run_name": 't_panyutina',
            "parent_run_id": parent_run.info.run_id,
        }
    return metrics

def get_data(**kwargs) -> Dict[str, Any]:
    metrics = kwargs['ti'].xcom_pull(task_ids='init')
    metrics['start_get_data_time'] = datetime.now().strftime('%H:%M:%S %d.%m.%y')

    # Получим датасет California housing
    housing = fetch_california_housing(as_frame=True)
    metrics['end_get_data_time'] = datetime.now().strftime('%H:%M:%S %d.%m.%y')
    # Объединим фичи и таргет в один np.array
    data = pd.concat([housing["data"], pd.DataFrame(housing["target"])], axis=1)
    metrics['dataset_shape'] = data.shape
    save_data_to_s3(data, "TatyanaPanyutina/datasets/data.pkl")
    return metrics

def prepare_data(**kwargs) -> Dict[str, Any]:
    metrics = kwargs['ti'].xcom_pull(task_ids='get_data')
    # Использовать созданный ранее S3 connection
    metrics['start_data_preparation_time'] = datetime.now().strftime('%H:%M:%S %d.%m.%y')
    data = load_data_from_s3("TatyanaPanyutina/datasets/data.pkl")

    # Сделать препроцессинг
    # Разделить на фичи и таргет
    X, y = data[FEATURES], data[TARGET]

    # Разделить данные на обучение и тест
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Обучить стандартизатор на train
    scaler = StandardScaler()
    X_train_fitted = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_fitted = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # Сохранить готовые данные на S3
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")

    for name, data in zip(
            ["X_train", "X_test", "y_train", "y_test"],
            [X_train_fitted, X_test_fitted, y_train, y_test],
    ):
        save_data_to_s3(data, f"TatyanaPanyutina/datasets/{name}.pkl")

    metrics['end_data_preparation_time'] = datetime.now().strftime('%H:%M:%S %d.%m.%y')
    return metrics

def train_model(m_name, **kwargs) -> Dict[str, Any]:
    metrics = kwargs['ti'].xcom_pull(task_ids='prepare_data')
    metrics_model = {}
    metrics_model['start_train_model_time'] = datetime.now().strftime('%H:%M:%S %d.%m.%y')
    # Загрузить готовые данные с S3
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        data[name] = load_data_from_s3(f"TatyanaPanyutina/datasets/{name}.pkl")

    # Обучить модель
    model = models[m_name]
    model.fit(data["X_train"], data["y_train"])

    configure_mlflow()

    # MLFlow child run
    with mlflow.start_run(
            run_name=m_name,
            experiment_id=metrics["exp_id"],
            nested=True,
    ) as child_run:
        # Log model
        signature = infer_signature(data['X_test'], model.predict(data['X_test']))
        model_info = mlflow.sklearn.log_model(
            model,
            m_name,
            signature=signature,
            registered_model_name=f"sk-learn-{m_name}-model",
        )

        # Evaluate model
        mlflow.evaluate(
            model=model_info.model_uri,
            data=pd.concat([data['X_test'], data['y_test']], axis=1).reset_index(drop=True),
            targets=TARGET,
            model_type="regressor",
            evaluators=["default"],
        )

    metrics_model['end_train_model_time'] = datetime.now().strftime('%H:%M:%S %d.%m.%y')
    return metrics_model

def save_results(**kwargs) -> None:
    all_models_metrics = {}
    all_models_metrics['random_forest'] = kwargs['ti'].xcom_pull(task_ids=f'train_model_random_forest')
    all_models_metrics['linear_regression'] = kwargs['ti'].xcom_pull(task_ids=f'train_model_linear_regression')
    all_models_metrics['decision_tree'] = kwargs['ti'].xcom_pull(task_ids=f'train_model_decision_tree')
    all_models_metrics['common_metrics'] = kwargs['ti'].xcom_pull(task_ids='prepare_data')
    path = f"TatyanaPanyutina/results/metrics.pkl"

    filebuffer = io.BytesIO()
    filebuffer.write(json.dumps(all_models_metrics).encode())
    filebuffer.seek(0)
    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key=path,
        bucket_name=BUCKET,
        replace=True,
    )


task_init = PythonOperator(task_id='init', python_callable=init, dag=dag)

task_get_data = PythonOperator(task_id='get_data', python_callable=get_data, dag=dag)

task_prepare_data = PythonOperator(task_id='prepare_data', python_callable=prepare_data, dag=dag)

training_model_tasks = [PythonOperator(
                                        task_id=f'train_model_{m_name}',
                                        python_callable=train_model,
                                        op_kwargs={'m_name': m_name},
                                        dag=dag
                                      )
                        for m_name in models.keys()]

task_save_results = PythonOperator(task_id='save_results', python_callable=save_results, dag=dag)

task_init >> task_get_data >> task_prepare_data >> training_model_tasks >> task_save_results