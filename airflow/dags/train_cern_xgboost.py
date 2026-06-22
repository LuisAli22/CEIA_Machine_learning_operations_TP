"""
Airflow DAG for training XGBoost model on CERN electron collision data.
"""

from datetime import datetime, timedelta
from typing import Dict, Any

import polars as pl
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
import optuna
import pickle
from optuna.integration.mlflow import MLflowCallback
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_error,
    mean_absolute_percentage_error
)

from airflow import DAG
from airflow.operators.python import PythonOperator

from postgres_utils import insert_training_metrics, update_model_promotion


# Configuration
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_NAME = "cern_xgboost"
EXPERIMENT_NAME = "cern_electron_collision"

# Dataset path (mounted in Docker)
DATA_PATH = "/opt/airflow/dags/data/dielectron.parquet"

# Optuna configuration
N_TRIALS = 5   # Number of optimization trials (reduced for low resources)
CV_FOLDS = 2   # Cross-validation folds (reduced for low resources)

# Model parameters (will be overridden by Optuna)
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

# Features to use
FEATURES = [
    'pt1', 'pt2',
    'E_total',
    'delta_eta',
    'delta_phi',
    'delta_R',
    'pt_product',
    'pt_ratio',
    'is_os'
]

TARGET = 'M'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Quality thresholds
R2_THRESHOLD = 0.90
MAE_THRESHOLD = 5.0

# Default arguments for the DAG
default_args = {
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# ============================================================================
# DATA LOADING FUNCTIONS (Modular)
# ============================================================================

def load_parquet_data(path: str) -> pl.DataFrame:
    print(f"Cargando datos desde {path}...")
    df = pl.read_parquet(path)
    print(f"Forma del dataset: {df.shape}")
    return df


def prepare_features_target(df: pl.DataFrame, features: list[str], target: str) -> tuple:
    print(f"Características: {features}")
    print(f"Variable objetivo: {target}")

    X = df.select(features).to_numpy()
    y = df.select(target).to_numpy().ravel()

    return X, y


def split_train_test(X: np.ndarray, y: np.ndarray, test_size: float, random_state: int) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Tamaño conjunto entrenamiento: {len(X_train)}")
    print(f"Tamaño conjunto prueba: {len(X_test)}")
    print(f"Media del objetivo: {y.mean():.2f}")
    print(f"Desviación estándar del objetivo: {y.std():.2f}")

    return X_train, X_test, y_train, y_test


def load_and_prepare_data(**context) -> Dict[str, Any]:
    print("Cargando datos de colisiones de electrones CERN...")
    df = load_parquet_data(DATA_PATH)
    X, y = prepare_features_target(df, FEATURES, TARGET)
    
    # Validate data
    print("Validando datos...")
    print(f"  - NaN en X: {np.isnan(X).sum()}")
    print(f"  - Inf en X: {np.isinf(X).sum()}")
    print(f"  - NaN en y: {np.isnan(y).sum()}")
    print(f"  - Inf en y: {np.isinf(y).sum()}")
    
    # Remove any rows with NaN or Inf
    mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | np.isnan(y) | np.isinf(y))
    if not mask.all():
        print(f"  - Eliminando {(~mask).sum()} filas con valores inválidos")
        X = X[mask]
        y = y[mask]
    
    X_train, X_test, y_train, y_test = split_train_test(X, y, TEST_SIZE, RANDOM_STATE)

    data_info = {
        'train_size': len(X_train),
        'test_size': len(X_test),
        'n_features': X_train.shape[1],
        'target_mean': float(y.mean()),
        'target_std': float(y.std())
    }

    # Save data to pickle file (more reliable than XCom for large arrays)
    data_path = '/opt/airflow/dags/data/temp_train_data.pkl'
    with open(data_path, 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }, f)

    context['ti'].xcom_push(key='data_path', value=data_path)
    context['ti'].xcom_push(key='data_info', value=data_info)

    return data_info

def create_optuna_objective(X_train: np.ndarray, y_train: np.ndarray):
    def objective(trial: optuna.Trial) -> float:
        try:
            estimator_amount = trial.suggest_int('estimator_amount', 100, 500)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            subsample_ratio = trial.suggest_float('subsample_ratio', 0.6, 1.0)
            column_sample_ratio = trial.suggest_float('column_sample_ratio', 0.6, 1.0)
            min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
            gamma = trial.suggest_float('gamma', 0.0, 0.5)
            regularization_alpha = trial.suggest_float('regularization_alpha', 0.0, 1.0)
            regularization_lambda = trial.suggest_float('regularization_lambda', 0.0, 1.0)

            # Map readable names to XGBoost parameter names
            params = {
                'n_estimators': estimator_amount,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'subsample': subsample_ratio,
                'colsample_bytree': column_sample_ratio,
                'min_child_weight': min_child_weight,
                'gamma': gamma,
                'reg_alpha': regularization_alpha,
                'reg_lambda': regularization_lambda,
                'random_state': 42,
                'n_jobs': 1  # Force single thread to avoid memory issues
            }

            # Create and evaluate model with cross-validation
            model = xgb.XGBRegressor(**params)

            # Use negative RMSE as score (Optuna maximizes, so we use negative)
            # Set error_score to return a large penalty instead of failing
            scores = cross_val_score(
                model, X_train, y_train,
                cv=CV_FOLDS,
                scoring='neg_root_mean_squared_error',
                n_jobs=1,  # Force single thread
                error_score='raise'  # Raise error to see what's failing
            )

            # Return mean score (negative RMSE)
            return scores.mean()
        except Exception as e:
            print(f"Error in trial {trial.number}: {str(e)}")
            print(f"Parameters that caused error: {trial.params}")
            # Return a large penalty so Optuna learns to avoid these parameters
            return -999999.0

    return objective


def optimize_hyperparameters(**context) -> Dict[str, Any]:
    
    print(f"Optimizando hiperparámetros con Optuna ({N_TRIALS} intentos)...")
    configure_mlflow(MLFLOW_TRACKING_URI, EXPERIMENT_NAME)
    ti = context['ti']
    data_path = ti.xcom_pull(key='data_path', task_ids='load_data')
    X_train, y_train = load_train_data(data_path)
    study = optuna.create_study(
        direction='minimize',
        study_name=f'estimacion_masa_invariante_xgb_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )

    objective = create_optuna_objective(X_train, y_train)
    mlflow_callback = MLflowCallback(
        tracking_uri=MLFLOW_TRACKING_URI,
        metric_name='cv_rmse'
    )

    print(f"Iniciando optimización con {N_TRIALS} intentos y {CV_FOLDS} folds de validación cruzada...")
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        callbacks=[mlflow_callback],
        show_progress_bar=True
    )

    # Get best parameters
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1

    print("\n Optimización completada!")
    print(f"  Mejor RMSE CV: {-study.best_value:.4f}")
    print(f"  Mejores parámetros: {best_params}")

    # Save best parameters to XCom
    ti.xcom_push(key='best_params', value=best_params)
    ti.xcom_push(key='best_cv_rmse', value=-study.best_value)

    return best_params


def load_train_data(data_path: str) -> tuple:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data['X_train'], data['y_train']


def configure_mlflow(tracking_uri: str, experiment_name: str):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> xgb.XGBRegressor:
    print("Entrenando XGBoost...")
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


def log_model_to_mlflow(model: xgb.XGBRegressor, model_name: str, params: dict, features: list):
    mlflow.log_params(params)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("n_features", len(features))

    # Log model
    print("Registrando modelo en MLflow...")
    mlflow.xgboost.log_model(
        model,
        "model",
        registered_model_name=model_name
    )

    # Log tags
    mlflow.set_tag("model_type", "XGBoost")
    mlflow.set_tag("dataset", "CERN_dielectron")
    mlflow.set_tag("training_date", datetime.now().isoformat())
    mlflow.set_tag("features", str(features))


def train_xgboost_model(**context) -> str:
    print("Entrenando modelo XGBoost con parámetros optimizados...")
    configure_mlflow(MLFLOW_TRACKING_URI, EXPERIMENT_NAME)

    ti = context['ti']
    data_path = ti.xcom_pull(key='data_path', task_ids='load_data')
    best_params = ti.xcom_pull(key='best_params', task_ids='optimize_hyperparameters')

    params = best_params if best_params else XGBOOST_PARAMS

    print(f"Usando parámetros: {params}")
    X_train, y_train = load_train_data(data_path)
    with mlflow.start_run(run_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        model = train_xgboost(X_train, y_train, params)
        log_model_to_mlflow(model, MODEL_NAME, params, FEATURES)
        if best_params:
            best_cv_rmse = ti.xcom_pull(key='best_cv_rmse', task_ids='optimize_hyperparameters')
            mlflow.log_metric('optuna_best_cv_rmse', best_cv_rmse)
            mlflow.set_tag('optimized', 'true')
        else:
            mlflow.set_tag('optimized', 'false')

        run_id = run.info.run_id
        print(f"Modelo entrenado. Run ID: {run_id}")

        ti.xcom_push(key='run_id', value=run_id)

        return run_id

def load_test_data(data_path: str) -> tuple:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = '') -> dict:
    return {
        f'{prefix}r2': r2_score(y_true, y_pred),
        f'{prefix}mae': mean_absolute_error(y_true, y_pred),
        f'{prefix}rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def print_metrics(metrics: dict):
    print("\n📊 Métricas de Evaluación:")
    print(f"  R² Entrenamiento: {metrics['train_r2']:.4f}")
    print(f"  R² Prueba: {metrics['test_r2']:.4f}")
    print(f"  MAE Entrenamiento: {metrics['train_mae']:.4f}")
    print(f"  MAE Prueba: {metrics['test_mae']:.4f}")
    print(f"  RMSE Entrenamiento: {metrics['train_rmse']:.4f}")
    print(f"  RMSE Prueba: {metrics['test_rmse']:.4f}")
    if 'test_mape' in metrics:
        print(f"  MAPE Prueba: {metrics['test_mape']:.4f}")


def evaluate_model(**context) -> Dict[str, float]:
    print("Evaluando modelo XGBoost...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    ti = context['ti']
    data_path = ti.xcom_pull(key='data_path', task_ids='load_data')
    run_id = ti.xcom_pull(key='run_id', task_ids='train_model')

    X_train, y_train, X_test, y_test = load_test_data(data_path)

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.xgboost.load_model(model_uri)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_metrics = calculate_metrics(y_train, y_pred_train, 'train_')
    test_metrics = calculate_metrics(y_test, y_pred_test, 'test_')
    
    metrics = {**train_metrics, **test_metrics}
    metrics['test_mape'] = mean_absolute_percentage_error(y_test, y_pred_test)

    print_metrics(metrics)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)

    ti.xcom_push(key='metrics', value=metrics)

    return metrics


def validate_model(**context) -> bool:
    
    print("Validando modelo...")
    ti = context['ti']
    metrics = ti.xcom_pull(key='metrics', task_ids='evaluate_model')

    is_valid = (
        metrics['test_r2'] >= R2_THRESHOLD and
        metrics['test_mae'] <= MAE_THRESHOLD
    )

    if is_valid:
        print("✓ Modelo aprobó la validación")
        print(f"  R² Prueba: {metrics['test_r2']:.4f} >= {R2_THRESHOLD}")
        print(f"  MAE Prueba: {metrics['test_mae']:.4f} <= {MAE_THRESHOLD}")
    else:
        print("✗ Modelo no aprobó la validación")
        print(f"  R² Prueba: {metrics['test_r2']:.4f} < {R2_THRESHOLD}")
        print(f"  MAE Prueba: {metrics['test_mae']:.4f} > {MAE_THRESHOLD}")

    ti.xcom_push(key='is_valid', value=is_valid)

    return is_valid


def save_metrics_to_postgres(**context) -> int:
    """
    Save training metrics to PostgreSQL database.
    """
    print("Guardando métricas en PostgreSQL...")
    ti = context['ti']
    
    # Get data from previous tasks
    run_id = ti.xcom_pull(key='run_id', task_ids='train_model')
    metrics = ti.xcom_pull(key='metrics', task_ids='evaluate_model')
    data_info = ti.xcom_pull(key='data_info', task_ids='load_data')
    is_valid = ti.xcom_pull(key='is_valid', task_ids='validate_model')
    best_params = ti.xcom_pull(key='best_params', task_ids='optimize_hyperparameters')
    best_cv_rmse = ti.xcom_pull(key='best_cv_rmse', task_ids='optimize_hyperparameters')
    
    # Insert metrics
    record_id = insert_training_metrics(
        run_id=run_id,
        model_name=MODEL_NAME,
        experiment_name=EXPERIMENT_NAME,
        metrics=metrics,
        model_params=best_params if best_params else XGBOOST_PARAMS,
        data_info=data_info,
        is_valid=is_valid,
        optuna_cv_rmse=best_cv_rmse,
        n_trials=N_TRIALS if best_params else None
    )
    
    ti.xcom_push(key='postgres_record_id', value=record_id)
    return record_id


def promote_model_to_production(**context) -> None:
    print("Promoviendo modelo a Producción...")
    ti = context['ti']
    is_valid = ti.xcom_pull(key='is_valid', task_ids='validate_model')
    run_id = ti.xcom_pull(key='run_id', task_ids='train_model')

    if not is_valid:
        print("El modelo no aprobó la validación. Omitiendo promoción.")
        # Clean up temporary file even if validation fails
        data_path = ti.xcom_pull(key='data_path', task_ids='load_data')
        import os
        if data_path and os.path.exists(data_path):
            os.remove(data_path)
            print(f"Archivo temporal eliminado: {data_path}")
        return

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Get latest version
    latest_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])

    if not latest_versions:
        print("No se encontró versión del modelo para promover")
        return

    latest_version = latest_versions[0].version

    # Transition to Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f" Modelo versión {latest_version} promovido a Producción")
    
    # Update PostgreSQL with promotion info
    try:
        update_model_promotion(
            run_id=run_id,
            model_version=str(latest_version),
            model_stage="Production"
        )
    except Exception as e:
        print(f"⚠ Error actualizando PostgreSQL: {str(e)}")
    
    # Clean up temporary file
    data_path = ti.xcom_pull(key='data_path', task_ids='load_data')
    import os
    if data_path and os.path.exists(data_path):
        os.remove(data_path)
        print(f"Archivo temporal eliminado: {data_path}")

with DAG(
    'train_cern_xgboost',
    default_args=default_args,
    description='Train XGBoost model on CERN electron collision data with Optuna optimization',
    schedule=None,  # Manual execution only (changed from @weekly)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training', 'cern', 'xgboost', 'optuna'],
) as dag:
    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_and_prepare_data,
    )
    optimize_task = PythonOperator(
        task_id='optimize_hyperparameters',
        python_callable=optimize_hyperparameters,
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_xgboost_model,
    )

    evaluate_model_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
    )

    validate_model_task = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model,
    )

    save_to_postgres_task = PythonOperator(
        task_id='save_to_postgres',
        python_callable=save_metrics_to_postgres,
    )

    promote_model_task = PythonOperator(
        task_id='promote_model',
        python_callable=promote_model_to_production,
    )

    load_data_task >> optimize_task >> train_model_task >> evaluate_model_task >> validate_model_task >> save_to_postgres_task >> promote_model_task
