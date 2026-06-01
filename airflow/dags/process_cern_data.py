

import os
from datetime import datetime, timedelta
from typing import Dict, Any

import polars as pl
import numpy as np

from airflow import DAG  # pyright: ignore
from airflow.operators.python import PythonOperator  # pyright: ignore
from airflow.operators.trigger_dagrun import TriggerDagRunOperator  # pyright: ignore


RAW_DATA_PATH = "/opt/airflow/dags/data/dielectron.csv"
PROCESSED_DATA_PATH = "/opt/airflow/dags/data/dielectron.parquet"

COLUMNS_TO_KEEP = [
    'E1', 'px1', 'py1', 'pz1', 'pt1', 'eta1', 'phi1', 'Q1',
    'E2', 'px2', 'py2', 'pz2', 'pt2', 'eta2', 'phi2', 'Q2',
    'M'
]

FINAL_COLUMNS = [
    'E1', 'px1', 'py1', 'pz1', 'pt1', 'eta1', 'phi1', 'Q1',
    'E2', 'px2', 'py2', 'pz2', 'pt2', 'eta2', 'phi2', 'Q2',
    'M',
    'delta_eta', 'delta_phi', 'delta_R',
    'pt_product', 'pt_ratio', 'E_total', 'is_os'
]

default_args = {
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'dagrun_timeout': timedelta(minutes=15)
}


def load_raw_data(**context) -> Dict[str, Any]:
    print("Cargando dataset CERN...")
    df = pl.read_csv(RAW_DATA_PATH)
    
    # Clean column names (remove leading/trailing whitespace)
    df.columns = [col.strip() for col in df.columns]
    print(f"Columnas después de limpiar: {df.columns}")

    print("📊 Dataset crudo cargado:")
    print(f"   Forma: {df.shape}")
    print(f"   Columnas: {df.shape[1]}")
    print(f"   Filas: {df.shape[0]:,}")

    # Basic info
    data_info = {
        'initial_rows': len(df),
        'initial_cols': df.shape[1],
        'columns': list(df.columns)  # Convert to list for serialization
    }

    # Save DataFrame to temporary parquet file instead of XCom
    temp_path = "/opt/airflow/dags/data/temp_raw.parquet"
    df.write_parquet(temp_path)
    context['ti'].xcom_push(key='temp_path', value=temp_path)
    context['ti'].xcom_push(key='data_info', value=data_info)

    return data_info


def remove_duplicates(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    initial_rows = len(df)
    df_clean = df.unique()
    duplicates_removed = initial_rows - len(df_clean)
    
    print(f"   Duplicados removidos: {duplicates_removed}")
    return df_clean, duplicates_removed


def remove_missing_values(df: pl.DataFrame, subset: list[str] = None) -> tuple[pl.DataFrame, int]:
    if subset:
        missing_before = df.filter(
            pl.any_horizontal([pl.col(col).is_null() for col in subset])
        ).height
        df_clean = df.drop_nulls(subset=subset)
    else:
        missing_before = df.null_count().sum_horizontal()[0]
        df_clean = df.drop_nulls()
    print(f"   Valores faltantes removidos: {missing_before}")
    return df_clean, missing_before


def select_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    df_selected = df.select(columns)
    print(f"   Columnas conservadas: {len(columns)}")
    return df_selected


def filter_physical_outliers(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    initial_rows = len(df)
    df_filtered = df.filter(
        (pl.col('E1') > 0) & 
        (pl.col('E2') > 0) &
        (pl.col('pt1') > 0) & 
        (pl.col('pt2') > 0) &
        (pl.col('M') > 0)
    )
    outliers_removed = initial_rows - len(df_filtered)
    print(f"   Outliers físicos removidos: {outliers_removed}")
    return df_filtered, outliers_removed

def clean_data(**context) -> Dict[str, Any]:
    print("Limpiando el dataset...")
    ti = context['ti']
    temp_path = ti.xcom_pull(key='temp_path', task_ids='load_raw_data')
    df = pl.read_parquet(temp_path)

    initial_rows = len(df)
    df, duplicates_removed = remove_duplicates(df)
    df, missing_removed = remove_missing_values(df, subset=['M'])
    df = select_columns(df, COLUMNS_TO_KEEP)
    df, outliers_removed = filter_physical_outliers(df)
    final_rows = len(df)

    cleaning_stats = {
        'initial_rows': initial_rows,
        'duplicates_removed': duplicates_removed,
        'missing_removed': missing_removed,
        'outliers_removed': outliers_removed,
        'final_rows': final_rows,
        'rows_removed': initial_rows - final_rows
    }

    print("\n Limpieza completada:")
    print(f"   Filas iniciales: {initial_rows:,}")
    print(f"   Filas finales: {final_rows:,}")
    
    # Save cleaned data to temp file
    temp_clean_path = "/opt/airflow/dags/data/temp_clean.parquet"
    df.write_parquet(temp_clean_path)
    ti.xcom_push(key='temp_clean_path', value=temp_clean_path)
    ti.xcom_push(key='cleaning_stats', value=cleaning_stats)
    return cleaning_stats


def create_angular_features(df: pl.DataFrame) -> pl.DataFrame:
    print("   Creando características angulares (delta_eta, delta_phi, delta_R)...")
    
    df = df.with_columns([
        (pl.col('eta1') - pl.col('eta2')).abs().alias('delta_eta')
    ])

    df = df.with_columns([
        (pl.col('phi1') - pl.col('phi2')).abs().alias('delta_phi')
    ])
    
    df = df.with_columns([
        pl.when(pl.col('delta_phi') > np.pi)
        .then(2 * np.pi - pl.col('delta_phi'))
        .otherwise(pl.col('delta_phi'))
        .alias('delta_phi')
    ])

    df = df.with_columns([
        (pl.col('delta_eta')**2 + pl.col('delta_phi')**2).sqrt().alias('delta_R')
    ])
    
    return df


def create_kinematic_features(df: pl.DataFrame) -> pl.DataFrame:
    print("   Creando características cinemáticas (pt_product, pt_ratio, E_total)...")
    
    df = df.with_columns([
        (pl.col('pt1') * pl.col('pt2')).alias('pt_product'),
        (pl.col('pt1') / pl.col('pt2')).alias('pt_ratio'),
        (pl.col('E1') + pl.col('E2')).alias('E_total')
    ])
    
    return df


def create_charge_features(df: pl.DataFrame) -> pl.DataFrame:
    print("   Creando características de carga (is_os)...")
    
    df = df.with_columns([
        (pl.col('Q1') * pl.col('Q2') < 0).cast(pl.Int32).alias('is_os')
    ])
    
    return df


def create_features(**context) -> Dict[str, Any]:
    print("Creando características...")

    ti = context['ti']
    temp_clean_path = ti.xcom_pull(key='temp_clean_path', task_ids='clean_data')
    df = pl.read_parquet(temp_clean_path)
    initial_features = df.shape[1]

    df = create_angular_features(df)
    df = create_kinematic_features(df)
    df = create_charge_features(df)
    final_features = df.shape[1]
    features_created = final_features - initial_features
    df = df.select(FINAL_COLUMNS)

    feature_stats = {
        'initial_features': initial_features,
        'features_created': features_created,
        'final_features': final_features,
        'new_features': [
            'delta_eta', 'delta_phi', 'delta_R',
            'pt_product', 'pt_ratio', 'E_total', 'is_os'
        ]
    }

    print("\n✓ Características creadas:")
    print(f"   Características iniciales: {initial_features}")
    print(f"   Características nuevas: {features_created}")
    print(f"   Características finales: {final_features}")
    print(f"   Nombres de nuevas características: {feature_stats['new_features']}")

    # Save processed data to temp file
    temp_processed_path = "/opt/airflow/dags/data/temp_processed.parquet"
    df.write_parquet(temp_processed_path)
    ti.xcom_push(key='temp_processed_path', value=temp_processed_path)
    ti.xcom_push(key='feature_stats', value=feature_stats)

    return feature_stats



def save_processed_data(**context) -> Dict[str, Any]:

    print("Guardando datos procesados...")
    ti = context['ti']
    temp_processed_path = ti.xcom_pull(key='temp_processed_path', task_ids='create_features')
    df = pl.read_parquet(temp_processed_path)

    df.write_parquet(PROCESSED_DATA_PATH)

    save_stats = {
        'output_path': PROCESSED_DATA_PATH,
        'rows': len(df),
        'columns': df.shape[1]
    }

    print("\n Datos guardados en formato Parquet")
    
    # Clean up temporary files
    import os
    for temp_file in ['/opt/airflow/dags/data/temp_raw.parquet', 
                      '/opt/airflow/dags/data/temp_clean.parquet',
                      '/opt/airflow/dags/data/temp_processed.parquet']:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return save_stats




with DAG(
    'process_cern_data',
    default_args=default_args,
    description='Limpieza de datos de dataset CERN electron collision',
    tags=['etl', 'cern', 'preprocessing'],
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    load_data_task = PythonOperator(
        task_id='load_raw_data',
        python_callable=load_raw_data,
    )

    clean_data_task = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
    )

    create_features_task = PythonOperator(
        task_id='create_features',
        python_callable=create_features,
    )

    save_data_task = PythonOperator(
        task_id='save_processed_data',
        python_callable=save_processed_data,
    )

    trigger_training = TriggerDagRunOperator(
        task_id='trigger_training',
        trigger_dag_id='train_cern_xgboost',
        wait_for_completion=False,
    )

    # Define task dependencies
    (
        load_data_task
        >> clean_data_task
        >> create_features_task
        >> save_data_task
        >> trigger_training
    )  # pyright: ignore
