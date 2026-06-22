"""
Utilities for PostgreSQL database operations.
"""

import json
import psycopg2
from psycopg2.extras import Json
from typing import Dict, Any, Optional
import os


def get_postgres_connection():
    """
    Create PostgreSQL connection using environment variables.
    
    Returns:
        psycopg2.connection: Database connection
    """
    return psycopg2.connect(
        host='postgres',
        port=5432,
        database=os.getenv('PG_DATABASE', 'airflow'),
        user=os.getenv('PG_USER', 'airflow'),
        password=os.getenv('PG_PASSWORD', 'airflow')
    )


def insert_training_metrics(
    run_id: str,
    model_name: str,
    experiment_name: str,
    metrics: Dict[str, float],
    model_params: Dict[str, Any],
    data_info: Dict[str, Any],
    is_valid: bool,
    optuna_cv_rmse: Optional[float] = None,
    n_trials: Optional[int] = None
) -> int:
    """
    Insert training metrics into PostgreSQL database.
    
    Args:
        run_id: MLflow run ID
        model_name: Name of the model
        experiment_name: Name of the experiment
        metrics: Dictionary with training metrics
        model_params: Model hyperparameters
        data_info: Dataset information
        is_valid: Whether model passed validation
        optuna_cv_rmse: Optuna best CV RMSE (optional)
        n_trials: Number of Optuna trials (optional)
        
    Returns:
        int: ID of inserted record
    """
    conn = get_postgres_connection()
    cur = conn.cursor()
    
    try:
        insert_query = """
        INSERT INTO ml_metrics.model_training (
            run_id, model_name, experiment_name,
            train_r2, test_r2, train_mae, test_mae, train_rmse, test_rmse, test_mape,
            optuna_best_cv_rmse, n_trials,
            model_params,
            train_size, test_size, n_features, target_mean, target_std,
            is_valid
        ) VALUES (
            %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s,
            %s,
            %s, %s, %s, %s, %s,
            %s
        )
        ON CONFLICT (run_id) 
        DO UPDATE SET
            test_r2 = EXCLUDED.test_r2,
            test_mae = EXCLUDED.test_mae,
            test_rmse = EXCLUDED.test_rmse,
            test_mape = EXCLUDED.test_mape,
            is_valid = EXCLUDED.is_valid
        RETURNING id;
        """
        
        cur.execute(insert_query, (
            run_id, model_name, experiment_name,
            metrics.get('train_r2'), metrics.get('test_r2'),
            metrics.get('train_mae'), metrics.get('test_mae'),
            metrics.get('train_rmse'), metrics.get('test_rmse'),
            metrics.get('test_mape'),
            optuna_cv_rmse, n_trials,
            Json(model_params),
            data_info.get('train_size'), data_info.get('test_size'),
            data_info.get('n_features'), data_info.get('target_mean'),
            data_info.get('target_std'),
            is_valid
        ))
        
        record_id = cur.fetchone()[0]
        conn.commit()
        
        print(f"✓ Métricas guardadas en PostgreSQL (ID: {record_id})")
        return record_id
        
    except Exception as e:
        conn.rollback()
        print(f"✗ Error guardando métricas en PostgreSQL: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()


def update_model_promotion(run_id: str, model_version: str, model_stage: str) -> None:
    """
    Update model promotion status in PostgreSQL.
    
    Args:
        run_id: MLflow run ID
        model_version: Model version number
        model_stage: Model stage (Production, Staging, etc.)
    """
    conn = get_postgres_connection()
    cur = conn.cursor()
    
    try:
        update_query = """
        UPDATE ml_metrics.model_training
        SET is_promoted = TRUE,
            model_version = %s,
            model_stage = %s
        WHERE run_id = %s;
        """
        
        cur.execute(update_query, (model_version, model_stage, run_id))
        conn.commit()
        
        print(f"✓ Estado de promoción actualizado en PostgreSQL")
        
    except Exception as e:
        conn.rollback()
        print(f"✗ Error actualizando promoción en PostgreSQL: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()


def get_latest_metrics(model_name: str, limit: int = 10) -> list:
    """
    Get latest training metrics for a model.
    
    Args:
        model_name: Name of the model
        limit: Number of records to retrieve
        
    Returns:
        list: List of metric records
    """
    conn = get_postgres_connection()
    cur = conn.cursor()
    
    try:
        query = """
        SELECT 
            run_id, training_date, test_r2, test_mae, test_rmse, test_mape,
            is_valid, is_promoted, model_version, model_stage
        FROM ml_metrics.model_training
        WHERE model_name = %s
        ORDER BY training_date DESC
        LIMIT %s;
        """
        
        cur.execute(query, (model_name, limit))
        results = cur.fetchall()
        
        return results
        
    finally:
        cur.close()
        conn.close()
