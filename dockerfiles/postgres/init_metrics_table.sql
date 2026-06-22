-- Create schema for ML metrics
CREATE SCHEMA IF NOT EXISTS ml_metrics;

-- Create table for model training metrics
CREATE TABLE IF NOT EXISTS ml_metrics.model_training (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    experiment_name VARCHAR(255),
    training_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Metrics
    train_r2 FLOAT,
    test_r2 FLOAT,
    train_mae FLOAT,
    test_mae FLOAT,
    train_rmse FLOAT,
    test_rmse FLOAT,
    test_mape FLOAT,
    
    -- Optuna metrics
    optuna_best_cv_rmse FLOAT,
    n_trials INT,
    
    -- Model parameters (stored as JSONB for flexibility)
    model_params JSONB,
    
    -- Dataset info
    train_size INT,
    test_size INT,
    n_features INT,
    target_mean FLOAT,
    target_std FLOAT,
    
    -- Validation
    is_valid BOOLEAN,
    is_promoted BOOLEAN DEFAULT FALSE,
    model_version VARCHAR(50),
    model_stage VARCHAR(50),
    
    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(run_id)
);

-- Create index on training_date for faster queries
CREATE INDEX IF NOT EXISTS idx_training_date ON ml_metrics.model_training(training_date DESC);

-- Create index on model_name for faster queries
CREATE INDEX IF NOT EXISTS idx_model_name ON ml_metrics.model_training(model_name);

-- Create index on run_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_run_id ON ml_metrics.model_training(run_id);

-- Grant permissions
GRANT USAGE ON SCHEMA ml_metrics TO airflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml_metrics TO airflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ml_metrics TO airflow;

-- Comment on table
COMMENT ON TABLE ml_metrics.model_training IS 'Historical tracking of ML model training runs and their metrics';
