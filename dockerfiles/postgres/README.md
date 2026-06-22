# PostgreSQL - Almacenamiento de Métricas ML

Este directorio contiene la configuración de PostgreSQL para el proyecto, incluyendo el almacenamiento de métricas de entrenamiento de modelos ML.

## 📊 Esquema de Base de Datos

### ml_metrics.model_training

Tabla principal que almacena el historial de entrenamientos de modelos con sus métricas y parámetros.

**Columnas principales:**

- `id`: ID autoincremental
- `run_id`: MLflow run ID (único)
- `model_name`: Nombre del modelo (ej: cern_xgboost)
- `experiment_name`: Nombre del experimento
- `training_date`: Fecha y hora del entrenamiento

**Métricas:**
- `train_r2`, `test_r2`: Coeficiente de determinación
- `train_mae`, `test_mae`: Error absoluto medio
- `train_rmse`, `test_rmse`: Raíz del error cuadrático medio
- `test_mape`: Error porcentual absoluto medio

**Optuna:**
- `optuna_best_cv_rmse`: Mejor RMSE de validación cruzada
- `n_trials`: Número de intentos de optimización

**Dataset:**
- `train_size`, `test_size`: Tamaños de conjuntos
- `n_features`: Número de características
- `target_mean`, `target_std`: Estadísticas del objetivo

**Modelo:**
- `model_params`: Parámetros del modelo (JSONB)
- `is_valid`: Si aprobó la validación
- `is_promoted`: Si fue promovido a producción
- `model_version`: Versión del modelo en MLflow
- `model_stage`: Stage del modelo (Production, Staging, etc.)

## 🔍 Consultas Útiles

### Ver últimos 10 entrenamientos

```sql
SELECT 
    training_date,
    model_name,
    test_r2,
    test_mae,
    test_rmse,
    is_valid,
    is_promoted,
    model_version
FROM ml_metrics.model_training
ORDER BY training_date DESC
LIMIT 10;
```

### Ver evolución de métricas

```sql
SELECT 
    training_date::date as fecha,
    ROUND(AVG(test_r2)::numeric, 4) as r2_promedio,
    ROUND(AVG(test_mae)::numeric, 4) as mae_promedio,
    COUNT(*) as num_entrenamientos
FROM ml_metrics.model_training
WHERE model_name = 'cern_xgboost'
GROUP BY training_date::date
ORDER BY fecha DESC;
```

### Ver modelos promovidos a producción

```sql
SELECT 
    model_name,
    model_version,
    training_date,
    test_r2,
    test_mae,
    test_rmse
FROM ml_metrics.model_training
WHERE is_promoted = TRUE
ORDER BY training_date DESC;
```

### Comparar parámetros de mejores modelos

```sql
SELECT 
    run_id,
    training_date,
    test_r2,
    test_mae,
    model_params->>'n_estimators' as n_estimators,
    model_params->>'max_depth' as max_depth,
    model_params->>'learning_rate' as learning_rate
FROM ml_metrics.model_training
WHERE model_name = 'cern_xgboost'
ORDER BY test_r2 DESC
LIMIT 5;
```

### Ver tasa de validación exitosa

```sql
SELECT 
    model_name,
    COUNT(*) as total_entrenamientos,
    SUM(CASE WHEN is_valid THEN 1 ELSE 0 END) as validados,
    ROUND(100.0 * SUM(CASE WHEN is_valid THEN 1 ELSE 0 END) / COUNT(*), 2) as porcentaje_validacion
FROM ml_metrics.model_training
GROUP BY model_name;
```

## 🔗 Conexión desde Python

```python
import psycopg2
import pandas as pd

# Conectar
conn = psycopg2.connect(
    host='localhost',  # o 'postgres' desde dentro de Docker
    port=5432,
    database='airflow',
    user='airflow',
    password='airflow'
)

# Consultar
query = """
SELECT * FROM ml_metrics.model_training
WHERE model_name = 'cern_xgboost'
ORDER BY training_date DESC
LIMIT 10;
"""

df = pd.read_sql(query, conn)
conn.close()

print(df)
```

## 🐳 Acceso desde Docker

Para conectarte a PostgreSQL desde tu máquina local:

```bash
# Usando psql desde el contenedor
docker exec -it postgres psql -U airflow -d airflow

# O desde tu máquina (si tienes psql instalado)
psql -h localhost -p 5432 -U airflow -d airflow
```

Luego puedes ejecutar:
```sql
\dt ml_metrics.*  -- Ver tablas en el esquema
\d ml_metrics.model_training  -- Ver estructura de tabla
```

## 📈 Visualización

Puedes conectar herramientas como:
- **pgAdmin**: http://localhost:5432
- **DBeaver**: Cliente SQL universal
- **Metabase**: Para dashboards
- **Grafana**: Para monitoreo en tiempo real

## 🔄 Mantenimiento

### Limpiar entrenamientos antiguos

```sql
-- Eliminar entrenamientos de más de 6 meses que no fueron promovidos
DELETE FROM ml_metrics.model_training
WHERE training_date < NOW() - INTERVAL '6 months'
AND is_promoted = FALSE;
```

### Backup de métricas

```bash
# Exportar tabla a CSV
docker exec -it postgres psql -U airflow -d airflow \
  -c "COPY ml_metrics.model_training TO STDOUT WITH CSV HEADER" \
  > metrics_backup.csv
```

## 🛠️ Inicialización

Los scripts SQL en este directorio se ejecutan automáticamente al inicializar PostgreSQL por primera vez:

1. `01_mlflow.sql`: Crea la base de datos de MLflow
2. `02_init_metrics_table.sql`: Crea el esquema y tabla de métricas

Si ya tienes PostgreSQL corriendo y quieres agregar la tabla de métricas, ejecuta manualmente:

```bash
docker exec -i postgres psql -U airflow -d airflow < dockerfiles/postgres/init_metrics_table.sql
```
