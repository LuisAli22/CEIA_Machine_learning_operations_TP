# FastAPI ML Model Service

Servicio de API REST para servir modelos de Machine Learning con integración a MLflow.

## 🚀 Características

- ✅ Carga automática de modelos desde MLflow al iniciar
- ✅ Validación de datos con Pydantic
- ✅ Manejo robusto de errores
- ✅ Health checks
- ✅ Recarga de modelos en caliente
- ✅ Documentación automática (Swagger/OpenAPI)
- ✅ CORS configurado

## 📋 Endpoints

### General
- `GET /` - Mensaje de bienvenida
- `GET /docs` - Documentación interactiva (Swagger UI)
- `GET /redoc` - Documentación alternativa (ReDoc)

### Health
- `GET /health` - Estado del servicio y modelo

### Model
- `GET /model/info` - Información del modelo cargado
- `POST /model/reload` - Recargar modelo desde MLflow

### Prediction
- `POST /predict` - Hacer predicciones

## 🔧 Configuración

### Variables de Entorno

El directorio `dockerfiles/fastapi/` contiene un archivo `.env.example` con todas las configuraciones disponibles:

```bash
# MLflow Settings
MLFLOW_TRACKING_URI=http://mlflow:5001
MODEL_NAME=my_model
MODEL_VERSION=1
MODEL_STAGE=Production

# AWS/MinIO Settings
AWS_ACCESS_KEY_ID=minio
AWS_SECRET_ACCESS_KEY=minio123
AWS_ENDPOINT_URL_S3=http://s3:9000
MLFLOW_S3_ENDPOINT_URL=http://s3:9000
```

**Nota**: En el entorno de Docker Compose, estas variables se configuran automáticamente en el archivo `docker-compose.yaml`.

### Cargar Modelo por Versión

```env
MODEL_NAME=my_model
MODEL_VERSION=2
MODEL_STAGE=None
```

### Cargar Modelo por Stage

```env
MODEL_NAME=my_model
MODEL_STAGE=Production
```

## 📝 Uso

### Ejemplo de Predicción con Datos CERN

El modelo entrenado con el DAG `process_cern_data` predice la **masa invariante (M)** a partir de 9 características:

**Características de entrada:**
- `pt1`, `pt2` - Momento transversal de cada electrón
- `E_total` - Energía total del sistema
- `delta_eta`, `delta_phi`, `delta_R` - Características angulares
- `pt_product` - Producto de momentos transversales
- `pt_ratio` - Ratio de momentos transversales
- `is_os` - Opposite sign charge (1 si tienen carga opuesta, 0 si no)

**Variable predicha:**
- `M` - Masa invariante del sistema de dos electrones

#### Ejemplo de Request

```bash
curl -X POST "http://localhost:8800/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "pt1": 76.4,
        "pt2": 37.3,
        "E_total": 122.0,
        "delta_eta": 0.02,
        "delta_phi": 2.80,
        "delta_R": 2.80,
        "pt_product": 2849.72,
        "pt_ratio": 2.05,
        "is_os": 1
      }
    ]
  }'
```

#### Respuesta

```json
{
  "predictions": [90.5],
  "model_name": "cern_xgboost",
  "model_version": "1",
  "timestamp": "2024-01-01T00:00:00"
}
```

La predicción `90.5` representa la masa invariante estimada en GeV/c².

**Nota**: Para obtener datos reales del dataset procesado:

```bash
# Desde el contenedor de Airflow, obtener una fila de ejemplo
docker exec -it airflow-scheduler python -c "
import polars as pl
df = pl.read_parquet('/opt/airflow/dags/data/dielectron.parquet')
features = ['pt1', 'pt2', 'E_total', 'delta_eta', 'delta_phi', 'delta_R', 'pt_product', 'pt_ratio', 'is_os']
print(df.select(features).head(1).to_dicts()[0])
"
```

### Ejemplo Genérico de Predicción

### Ejemplo Genérico de Predicción

Para otros modelos, el formato general es:

```bash
curl -X POST "http://localhost:8800/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"feature1": 1.0, "feature2": 2.0},
      {"feature1": 3.0, "feature2": 4.0}
    ],
    "return_probabilities": true
  }'
```

### Respuesta con Probabilidades

```json
{
  "predictions": [0, 1],
  "probabilities": [[0.8, 0.2], [0.3, 0.7]],
  "model_name": "my_model",
  "model_version": "1",
  "timestamp": "2024-01-01T00:00:00"
}
```

### Health Check

```bash
curl http://localhost:8800/health
```

### Información del Modelo

```bash
curl http://localhost:8800/model/info
```

### Recargar Modelo

```bash
curl -X POST http://localhost:8800/model/reload
```

## 🐳 Docker

El servicio se ejecuta automáticamente con `docker-compose`:

```bash
docker compose --profile all up
```

## 🧪 Testing

Para probar el servicio localmente:

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
python main.py
```

## 📚 Documentación Adicional

Una vez iniciado el servicio, visita:
- Swagger UI: http://localhost:8800/docs
- ReDoc: http://localhost:8800/redoc

## 🔒 Seguridad

- Validación de entrada con Pydantic
- Manejo seguro de excepciones
- Logging de errores sin exponer información sensible
- CORS configurable
