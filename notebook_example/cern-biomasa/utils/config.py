import os

# Base de directorios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
# Paths específicos
RAW_DATA_FILE_PATH = os.path.join(DATASETS_DIR, 'raw', 'dielectron.csv')
PROCESSED_FILE_PATH = os.path.join(DATASETS_DIR, 'processed', 'dielectron.parquet')
RANDOM_FOREST_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'invariant_mass_model_random_forest.pkl')
XGBOOST_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'invariant_mass_model_xgboost.pkl')
LOGO_FIUBA_PATH = os.path.join(BASE_DIR, 'figures', 'logoFIUBA.png')
RANDOM_STATE = 42
TEST_SIZE = 0.2
# Lista de las 9 variables que REALMENTE importan para predecir la masa M
FEATURES_FINALES = [
    'pt1', 'pt2',          # Magnitud del momento (Polar)
    'E_total',            # Energía total
    'delta_eta',           # Diferencia angular (Calculada)
    'delta_phi',           # Diferencia angular corregida (Calculada)
    'delta_R',             # Distancia angular total (Calculada)
    'pt_product',          # Producto cinemático (Calculada)
    'pt_ratio',            # Simetría (Calculada)
    'is_os',                # Interruptor de carga (Calculada)
    'M'                    # Masa invariante (Target)
]
def render_physics_table(df, caption):
    return (df.head()
     .style
     .set_properties(**{'background-color': '#161e2e', 'color': '#e2e8f0', 'border-color': '#2d3748'})
     .format(precision=4)
     .highlight_max(axis=0, color='#2d4a3e')
     .set_caption(caption)
     )