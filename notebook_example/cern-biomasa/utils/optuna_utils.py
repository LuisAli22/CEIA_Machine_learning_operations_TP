import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from utils.config import RANDOM_STATE


def get_random_forest_objective(X, y, cv):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 10, 40),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'n_jobs': -1,
            'random_state': RANDOM_STATE
        }

        model = RandomForestRegressor(**params)
        neg_mse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        return  np.sqrt(-neg_mse_scores.mean())
    return objective


def get_xgboost_objective(X, y, cv):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'n_jobs': -1,
            'random_state': RANDOM_STATE,
            'tree_method': 'hist'
        }

        model = xgb.XGBRegressor(**params)
        neg_mse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        return np.sqrt(-neg_mse_scores.mean())

    return objective