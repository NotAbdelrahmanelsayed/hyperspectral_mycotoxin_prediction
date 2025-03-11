from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def train_baseline_model(X_train, y_train, model_type='rf'):
    """Train baseline model"""
    if model_type == 'rf':
        model = RandomForestRegressor(random_state=2025)
    elif model_type == 'xgb':
        model = XGBRegressor(random_state=2025)
    model.fit(X_train, y_train)
    return model

def tune_random_forest(X_train, y_train):
    """Random Forest hyperparameter tuning"""
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    }
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=2025),
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def tune_xgboost(X_train, y_train):
    """XGBoost hyperparameter tuning"""
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1],
        "colsample_bytree": [0.8, 1]
    }
    random_search = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=2025),
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        random_state=2025,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_