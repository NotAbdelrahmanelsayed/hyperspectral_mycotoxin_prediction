from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import pandas as pd
import numpy as np

def evaluate_model(model, X_test, log_y_test):
    """Evaluate model performance on ORIGINAL scale"""
    # Predict on log scale
    log_y_pred = model.predict(X_test)
    
    # Convert back to original scale
    y_test = np.expm1(log_y_test)  # Convert test data
    y_pred = np.expm1(log_y_pred)  # Convert predictions
    
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "y_test": y_test,  
        "y_pred": y_pred
    }


def log_results(results, model_name):
    """Store and display results"""
    results["model_name"] = model_name
    print(f"{model_name} Results:")
    print(f"MAE: {results['mae']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"R²: {results['r2']:.4f}\n")
    return pd.DataFrame([results])
