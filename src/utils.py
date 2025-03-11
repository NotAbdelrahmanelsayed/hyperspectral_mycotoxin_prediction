import matplotlib.pyplot as plt

def plot_predicted_vs_actual(y_test, y_pred, model_name):
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.7, label=model_name)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel("Actual vomitoxin_ppb (Scaled)")
    plt.ylabel("Predicted vomitoxin_ppb (Scaled)")
    plt.title(f"{model_name} - Actual vs. Predicted")
    plt.legend()
    plt.show()

def store_results(model_name, mae, rmse, r2):
     print(f"{model_name} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}")
     return {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

def plot_metrics(results):
    plt.figure(figsize=(10, 5))
    results.set_index("model_name").plot(kind="bar", figsize=(10, 5), rot=0)
    plt.ylabel("Metric Value")
    plt.title("Model Performance Comparison")
    plt.legend(title="Metrics")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()