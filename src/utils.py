import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def vomitoxin_distribution(df):
    plt.figure(figsize=(8,6))
    sns.histplot(df['vomitoxin_ppb'], bins=50)
    plt.title("Distribution of Vomitoxin_ppb")
    plt.xlabel("vomitoxin_ppb")
    plt.ylabel("Frequency")
    plt.show()


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

def plot_pca_analysis(pca):
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, 11), explained_variance, marker='o', linestyle='--')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Analysis")
    plt.show()

def plot_pca_projection(df_pca, comp_num=1, comp_num_2=2):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=df_pca[f'PC{comp_num}'], y=df_pca[f'PC{comp_num_2}'], hue=df_pca['vomitoxin_ppb'], palette='coolwarm')
    plt.title("PCA Projection of Spectral Data")
    plt.xlabel(f"Principal Component {comp_num}")
    plt.ylabel(f"Principal Component {comp_num_2}")
    plt.show()