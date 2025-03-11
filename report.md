## Methodology

### Data and Preprocessing
- Dataset: 500 corn samples, 448 spectral bands.
- Target: DON concentration (ppb), highly skewed initially (skew=7.2).
- Applied log-transform (`log(1+x)`), reducing skewness significantly (skew=-1.61).
- Data split (80/20) performed before scaling with RobustScaler to avoid data leakage.

### Dimensionality Reduction
- PCA:
  - 6 components explained 97% variance; PC1 alone explained 85%.

### Model Evaluation
- Models evaluated: Random Forest and XGBoost.
- Best Model: XGBoost (R²=0.65, RMSE=9961 ppb, MAE=2953 ppb).
- Model performance impacted by extreme DON values (>10k ppb).

### Alternative Transformations for Future Exploration
- Box-Cox.
- Quantile Transformer.

## Key Findings
- XGBoost outperformed Random Forest despite high error rates.
- PCA effectively captured data variance.
- Extreme DON values remained challenging across both models.

## Future Directions
- Implement quantile loss or anomaly detection for handling extreme values.
- Compare wavelength selection against PCA for improved interpretability.
- Evaluate deep learning models such as CNNs or attention mechanisms.
- Consider integrating temporal spectral data if available.

