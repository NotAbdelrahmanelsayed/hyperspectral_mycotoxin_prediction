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
| Model        | MAE (ppb) | RMSE (ppb) | R² Score |
|--------------|-----------|------------|----------|
| RF Baseline  | 2763.85   | 10383.99   | 0.61     |
| RF Tuned     | 2857.86   | 11263.00   | 0.55     |
| XGB Baseline | 2953.31   | 9961.00    | 0.65     |
| XGB Tuned    | 3779.58   | 15147.54   | 0.18     |

- Best Model: XGBoost Baseline (R²=0.65, RMSE=9961 ppb, MAE=2953 ppb).
- Model performance impacted by extreme DON values (>10k ppb).

### Alternative Transformations for Future Exploration
- Box-Cox.
- Quantile Transformer.

## Key Findings
- XGBoost Baseline outperformed other models.
- PCA effectively captured data variance.
- Extreme DON values remained challenging for all models.

## Future Directions
- Implement quantile loss or anomaly detection for handling extreme values.
- Compare wavelength selection against PCA for improved interpretability.
- Evaluate deep learning models such as CNNs or attention mechanisms.
- Consider integrating temporal spectral data if available.
