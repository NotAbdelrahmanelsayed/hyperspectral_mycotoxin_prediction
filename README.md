# Hyperspectral Mycotoxin Prediction

## Video Explanation

A detailed walkthrough of the project's code, methodology, and findings is available via video:

[![Watch the video](https://img.youtube.com/vi/cUAvcI0lOQo/maxresdefault.jpg)](https://youtu.be/cUAvcI0lOQo)


## Project Overview

This repository contains my solution to the following ML Intern task:

> **Objective**: Process hyperspectral imaging data and develop a predictive model for mycotoxin levels (DON concentration) in corn samples.

## Repository Structure

```
notabdelrahmanelsayed-hyperspectral_mycotoxin_prediction/
├── README.md
├── notebook.ipynb
├── report.md
├── requirements.txt
├── data/
└── src/
    ├── evaluation.py
    ├── modeling.py
    └── utils.py
```

## Installation

### Clone the repository

```bash
git clone https://github.com/NotAbdelrahmanelsayed/hyperspectral_mycotoxin_prediction.git
cd hyperspectral_mycotoxin_prediction
```

### Setup Virtual Environment

**Windows:**

```bash
pip install virtualenv
virtualenv venv --python=3.10
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/MacOS**

```bash
pip install virtualenv
virtualenv venv --python=3.10
source venv/bin/activate
pip install -r requirements.txt
```

## Running the project
To run and reproduce the results
1. open the notebook.ipynb
2. run all 

## Methodology

### Data and Preprocessing

- Dataset: 500 samples, 448 spectral bands.
- Target: DON concentration (ppb), initially skewed (7.2).
- Applied log-transform (`log(1+x)`), significantly reducing skewness (skew=-1.61).
- Data split (80/20) performed before scaling with RobustScaler to prevent data leakage.

### Dimensionality Reduction

- PCA:
  - 6 components explained 97% variance; PC1 alone explained 85%.

### Model Evaluation

| Model        | MAE (ppb) | RMSE (ppb) | R² Score |
| ------------ | --------- | ---------- | -------- |
| RF Baseline  | 2763.85   | 10383.99   | 0.61     |
| RF Tuned     | 2857.86   | 11263.00   | 0.55     |
| XGB Baseline | 2953.31   | 9961.00    | 0.65     |
| XGB Tuned    | 3779.58   | 15147.54   | 0.18     |

- Best Model: **XGBoost Baseline** (R²=0.65, RMSE=9961 ppb, MAE=2953 ppb).
- Performance impacted by extreme DON values (>10k ppb).

## Future Exploration
- Alternative transformations: Box-Cox, Quantile Transformer.
- Advanced modeling: CNNs, attention mechanisms, temporal spectral analysis.

