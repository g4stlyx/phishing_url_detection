# Phishing URL Detection Using Data Mining and Machine Learning

## Overview
This project aims to build a robust machine learning system to detect phishing URLs. Using the PhiUSIIL Phishing URL Dataset, I implemented an end-to-end data mining pipeline including data cleaning, exploratory data analysis (EDA), feature engineering, and modeling with six different algorithms.

## Dataset
- **Name**: PhiUSIIL Phishing URL Dataset (https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
- **Source**: UCI Machine Learning Repository / Kaggle
- **Size**: 235,795 samples, 56 features
- **Target**: `label` (1: Phishing, 0: Legitimate)

## Project Structure
```
.
├── dataset/                # Contains the raw CSV dataset
├── docs/                   # Project documentation and reports
├── outputs/                # Generated models (.pkl) and visualization plots (.png)
├── tools/                  # Source code for the project
│   ├── preprocess_data.py      # Data cleaning and preprocessing
│   ├── train_models.py         # Model training script
│   ├── evaluate_models.py      # Evaluation and plotting script
│   └── investigate_leakage.py  # Script used to detect data leakage
├── README.md               # Project documentation
└── TODO.md                 # Task tracking
```

## Installation
Ensure you have Python 3.12+ installed. Install the required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

### 0. Install the Dataset

Install the <a href="https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset">dataset</a> and put it into the "data" folder.

### 1. Preprocessing
Clean the data, handle outliers, and remove leakage features.
```bash
python tools/preprocess_data.py
```

### 2. Training
Train the machine learning models (Decision Tree, Naive Bayes, SVM, MLP, Random Forest, Logistic Regression).
```bash
python tools/train_models.py
```

### 3. Evaluation
Generate performance metrics, ROC curves, and confusion matrices.
```bash
python tools/evaluate_models.py
```

## Methodology
1.  **Data Cleaning**: Removed metadata columns (`URL`, `Domain`, etc.) and duplicates.
2.  **Leakage Removal**: Identified and removed `URLSimilarityIndex` which had a 100% correlation with the target.
3.  **Preprocessing**: Applied outlier clipping and StandardScaler normalization.
4.  **Modeling**: Trained 6 models using `RandomizedSearchCV` for hyperparameter tuning.

## Results
The models achieved high accuracy, with Random Forest and MLP performing best.

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| MLP (ANN) | 99.99% | 1.0000 |
| Random Forest | 99.99% | 1.0000 |
| SVM (Linear) | 99.99% | 1.0000 |
| Naive Bayes | 95.06% | 0.9829 |

*Note: The high accuracy is driven by distinct structural features (e.g., `LineOfCode`, `NoOfExternalRef`) that strongly differentiate phishing sites in this dataset.*

## Key Findings
- **Data Leakage**: The feature `URLSimilarityIndex` was a dead giveaway for phishing sites and was removed to ensure realistic evaluation.
- **Structural Features**: Phishing sites in this dataset are characterized by being simple HTML forms with few lines of code and external references, compared to complex legitimate sites.
