Customer Purchase Prediction — ML Classification

A machine learning project that predicts whether an online shopper will make a purchase, using three classification algorithms: KNN, Naive Bayes, and Logistic Regression.


## Project Overview

Most e-commerce businesses struggle to identify which visitors are likely to buy. This project builds a prediction model trained on real session data — browsing behavior, visitor type, timing, and more — to answer one question:
**Will this customer make a purchase?**
Three models were trained and compared: KNN, Naive Bayes, and Logistic Regression. Each was evaluated using Accuracy, Precision, Recall, and F1 Score to determine the best performer on this imbalanced dataset.


## Dataset

**Source:** [Online Shoppers Purchasing Intention Dataset](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset) — UCI Machine Learning Repository via Kaggle

| Property | Value |
|---|---|
| Rows | 12,330 sessions |
| Columns | 18 features |
| Target | `Revenue` (1 = bought, 0 = didn't buy) |
| Class balance | 84.5% No / 15.5% Yes (imbalanced) |

### Key Features

| Feature | Description |
|---|---|
| `Administrative`, `Informational`, `ProductRelated` | Number of pages visited per category |
| `*_Duration` | Time spent on each page category |
| `BounceRates` | % of visitors who left immediately |
| `ExitRates` | % who exited from that page |
| `PageValues` | Average value of pages visited before purchase |
| `SpecialDay` | Closeness to a special day (Valentine's, etc.) |
| `Month` | Month of the session |
| `VisitorType` | New visitor, returning, or other |
| `Weekend` | Whether the session was on a weekend |


## Project Structure

```
ml_project/
│
├── dataset.py      # Load, explore, and encode the dataset
├── prepare.py      # Train/test split, SMOTE, normalization
├── train.py        # Train KNN, Naive Bayes, Logistic Regression
├── evaluate.py     # Evaluate and compare all models
└── README.md
```


## ML Pipeline

```
1. Load & Explore Data        →  dataset.py
2. Encode Categorical Columns →  dataset.py
3. Train/Test Split (80/20)   →  prepare.py
4. Handle Imbalance (SMOTE)   →  prepare.py
5. Normalize Features          →  prepare.py
6. Train 3 Models              →  train.py
7. Evaluate & Compare          →  evaluate.py
```


## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| KNN | 81.75% | 36.92% | 25.13% | 29.91% |
| Naive Bayes | 64.88% | 26.04% | 68.85% | 37.79% |
| Logistic Regression | **88.32%** | **66.43%** | **49.74%** | **56.89%** |

**Best model: Logistic Regression** — highest accuracy and F1 score, best balance between catching real buyers and avoiding false predictions.


## Key Techniques

| Technique | Why |
|---|---|
| One-Hot Encoding | Convert categorical features (Month, VisitorType) to numbers |
| SMOTE | Fix class imbalance — generate synthetic minority class examples |
| MinMaxScaler | Normalize all features to 0–1 range so no feature dominates |
| Stratified Split | Preserve 85/15 class ratio in both train and test sets |
