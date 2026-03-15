# Ensemble Learning Techniques

A collection of notebooks exploring ensemble learning methods — from boosting and bagging to voting and stacking — implemented using scikit-learn and XGBoost.

---

## Notebooks Overview

### 1. AdaBoost (`adaboost.ipynb`)
Adaptive Boosting using Decision Tree stumps as base learners. Combines multiple weak classifiers sequentially, each focusing on previously misclassified samples.

- Base estimator: Decision Tree (`max_depth=1`)
- Estimators: 200
- Accuracy: **77.3%**
- Includes GridSearchCV pipeline for hyperparameter tuning (n_estimators, learning_rate, max_depth)

---

### 2. Heterogeneous Ensemble (`ensemble_heterogenous.ipynb`)
Combines different types of models together — covers both Voting and Stacking approaches.

**Voting Classifier** (LR + SVC + Decision Tree)
- Accuracy: **83.3%**

**Voting Regressor** (Linear Regression + Decision Tree + SVR)
- R² Score: **0.809**

**Stacking Classifier** (LR + SVC + DT → meta: Logistic Regression)
- Accuracy: **88.0%**

**Stacking Regressor** (LR + DT + SVR)
- R² Score: **~1.0** *(near-perfect on synthetic data)*

---

### 3. Gradient Boosting (`gradient_boosting.ipynb`)
Sequential boosting where each tree corrects the residual errors of the previous one.

**Regressor** (n_estimators=200, lr=0.05, max_depth=3)
- R² Score: **0.9196**

**Classifier** (n_estimators=150, lr=0.1, max_depth=3)
- Synthetic classification dataset (500 samples, 20 features)

---

### 4. XGBoost Classification (`XGBoost_Classification.ipynb`)
Extreme Gradient Boosting — optimized and regularized implementation of gradient boosting.

- n_estimators=100, max_depth=3, learning_rate=0.1
- Accuracy: **88.7%**
- Precision / Recall / F1: **~0.89** across both classes

---

## Results Summary

| Notebook | Task | Model | Score |
|---|---|---|---|
| AdaBoost | Classification | AdaBoost (DT stumps) | 77.3% acc |
| Heterogeneous | Classification | Voting (LR+SVC+DT) | 83.3% acc |
| Heterogeneous | Classification | Stacking (LR+SVC+DT) | 88.0% acc |
| Heterogeneous | Regression | Voting Regressor | R²: 0.809 |
| Gradient Boosting | Regression | GradientBoostingRegressor | R²: 0.9196 |
| XGBoost | Classification | XGBClassifier | 88.7% acc |

---

## How to Run

```bash
pip install scikit-learn xgboost jupyter
jupyter notebook
```

Open any `.ipynb` file and run all cells.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

## Author

**Amit Parmar**
GitHub: [@paramramit305-a11y](https://github.com/paramramit305-a11y)

> Part of my AIML learning journey — ML → Deep Learning → AI Engineer 🚀
