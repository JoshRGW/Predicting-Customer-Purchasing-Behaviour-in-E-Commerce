# Predicting Customer Purchasing Behaviour in E-Commerce

This repository contains the code developed for the **Customer Purchasing Behaviour Prediction** project, which evaluates the performance of multiple machine learning models for predicting customer purchase decisions in e-commerce. The project implements several classifiers, including Support Vector Machines (SVM), k-Nearest Neighbours (k-NN), Decision Trees, Random Forest, AdaBoost, and Gradient Boosting, while optimising their hyperparameters using grid search.

## Project Overview

The **Customer Purchasing Behaviour Prediction** project aims to compare the performance of various machine learning models to predict whether a customer will make a purchase based on e-commerce data. Each model undergoes training and hyperparameter tuning, with results evaluated using metrics such as **accuracy**, **precision**, **recall**, and **F1 score**. 

### Key Machine Learning Models:
1. **Support Vector Machine (SVC)**: Performs hyperparameter tuning using cross-validation to select the optimal `C` parameter.
2. **k-Nearest Neighbours (k-NN)**: Uses grid search to find the best value of `k`.
3. **Decision Trees**: Explores various `criterion` (Gini or Entropy) and `max_features` settings.
4. **Random Forest**: Evaluates different `n_estimators` and `criterion` combinations to identify the optimal random forest configuration.
5. **AdaBoost Classifier**: Trains on different numbers of estimators to find the best configuration.
6. **Gradient Boosting**: Fine-tunes the `n_estimators` parameter for performance optimisation.

### Contents:
- **Assessment Analysis.py**: The Python script containing the code for model training, hyperparameter tuning, and evaluation.

### License
This project is licensed under the MIT License.
