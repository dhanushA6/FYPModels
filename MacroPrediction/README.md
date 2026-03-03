# Macro Prediction Model Implementation Details

## Overview
This document outlines the core implementation details of a machine learning pipeline developed for macro prediction. The pipeline encompasses data preprocessing, selection and evaluation of various regression models, training, and a mechanism for making predictions using saved models. The primary goal is to predict multiple macroeconomic indicators.

## Model Architecture
The system evaluates and utilizes several regression models suitable for predicting continuous target variables. For scenarios involving multiple output targets, specific models are wrapped to handle this complexity effectively. The models explored include:
-   **Linear Models:**
    -   Ridge Regression (`sklearn.linear_model.Ridge`)
    -   ElasticNet Regression (`sklearn.linear_model.ElasticNet`)
-   **Ensemble Models:**
    -   Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`)
    -   Extra Trees Regressor (`sklearn.ensemble.ExtraTreesRegressor`)
    -   XGBoost Regressor (`xgboost.XGBRegressor`) - utilized with `sklearn.multioutput.MultiOutputRegressor` for multi-target prediction.
    -   LightGBM Regressor (`lightgbm.LGBMRegressor`) - utilized with `sklearn.multioutput.MultiOutputRegressor` for multi-target prediction.

Each model is configured with a fixed random seed to ensure reproducibility of results during training and evaluation.

## Training Methodology

### Data Preparation
The input dataset is partitioned into training and testing sets using `sklearn.model_selection.train_test_split`. This separation ensures that models are evaluated on unseen data, providing an unbiased assessment of their generalization capabilities.

### Cross-Validation
A `kfold_compare_models` function orchestrates a KFold cross-validation process. This method divides the training data into `n_splits` (e.g., 5) distinct folds. For each fold:
1.  A model pipeline is constructed, integrating data preprocessing steps with the chosen regression model.
2.  The pipeline is trained on the training subsets of the folds.
3.  Predictions are generated on the validation subset of the fold.
4.  Performance metrics—R-squared (R2), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE)—are calculated for each target variable, providing a comprehensive evaluation of model performance across different data partitions.

### Model Training
Upon completion of the cross-validation and selection of optimal models, the final chosen models are trained on the complete training dataset. This step leverages the entire available training data to maximize the model's learning capacity before deployment.

## Prediction Mechanism
The system incorporates a robust mechanism for loading trained models and performing predictions on new, unseen data points.
1.  **Model Loading:** A `load_model` utility function is used to deserialize saved model pipelines (stored as `.pkl` files) using `joblib`.
2.  **Single Sample Prediction:** The `predict_single` function takes a loaded model and a dictionary representing a new data sample. This sample is converted into a structured format (e.g., a Pandas DataFrame), and the model then generates predictions for the multiple target macroeconomic indicators.

## Model Persistence
Trained model pipelines are saved to a designated `model_registry` directory. Each model is serialized using `joblib.dump`, ensuring that the entire pipeline—including preprocessing steps and the trained model—can be precisely reloaded for future predictions without retraining. This facilitates efficient deployment and consistent inference.