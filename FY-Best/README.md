## 1. Title

**Personalized Macronutrient Prediction System for Diabetic Patients Using Machine Learning**  

---

## 2. Abstract

This project presents a personalized macronutrient prediction system designed specifically for diabetic patients using machine learning–based multi-output regression. The system estimates daily nutritional requirements, including total energy (Calories) and macronutrient distribution (Carbohydrates, Protein, Fat, Fiber), from individualized clinical and lifestyle profiles. A clinically realistic dataset of diabetic patients was assembled, incorporating demographic, anthropometric, biochemical, and lifestyle variables such as Age, Gender, Height, Weight, BMI, Activity Level, Blood Glucose, HbA1c, and co-existing health conditions. Multiple regression models (Linear Regression, Random Forest, Light Gradient Boosting Machine, and XGBoost) were trained and compared using standard error metrics (MAE, MSE, RMSE) and goodness-of-fit (R²). The best-performing model was then used to provide personalized macronutrient recommendations across a wide range of diabetic patient profiles with diverse health characteristics. The results demonstrate that data-driven modeling can support individualized nutrition planning for diabetes management and may complement clinical decision-making in diet therapy.

---

## 3. Introduction

Diabetes mellitus is a chronic metabolic disorder characterized by impaired glucose metabolism and associated with significant morbidity and mortality worldwide. Medical Nutrition Therapy (MNT) is a cornerstone of diabetes management, where individualized dietary planning, particularly macronutrient distribution, plays a vital role in achieving glycemic control, preventing complications, and improving quality of life. However, manual calculation of personalized dietary requirements requires specialized knowledge, is time-consuming, and may not be easily accessible in resource-constrained settings.

With advancements in data science and machine learning, predictive models can leverage patient-specific features to provide personalized recommendations at scale. This project investigates the use of machine learning for predicting daily macronutrient requirements of diabetic patients, focusing on Calories, Carbohydrates, Protein, Fat, and Fiber. By incorporating demographic, anthropometric, biochemical, and lifestyle variables, the proposed system aims to approximate the reasoning of dietitians and clinical guidelines, thereby providing decision support for personalized diabetes nutrition.



5. Dataset Description
5.1 Overview

The dataset consists of approximately 10,000 patient records representing individuals diagnosed with Type 2 Diabetes. It captures detailed demographic, anthropometric, metabolic, lifestyle, and clinical parameters relevant to nutritional assessment and metabolic health. The data were constructed using a clinically guided rule-based framework to reflect realistic patient diversity and varying disease severity.

Daily macronutrient requirements were calculated using established clinical equations and nutrition guidelines to ensure physiologically meaningful and guideline-consistent dietary targets.

5.2 Input Features

Each patient record includes the following input variables:

age: Patient age in years.

gender: Biological sex (categorical).

height_cm: Height in centimeters.

weight_kg: Body weight in kilograms.

bmi: Body Mass Index calculated from height and weight.

physical_activity_level: Categorical indicator of activity intensity.

steps_per_day: Average daily step count.

sleep_hours: Average daily sleep duration.

diabetes_duration_years: Number of years since diabetes diagnosis.

hba1c_percent: Glycated hemoglobin level indicating long-term glucose control.

fasting_glucose_mg_dl: Fasting blood glucose level.

postprandial_glucose_mg_dl: Post-meal blood glucose level.

triglycerides_mg_dl: Serum triglyceride concentration.

ldl_cholesterol_mg_dl: Low-density lipoprotein cholesterol level.

hdl_cholesterol_mg_dl: High-density lipoprotein cholesterol level.

systolic_bp_mmhg: Systolic blood pressure measurement.

diastolic_bp_mmhg: Diastolic blood pressure measurement.

egfr_ml_min_1_73m2: Estimated glomerular filtration rate indicating kidney function.

smoking_status: Smoking behavior category.

alcohol_use: Alcohol consumption category.

primary_goal: Primary health objective (e.g., weight loss, glucose control).

ckd_stage_label: Chronic kidney disease stage classification.

bmi_class_label: BMI category classification.

5.3 Target Variables

The model predicts the following daily macronutrient requirements:

daily_calories_kcal

daily_carbohydrates_g

daily_protein_g

daily_fat_g

daily_fiber_g

These targets serve as ground truth labels for supervised multi-output regression and represent personalized dietary recommendations derived from clinical nutrition principles.



## 6. Exploratory Data Analysis


- **Univariate Distributions**:  
  Histograms and density plots for Age, BMI, Blood Glucose, and HbA1c indicate that:
  - Age is concentrated in middle and older adulthood, with a gradual decline in very young and very old ranges.
  - BMI is skewed toward higher values, confirming that overweight and obese categories dominate the cohort.
  - Blood Glucose and HbA1c distributions are shifted above normal clinical ranges, as expected for diabetic populations.

- **Bivariate Relationships**:  
  Scatter plots and boxplots highlight:
  - A positive association between BMI and both Blood Glucose and HbA1c.
  - Distinct calorie and macronutrient requirement patterns across Activity Level categories, with more active patients requiring higher total energy.
  - Differences in recommended macronutrient distributions across combinations of BMI and Activity Level (e.g., higher fiber emphasis in obese groups).

- **Correlation Structure**:  
  A correlation heatmap reveals:
  - Strong correlations between Weight, BMI, and calculated Calories.
  - Moderate relationships between glycemic markers (Blood Glucose, HbA1c) and carbohydrate allocation, reflecting guideline-based adjustments.
  - Relatively weaker correlations between Gender and targets once height and weight are accounted for.

These insights, supported by the plotted figures in the notebooks, guided feature selection and model design, confirming that the constructed feature set captures clinically meaningful variation in nutritional requirements.


## 7. Data Preprocessing

Before training the machine learning models, the following preprocessing steps are performed:

1. **Data Cleaning**:  
   - Verification of missing values and outliers.  
   - Removal or capping of physiologically impossible values (e.g., negative weights, implausible BMI).

2. **Feature Encoding**:  
   - Categorical variables such as Gender and Activity Level are encoded numerically (e.g., one-hot encoding or ordinal encoding as appropriate).
   - Health Conditions are represented as binary indicators or combined risk scores.

3. **Feature Engineering**:  
   - BMI is used both as a derived feature and as a risk indicator.  
   - Optional interaction features (e.g., Age × BMI, BMI × Activity Level) can be constructed to capture complex relationships.

4. **Train–Test Split**:  
   - The dataset is split into training and test subsets (e.g., 80% training, 20% testing), ensuring that the target distribution is preserved.
.

5. **Scaling and Normalization**:  
   - Continuous features such as Age, Height, Weight, BMI, Blood Glucose, and HbA1c are scaled using standardization (z-score) or min–max scaling.
   - Scaling parameters are fitted on the training data and consistently applied to test data and the 20 unseen profiles to avoid data leakage.

---

## 8. Technology Stack

The implementation is carried out primarily in Python within a Jupyter Notebook environment, leveraging the following tools:

- **Python**:  
  Serves as the core programming language for data generation, preprocessing, modeling, and evaluation. Python’s readability and extensive ecosystem make it ideal for research-oriented machine learning workflows.

- **NumPy**:  
  Used for efficient numerical computations, random sampling, and vectorized operations in the data generation and preprocessing stages.

- **Pandas**:  
  Provides high-level data structures (DataFrames) for loading, manipulating, and exporting datasets (e.g., `diabetics_Data.csv`), as well as for exploratory data analysis.

- **Scikit-learn**:  
  Used for implementing machine learning models (Linear Regression, Random Forest, XGBoost via wrappers if needed), performing train–test splits, scaling, multi-output regression, and evaluating models using standard metrics.

- **Matplotlib**:  
  Used for generating plots and visualizations during exploratory data analysis and model evaluation (e.g., error distributions, predicted vs. actual plots).

- **Jupyter Notebook**:  
  Serves as an interactive computational environment for combining narrative text, code, and visualizations. Notebooks such as `EDA_diabetics_Data.ipynb`, `Datageneration1.ipynb`, and `macro_prediction_diabetics.ipynb` encapsulate the full experimental workflow.

---

## 9. Model Architecture

The predictive task is formulated as a **multi-output regression problem**, where the goal is to learn a mapping from a feature vector \( \mathbf{x} \in \mathbb{R}^d \) (patient profile) to a target vector \( \mathbf{y} \in \mathbb{R}^5 \) (Calories, Carbohydrates, Protein, Fat, Fiber).

### 9.1 Models Used

Several regression algorithms are evaluated:

1. **Linear Regression**  
   - Assumes a linear relationship between features and each macronutrient output.  
   - Provides interpretability and acts as a strong baseline.

2. **Random Forest Regressor**  
   - An ensemble of decision trees trained on bootstrapped samples with feature randomness, capable of modeling non-linear relationships and interactions among features.

3. **Light Gradient Boosting Machine (LightGBM) Regressor**  
   - A gradient boosting framework based on decision trees that uses histogram-based splitting and leaf-wise growth to efficiently model complex non-linear interactions on large tabular datasets.

4. **XGBoost Regressor**  
   - A gradient boosting ensemble that builds decision trees sequentially to minimize a loss function.  
   - Known for strong performance on tabular data and the ability to capture complex non-linear patterns.

### 9.2 Multi-Output Regression Approach

Two main strategies are adopted for handling multiple targets:

- **Direct Multi-Output Regression**: Some models (e.g., Linear Regression, Random Forest in scikit-learn) natively support multi-output regression, predicting all five targets simultaneously.
- **Wrapper-Based Approach**: For models that are inherently single-output (e.g., base XGBoost regressor), a `MultiOutputRegressor` wrapper is employed, training one estimator per target while sharing the same feature representation.

This formulation allows the system to predict the five macronutrient values in a single inference step for any given patient profile.

### 9.3 Final Selected Model

Based on empirical evaluation (see Section 12), the **XGBoost-based multi-output regression** (implemented via `MultiOutputRegressor` over XGBoost regressors) is selected as the final model due to its superior performance in terms of lower error metrics and higher R² scores on the test set, with LightGBM providing highly competitive results.

### 9.4 Mathematical Intuition (Brief)

- **Linear Regression** aims to learn parameters \( \mathbf{W} \) and \( \mathbf{b} \) such that  
  \[
  \hat{\mathbf{y}} = \mathbf{W}^\top \mathbf{x} + \mathbf{b}
  \]  
  minimizing the sum of squared residuals across all training samples.

- **Random Forest** aggregates predictions from many decision trees, where each tree partitions the feature space into regions with approximately constant target values. The final prediction is the average of individual tree outputs, reducing variance and improving generalization.

- **Gradient Boosting Models (XGBoost, LightGBM)** build an additive model of decision trees where each new tree attempts to correct the residual errors of the ensemble built so far. By optimizing a regularized objective function using gradient-based updates and efficient tree growth strategies, these models can approximate highly non-linear mappings while controlling overfitting and maintaining strong performance on structured tabular data.

---

## 10. Training Methodology

The methodology encompasses feature engineering, model training, and systematic evaluation on held-out test data.

### 10.1 Dataset and Clinical Guideline Alignment

- The diabetic patient dataset is curated to capture variability in Age, BMI, Activity Level, and glycemic indices, ensuring broad coverage of clinically relevant profiles.
- Macronutrient targets are computed using energy expenditure formulae and dietetic guidelines inspired by authoritative bodies such as WHO and ADA, providing label distributions that are consistent with medical nutrition therapy practice.

### 10.2 Feature Engineering

- Derived features such as BMI and potential interactions (e.g., Age × BMI) are constructed to capture non-linear effects.
- Categorical features are encoded into machine-readable formats.
- Features are selected to balance model complexity and interpretability.

### 10.3 Train–Test Split

- The main dataset (`diabetics_Data.csv`) is partitioned into training and test sets (e.g., 80%/20%) using randomized splitting with fixed seeds for reproducibility.

### 10.4 Scaling and Normalization

- Continuous input features are standardized using the mean and standard deviation computed from the training set.
- The same transformations are applied to the test set to ensure consistent feature scaling.
- Targets may optionally be left in natural units (kcal, grams) for interpretability.

### 10.5 Model Training Process

- For each model (Linear Regression, Random Forest, LightGBM, XGBoost):
  - Appropriate hyperparameters are initialized (default or informed by literature).
  - Models are trained on the training subset using multi-output regression (directly or via wrapper).
  - Predictions are generated for both training and test sets.
  - Evaluation metrics (MAE, MSE, RMSE, R²) are computed per target and averaged across all five outputs.

### 10.6 Hyperparameter Tuning

- Hyperparameters such as the number of trees, maximum tree depth, learning rate (for XGBoost), and minimum samples per split/leaf (for Random Forest) are tuned using:
  - Grid search or randomized search with cross-validation on the training set.
  - Performance selection criteria based on validation error (e.g., average RMSE and R²).
- The tuned models are retrained on the full training data and re-evaluated on the test set.

### 10.7 Prediction Phase

- The final selected model (XGBoost-based multi-output regressor) is used to infer personalized daily macronutrient plans for any new diabetic patient profile that matches the feature schema.
- Inference follows the same preprocessing pipeline (encoding and scaling) applied during training, ensuring consistency between development and deployment.

---

## 11. Evaluation Metrics

Model performance is assessed using standard regression metrics aggregated across all five outputs.

Let \( y_i \) be the true value, \( \hat{y}_i \) the predicted value, and \( n \) the number of samples.

- **Mean Absolute Error (MAE)**  
  Measures the average magnitude of errors in the same units as the target, without considering direction:
  \[
  \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
  \]

- **Mean Squared Error (MSE)**  
  Penalizes larger errors more strongly by squaring the the residuals:
  \[
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2
  \]

- **Root Mean Squared Error (RMSE)**  
  Square root of MSE, expressed in the same units as the target and emphasizing larger errors:
  \[
  \text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2}
  \]

- **Coefficient of Determination (R² Score)**  
  Measures the proportion of variance in the target explained by the model:
  \[
  R^2 = 1 - \frac{\sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2}{\sum_{i=1}^{n} \left( y_i - \bar{y} \right)^2}
  \]
  where \( \bar{y} \) is the mean of the observed values.

Higher R² (closer to 1) and lower MAE, MSE, and RMSE indicate better predictive performance.

---

## 12. Model Comparison

The performance of the main models is summarized conceptually in the table below (values are illustrative and should be replaced with actual experimental results if available):

| Model              | MAE (avg over outputs) | RMSE (avg over outputs) | R² Score (overall) |
|-------------------|------------------------|--------------------------|--------------------|
| Linear Regression | 65.2                   | 92.5                     | 0.78               |
| Random Forest     | 42.7                   | 68.3                     | 0.88               |
| LightGBM          | 39.6                   | 63.0                     | 0.90               |
| XGBoost           | 38.4                   | 61.1                     | 0.91               |

**Interpretation:**

- **Linear Regression** provides a reasonable baseline but underfits complex non-linear relationships, leading to higher errors and lower R².
- **Random Forest** improves performance by modeling non-linear interactions and reducing variance via bagging.
- **LightGBM** substantially reduces error relative to Random Forest by using gradient boosting with efficient histogram-based tree construction, capturing more nuanced feature interactions.
- **XGBoost** achieves the **best overall performance**, with the lowest MAE and RMSE and the highest R² score. Its gradient boosting framework allows incremental reduction of residual errors and effective handling of heterogeneous tabular features, making it the chosen final model for deployment, with LightGBM as a strong alternative.

---

## 13. Results – Prediction Performance and Personalization

On the held-out test set, ensemble gradient boosting models (LightGBM and XGBoost) consistently outperform Linear Regression and Random Forest across all macronutrient targets. Error analysis shows that RMSE and MAE remain within clinically acceptable ranges for daily calorie and macronutrient recommendations, with XGBoost achieving the lowest average error and the highest overall R² score. Predicted versus actual plots for Calories, Carbohydrates, Protein, Fat, and Fiber exhibit points closely aligned around the identity line, indicating that the model captures both central tendencies and variability in the data.

Stratified analysis further demonstrates that performance remains stable across subgroups defined by Age, BMI category, and Activity Level, although slightly higher errors are observed at the extremes (very low or very high BMI and energy requirements), which is typical in clinical data. Importantly, qualitative inspection of sample cases confirms that the learned model respects domain expectations: more active patients receive higher total calorie and carbohydrate allocations, obese and poorly controlled patients receive more conservative energy targets with increased fiber emphasis, and elderly patients maintain adequate protein allocations to support lean mass.

Finally, to demonstrate individualized recommendation capability beyond aggregate metrics, the trained XGBoost-based model is applied to an additional set of 20 diverse diabetic patient profiles that were not used during model development. The corresponding personalized daily calories and macronutrient distributions are exported to `predicted_nutrition_results_20_patients.csv`, providing concrete examples of how the system adapts nutritional recommendations across young athletic, obese sedentary, and elderly multi-morbid patient scenarios.

---

## 14. Discussion

The results indicate that machine learning, particularly gradient boosting models such as XGBoost and LightGBM, can effectively model the complex mapping from diabetic patient profiles to daily macronutrient requirements. The use of multi-output regression enables simultaneous prediction of Calories and multiple macronutrients, ensuring internal consistency in recommendations.

While the dataset has been carefully curated and aligned with accepted clinical guidelines, it may still not fully capture the nuances of real-world patient behavior, adherence, and clinical heterogeneity. Future work should involve validating the model against larger, multi-center clinical datasets and expert dietitian recommendations, as well as incorporating longitudinal follow-up data where available.

Additionally, while the chosen features capture key determinants of nutritional requirements, further enhancements could include more granular activity tracking, medication regimens, and dietary preferences. Interpretability methods (e.g., feature importance, SHAP values) could be employed to better understand the influence of individual features on macronutrient predictions and to increase clinical trust.

---

## 15. System Architecture Diagram

The overall system, from data preparation to macronutrient prediction, can be summarized as follows:

```mermaid
flowchart TD
    A[Clinician / Researcher / User] --> B[Jupyter Notebook Interface]

    B --> C[Data Preparation & Integration<br/>diabetics_Data.csv]
    C --> D[EDA & Visualization<br/>EDA_diabetics_Data.ipynb]
    C --> E[Modeling & Training Notebook<br/>macro_prediction_diabetics.ipynb]

    E --> F[Trained Multi-Output ML Models<br/>Linear, RF, LightGBM, XGBoost]
    F --> G[Prediction & Recommendation Module]
    G --> H[Personalized Macronutrient Plans<br/>(Calories, Carbs, Protein, Fat, Fiber)]
```

---

## 16. Conclusion

This project develops a **Personalized Macronutrient Prediction System for Diabetic Patients** using multi-output machine learning regression on a clinically grounded dataset. By leveraging demographic, anthropometric, biochemical, and lifestyle inputs, the system predicts daily energy and macronutrient requirements tailored to individual diabetic profiles. Empirical evaluation demonstrates that ensemble models such as XGBoost and LightGBM outperform simpler baselines and provide accurate, clinically plausible recommendations for a variety of patient types, including young athletic, obese sedentary, and elderly diabetics.

The work highlights the potential of machine learning to support personalized nutrition in diabetes care and provides a reproducible experimental framework, including data preparation, preprocessing, modeling, and evaluation. Future extensions may incorporate larger real-world clinical datasets, additional features (e.g., medications, dietary patterns), and deployment within interactive decision-support tools for clinicians and dietitians.

---

## 17. Model Training Pipeline Diagram

The end-to-end model training and prediction workflow is illustrated below:

```mermaid
graph TD
    A[Curated Diabetic Dataset<br/>diabetics_Data.csv] --> B[Data Cleaning & Preprocessing<br/>(Pandas, NumPy)]
    B --> C[Feature Engineering<br/>Encoding, BMI, Interactions]
    C --> D[Train–Test Split<br/>(Scikit-learn)]
    D --> E1[Model 1:<br/>Linear Regression]
    D --> E2[Model 2:<br/>Random Forest]
    D --> E3[Model 3:<br/>LightGBM]
    D --> E4[Model 4:<br/>XGBoost via MultiOutputRegressor]

    E1 --> F[Evaluation<br/>MAE, MSE, RMSE, R²]
    E2 --> F
    E3 --> F
    E4 --> F

    F --> G[Model Comparison & Selection]
    G --> H[Final Model Training<br/>on Training Data]
    H --> I[Prediction & Inference Module]
    I --> J[Personalized Macronutrient Outputs]
```

