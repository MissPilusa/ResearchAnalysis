# Applying Machine Learning Algorithms to Analyse the Impact of Social Media Usage on Productivity

## Project Overview

This project investigates the relationship between social media usage and productivity using machine learning and data analytics techniques. It forms part of a postgraduate research study in Data Analytics.

The study uses the **Social Media vs Productivity** dataset to examine how social media usage, digital behaviour, lifestyle characteristics and workplace-related factors are associated with actual productivity.

Three machine learning algorithms are evaluated: Linear Regression, Random Forest Regression and XGBoost Regression.

## Research Objectives

The main research question is:

**How can machine learning algorithms be applied to analyse the impact of social media usage on productivity?**

The supporting research questions are:

1. Which factors related to social media usage influence productivity the most?
2. Which machine learning algorithms best predict productivity based on digital behaviour?
3. What is the relationship between social media usage and productivity?

## Dataset

**Source:** [Social Media vs Productivity – Kaggle](https://www.kaggle.com/datasets/mahdimashayekhi/social-media-vs-productivity)

The original dataset contains approximately 30,000 observations and 19 variables.

The target variable is actual_productivity_score.

Predictor variables include:

- **Social media and digital behaviour:** Daily social media time, number of notifications, social media platform preference, screen time before sleep, focus application usage and digital wellbeing settings.
- **Lifestyle and behavioural characteristics:** Sleep hours, stress levels, work breaks, coffee consumption and weekly offline hours.
- **Workplace and demographic characteristics:** Job satisfaction, working hours, job type, age and gender.

The variable perceived_productivity_score was excluded from the predictors to reduce the risk of target leakage.

## Technologies and Libraries

The analysis was conducted in Python using Jupyter Notebook within the Anaconda environment.

The main libraries include:

- **Pandas and NumPy:** Data manipulation and numerical analysis.
- **Matplotlib and Seaborn:** Data visualisation.
- **Scikit-learn:** Preprocessing, regression modelling, cross-validation, hyperparameter tuning and evaluation.
- **XGBoost:** Gradient boosting regression.
- **Statsmodels:** Statistical diagnostics.
- **SHAP:** Model interpretability and feature contribution analysis.

## Methodology

### 1. Data Cleaning and Preprocessing

The dataset was examined for missing values, duplicate records, invalid values and inconsistent data types.

Preprocessing included handling missing values, encoding categorical variables and scaling numerical variables where required.

After excluding observations with missing target values, 27,635 records remained for modelling.

The data was divided into 80% training and 20% testing subsets using random_state=42.

### 2. Exploratory Data Analysis

Exploratory data analysis was performed to understand variable distributions and investigate relationships with productivity.

The analysis included:

- Descriptive statistics and frequency distributions.
- Productivity score distribution and outlier analysis.
- Pearson correlation analysis.
- Scatterplots of social media usage and productivity.
- Boxplots comparing categorical digital behaviour variables with productivity.

### 3. Machine Learning Modelling

Three regression algorithms were selected:

**Linear Regression:** An interpretable baseline regression model for examining linear predictive relationships.

**Random Forest Regression:** An ensemble learning algorithm that combines multiple decision trees to capture more complex relationships.

**XGBoost Regression:** A gradient boosting algorithm that builds sequential decision trees to improve predictive accuracy.

Five-fold cross-validation was used to evaluate model performance. GridSearchCV was applied to optimise the Random Forest and XGBoost models.

### 4. Model Evaluation

Model performance was assessed using:

- R2 (Coefficient of Determination)
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)

Training and testing results were compared to assess generalisation and potential overfitting.

## Final Model Performance

| Model | Test R2 | Test MAE | Test MSE | Test RMSE |
|---|---:|---:|---:|---:|
| Linear Regression | 0.710 | 0.790 | 1.044 | 1.022 |
| Random Forest Regression | 0.719 | 0.772 | 1.013 | 1.007 |
| XGBoost Regression | 0.719 | 0.772 | 1.011 | 1.006 |

**XGBoost achieved the lowest test RMSE**, although its performance was very similar to Random Forest Regression.

## Key Findings

The analysis identified several important findings:

1. **Weak direct social media relationships:** Daily social media time, number of notifications and screen time before sleep showed Pearson correlations close to zero with actual productivity.
2. **Job satisfaction was the strongest predictor:** Job satisfaction demonstrated a strong positive correlation with productivity (approximately 0.883) and was prominent in the feature importance analysis.
3. **Social media variables alone had limited predictive value:** A model trained using social media and digital behaviour variables alone produced an R² close to zero.
4. **Machine learning provided useful predictions:** Models incorporating broader behavioural, lifestyle and workplace characteristics explained approximately 71–72% of the variation in productivity scores.
5. **Ensemble models performed slightly better:** Random Forest and XGBoost achieved lower prediction errors than Linear Regression, although the differences were relatively small.

These findings describe predictive associations within the dataset and should not be interpreted as evidence of causation.

## Running the Project

1. Download or clone this repository.
2. Install Python and Jupyter Notebook, preferably through Anaconda.
3. Install the required Python libraries.
4. Download the dataset from the Kaggle link provided above.
5. Place the dataset in the same working directory as the Jupyter Notebook, or adjust the dataset-loading code accordingly.
6. Open the notebook in Jupyter and run the cells sequentially.

## Limitations

The analysis relies on a single dataset and does not clearly distinguish between work-related and personal social media usage.

Furthermore, the study is observational, meaning that correlations and machine learning predictions cannot establish causal relationships.

Independent validation using additional datasets would be necessary to assess the generalisability of the findings.

## Future Work

Future research could extend the study by:

- Investigating work-related and personal social media usage separately.
- Evaluating additional datasets from different industries and populations.
- Exploring non-linear relationships and interaction effects.
- Further validating the strong association between job satisfaction and productivity.
- Assessing model robustness using repeated cross-validation and independent test datasets.
- Applying explainable AI techniques to the final tuned models.

## Conclusion

This project demonstrates how machine learning can be applied to investigate the relationship between social media usage and productivity.

Although the measured social media indicators showed weak direct relationships with productivity, broader workplace-related factors, particularly job satisfaction, provided substantially more predictive information.

The results highlight the importance of considering multiple behavioural and contextual factors when analysing productivity and demonstrate the usefulness of machine learning for identifying predictive patterns in behavioural datasets.

## Dataset Attribution

Dataset: *Social Media vs Productivity*  
Source: Kaggle  
Dataset creator: mahdimashayekhi  
Link: https://www.kaggle.com/datasets/mahdimashayekhi/social-media-vs-productivity 

This repository is intended for academic research and educational purposes.
