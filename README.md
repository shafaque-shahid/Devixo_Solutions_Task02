# Machine Learning Model Comparison

## Project Overview

This project predicts students' **Performance Index** using machine learning regression models. It compares three baseline models and two hyperparameter-tuned models to identify the best-performing approach.

## Dataset

- **Dataset:** [Student Performance (Multiple Linear Regression)](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)
- **Initial records:** 10,000
- **Features:** 5
- **Target:** `Performance Index`

### Features

- `Hours Studied`
- `Previous Scores`
- `Extracurricular Activities`
- `Sleep Hours`
- `Sample Question Papers Practiced`

The `Extracurricular Activities` column was encoded numerically (`Yes = 1`, `No = 0`).

## Data Preprocessing

The following steps were performed:

1. Inspected the dataset and checked for missing values.
2. Identified and removed 127 duplicate rows.
3. Encoded the categorical extracurricular-activities feature.
4. Split the data into training and testing sets using an 80/20 split (`random_state=42`).
5. Standardized the input features using `StandardScaler`, fitting the scaler on the training set only.

After removing duplicates, the dataset contained **9,873 records**. The training set contained 7,898 records and the test set contained 1,975 records.

## Models Evaluated

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Tuned Decision Tree Regressor (GridSearchCV)
5. Tuned Random Forest Regressor (GridSearchCV)

GridSearchCV used 5-fold cross-validation with negative mean squared error as the scoring metric for the tree-based models.

### Best Hyperparameters

**Tuned Decision Tree**
- `max_depth`: 10
- `min_samples_leaf`: 4
- `min_samples_split`: 10
- Best cross-validation RMSE: approximately 2.4871

**Tuned Random Forest**
- `max_depth`: 10
- `min_samples_leaf`: 2
- `n_estimators`: 100
- Best cross-validation RMSE: approximately 2.2369

## Test Results

Lower MAE, MSE, and RMSE indicate smaller prediction errors; a higher R² indicates a better fit on the test set.

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 1.6470 | 4.3059 | 2.0751 | 0.9884 |
| Decision Tree | 2.4196 | 9.2662 | 3.0440 | 0.9751 |
| Random Forest | 1.8975 | 5.6320 | 2.3732 | 0.9849 |
| Tuned Decision Tree | 1.9802 | 6.1916 | 2.4883 | 0.9834 |
| Tuned Random Forest | 1.8085 | 5.1586 | 2.2713 | 0.9861 |

## Conclusion

Linear Regression achieved the best overall test performance among the five model configurations, with the lowest MAE, MSE, and RMSE and the highest R² score. It also had the shortest training time among the three baseline models. Hyperparameter tuning improved the test performance of both tree-based models compared with their untuned versions, but neither surpassed Linear Regression.

Based on these results, **Linear Regression is the recommended final model** for this dataset.

## Visualizations

- R² score comparison
- Error metric comparison across all five models

## Tools and Libraries

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Google Colab

## How to Run

1. Download the dataset from the Kaggle link above.
2. Open the project notebook in Google Colab.
3. Upload the dataset or configure the notebook to access it.
4. Run the notebook cells in order, from preprocessing through model evaluation and visualization.

## Project Scope

This is a regression exercise for predicting the dataset's Performance Index. Results describe performance on the specified test split and should not be interpreted as proof of performance on every student population.
