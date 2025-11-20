# Simple Linear Regression – Employee Salary Prediction

## Overview

This repository demonstrates a lab for applying **Simple Linear Regression** to predict an employee's salary based on their years of experience. Using a dataset of 100 employees, the lab guides users through data exploration, model training, and performance evaluation.

## Objectives

- Understand and apply Simple Linear Regression.
- Load and explore a dataset.
- Train a salary prediction model based on years of experience.
- Visualize the results.
- Evaluate model performance using MSE, MAE, and R² Score.

## Dataset Description

The dataset contains two columns:

| Column         | Description                                       |
| -------------- | ------------------------------------------------- |
| **Experience** | Number of years of work experience (1–20 years)   |
| **Salary**     | Annual salary in USD (approx. $30,000 – $120,000) |

## Steps / Procedure

1. **Load and Explore Data:**  
   - Display full dataset, head, and tail.
   - Check shape, summary statistics, and look for missing values.

2. **Data Preprocessing:**  
   - Describe the dataset.
   - Validate there are no missing entries.
   - Split into independent (`Experience`) and dependent (`Salary`) variables.

3. **Train/Test Split:**  
   - Divide the dataset into training and testing sets using `train_test_split` from `sklearn`.

4. **Model Training:**  
   - Fit a Simple Linear Regression model using the training data.
   - Calculate coefficients and intercept.

5. **Prediction & Evaluation:**  
   - Predict on the test set.
   - Calculate error/residuals.
   - Report metrics: RMSE, R² Score.

## Example (Python)

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# Load dataset (replace with your own data loading method)
emp_data = pd.read_csv('data.csv')

# Split the data
X = emp_data[['Experience']]
y = emp_data['Salary']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train the model
reg = LinearRegression()
reg.fit(x_train, y_train)

# Prediction and Evaluation
y_predict = reg.predict(x_test)
rmse = np.sqrt(np.mean((y_test - y_predict) ** 2))
r2 = reg.score(x_test, y_test)
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")
```

## Results

- Typical RMSE ~4687.75
- Typical R² Score ~0.95

## Requirements

- Python
- pandas
- scikit-learn
- numpy

## References

- See the notebook: [`Lab02_Simple_LinearRegression.ipynb`](Lab02_Simple_LinearRegression.ipynb)
- Open in Colab ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)
