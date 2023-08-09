# Customer Credit Card Purchase Prediction using Multivariate Regression

## Introduction

Welcome to the Customer Credit Card Purchase Prediction project! This project focuses on developing accurate regression models to predict customer credit card purchases based on a well-structured dataset containing relevant customer data related to credit card transactions.

## Dataset Description

The dataset used in this analysis is sourced from [Kaggle](https://www.kaggle.com/datasets/mahnazarjmand/customer-segmentation) and contains essential customer attributes such as 'BALANCE,' 'PURCHASES_FREQUENCY,' and 'CREDIT_LIMIT,' which can significantly influence purchase decisions. The dataset is preprocessed to handle missing values using appropriate imputation techniques. The goal is to explore the relationships between these attributes and create regression models to predict customer purchases accurately.

## Project Outline

The project is structured as follows:

1. Data Preprocessing
   - Handle missing values using mean and median imputation
   - Drop irrelevant columns

2. Exploratory Data Analysis (EDA)
   - Visualize data distributions, correlations, and insights
   - Identify key features influencing purchase behavior

3. Multivariate Regression Models
   - Implement various regression algorithms:
     - Linear Regression
     - Decision Tree Regression
     - Random Forest Regression
     - Gradient Boosting Regression
     - Support Vector Regression (SVR)
     - Ridge Regression
     - Lasso Regression
     - ElasticNet Regression
   - Evaluate each model's performance using Mean Squared Error (MSE) and R-squared values

4. Model Evaluation and Comparison
   - Compare regression models based on MSE and R-squared
   - Select the best-performing model for predicting customer purchases

5. Conclusion and Practical Implications
   - Summarize findings and insights from the analysis
   - Discuss the practical applications of the predictive model for businesses

## Getting Started

To replicate and explore this project on your local machine, follow these steps:

1. Clone this repository to your local machine.
2. Install the required libraries and dependencies using the provided `requirements.txt` file.
3. Open the Jupyter Notebook file `Customer_Credit_Purchase_Prediction.ipynb` to access the complete project code, explanations, and visualizations.

## Usage

This project serves as a valuable resource for businesses aiming to predict customer credit card purchases accurately. By understanding the relationships between customer attributes and purchase behavior, businesses can make informed marketing strategies and improve decision-making.

## Further Information

For detailed code explanations, insights, and visualizations, refer to the Jupyter Notebook file `Customer_Credit_Purchase_Prediction.ipynb` provided in this repository.

---
*Note: This README provides a high-level overview of the project. For detailed code, data, and insights, refer to the accompanying Jupyter Notebook and other project files.*
