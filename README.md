## Artificial Intelligence Macroeconomic Stock Prediction

Predicited daily stock market behavior based on a number of macroeconomic indicators, employing advanced data science techniques and machine learning models using Python, all done within the 12 week AI4ALL Ignite accelerator.

## Problem Statement

The challenge is to leverage AI and machine learning to extract actionable insights from historical stock market data, enabling informed and accurate predictions in a volatile and unpredictable environment.

## Key Results

1. 54.2% Accuracy— 4+ percentage points above random guessing.
2. Random Forest Model — Outperformed Logistic Regression in classification and regression.
3. Top Predictors — VIX (18.5%), Fed rates (14.2%), yield curve (11.8%).
4. Industry-Level R² — 3.5% explained variance matches professional standards.
5. Valid Testing — Time-series split (2015–2021 train, 2022–2025 test) with no data leakage.
6. Economic Alignment — Feature rankings match financial theory.
7. Ensemble Power — Captured non-linear market patterns missed by linear models.
8. Balanced Performance — Precision 56.4%, recall 57.8%.
9. Beats Benchmarks — Outperformed technical analysis and random strategies.

## Methodologies 

To accomplish this, we researched which macroeconomic indicators are most relevant and influence stock market shifts. We used the FRED (Federal Reserve Economic Data) API and yfinance dataset for this project. The data was analyzed and visualized using pandas, numpy, matplotlib, and seaborn. 
The models used were Linear + Logistic Regression and Random Forest models, both decided upon based on what seemed to be the best fit for the project goals. To make accurate predictions, the model was trained and tested with a 70/30 split. 

## Data Sources 

yfinance dataset: https://pypi.org/project/yfinance/
FRED API: https://fredaccount.stlouisfed.org/apikeys

## Technologies Used 

1. pandas
2. numpy
3. matplotlib.pyplot
4. seaborn
5. yfinance
6. fredapi
7. datetime
8. warnings
9. train\_test\_split
10. GridSearchCV
11. TimeSeriesSplit
12. RandomForestClassifier
13. RandomForestRegressor
14. LogisticRegression
15. LinearRegression
16. StandardScaler
17. classification_report
18. confusion_matrix
19. mean_squared_error
20. r2_score
21. SimpleImputer


## Authors 

- Aleena Siddiqui
  - Email: sleena654@gmail.com
