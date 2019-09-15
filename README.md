# loan_default
This is a machine learning project aimed to predict whether a customer is likely to default on a loan. The project consists of 4 files:
  1. etl.py: getting the tables from MySQL database and performing initial cleaning, processing and feature engineering.
  2. model.py: stacked model of xgboost and random forest with a logistic meta classifier; and save the model using pickle
  3. report.py: predict & report on the results.
  4. main.py: main file to run.
