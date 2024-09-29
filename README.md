# Tesla Stock Prediction Project

## Overview
This project focuses on predicting Tesla stock prices using various machine learning models. The models are trained using historical stock data and technical indicators such as Simple Moving Average (SMA), Exponential Moving Average (EMA), and Relative Strength Index (RSI). The predictions are made using a Flask web application that takes user input for technical indicators and provides the predicted stock price.

## Features
- Data ingestion and transformation scripts for processing stock data.
- Application of machine learning models including Linear Regression, Random Forest, Gradient Boosting, and more.
- Integration of Flask web application for stock price prediction.
- Model training and evaluation based on R2 score and RMSE.

## Files and Structure

### 1. `Stock_data_ingestion.py`
- **Purpose**: Handles the ingestion of raw stock data and splits it into training and test datasets.
- **Key Operations**:
  - Loads the Tesla stock data from a CSV file.
  - Splits the data into train and test sets with a 70-30 ratio.
  - Saves the split data for further transformations.
  
### 2. `Stock_data_transformation.py`
- **Purpose**: Applies technical indicators like SMA, EMA, and RSI to the stock data and transforms it for model training.
- **Key Operations**:
  - Computes `SMA_20`, `EMA_20`, and `RSI` for the given stock prices.
  - Preprocesses the data by handling missing values and scaling features.
  - Splits the transformed data into training and testing sets for model building.

### 3. `Stock_model_trainer.py`
- **Purpose**: Trains multiple machine learning models on the transformed data and selects the best-performing model.
- **Key Operations**:
  - Trains models like Linear Regression, Ridge Regression, Random Forest, etc.
  - Evaluates models using R2 score and RMSE metrics.
  - Saves the best model and scaler for deployment.

### 4. `EDA-Model_training.ipynb`
- **Purpose**: An exploratory data analysis (EDA) notebook that includes visualizations and initial analysis of the stock data, followed by model training.
- **Key Operations**:
  - Visualizes stock trends and technical indicators.
  - Performs exploratory data analysis (EDA) to identify data patterns.
  - Trains models within the notebook for preliminary evaluation.

### 5. `app.py`
- **Purpose**: Flask web application that serves the model for real-time stock price predictions based on user input.
- **Key Operations**:
  - Collects user input for `SMA_20`, `EMA_20`, and `RSI`.
  - Loads the trained model and scaler for making predictions.
  - Displays the predicted Tesla stock price on the web interface.

 ## Graphs 
  ![image](https://github.com/user-attachments/assets/27ac9c5e-41a2-41b9-be50-84c02932c07f)

  ![image](https://github.com/user-attachments/assets/96d65425-2bc3-4a9e-9509-c2ca6039c5f9)

  ![image](https://github.com/user-attachments/assets/bc6c4461-ce16-4c2a-b480-f5a83164112c)

  ![image](https://github.com/user-attachments/assets/b8607b2a-6a99-4652-b7a6-7c49b8fb1766)

  ![image](https://github.com/user-attachments/assets/3bc08cb6-c44e-4819-9335-b1e4fa7fbbae)






