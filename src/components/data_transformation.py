import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    raw_data_path: str = os.path.join('artifacts', "raw_data.csv")
    transformed_train_data_path: str = os.path.join('artifacts', "transformed_train_data.csv")
    transformed_test_data_path: str = os.path.join('artifacts', "transformed_test_data.csv")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self):
        logging.info("Starting data transformation process")
        
        try:
            # Load the raw stock price data from CSV
            df = pd.read_csv("artifacts/transformed_data.csv")
            logging.info(f"Loaded dataset from {self.transformation_config.raw_data_path}")

            # Apply Technical Indicators (like SMA, EMA, RSI)
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

            # Adding RSI (Relative Strength Index)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Dropping rows with missing values due to rolling window calculations
            df.dropna(inplace=True)

            # Splitting data into train and test sets based on date (Time series split)
            train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
            logging.info(f"Data split into training and testing sets")

            # Normalize the training and test data (except for the target variable 'Close')
            scaler = MinMaxScaler()
            feature_cols = ['SMA_20', 'EMA_20', 'RSI']  # Features to normalize

            train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
            test_df[feature_cols] = scaler.transform(test_df[feature_cols])
            logging.info("Normalized the training and test datasets")

            # Save the transformed training and test datasets
            os.makedirs(os.path.dirname(self.transformation_config.transformed_train_data_path), exist_ok=True)
            train_df.to_csv(self.transformation_config.transformed_train_data_path, index=False)
            test_df.to_csv(self.transformation_config.transformed_test_data_path, index=False)
            logging.info(f"Transformed data saved at {self.transformation_config.transformed_train_data_path} and {self.transformation_config.transformed_test_data_path}")

            return self.transformation_config.transformed_train_data_path, self.transformation_config.transformed_test_data_path

        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    transformer = DataTransformation()
    transformer.initiate_data_transformation()
