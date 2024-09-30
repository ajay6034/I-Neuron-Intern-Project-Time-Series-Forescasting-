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
    transformed_train_data_path: str = os.path.join('artifacts', "transformed_train_data.csv")
    transformed_test_data_path: str = os.path.join('artifacts', "transformed_test_data.csv")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_data_path, test_data_path):
        logging.info("Starting data transformation process")

        try:
            # Load the train and test datasets
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            # Apply Technical Indicators (SMA, EMA, RSI)
            train_df['SMA_20'] = train_df['Close'].rolling(window=20).mean()
            train_df['EMA_20'] = train_df['Close'].ewm(span=20, adjust=False).mean()
            delta = train_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            train_df['RSI'] = 100 - (100 / (1 + rs))

            # Dropping rows with missing values
            train_df.dropna(inplace=True)

            # Test data: Apply the same transformations
            test_df['SMA_20'] = test_df['Close'].rolling(window=20).mean()
            test_df['EMA_20'] = test_df['Close'].ewm(span=20, adjust=False).mean()
            delta = test_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            test_df['RSI'] = 100 - (100 / (1 + rs))

            test_df.dropna(inplace=True)

            # Normalizing the features (SMA, EMA, RSI, Volume)
            scaler = MinMaxScaler()
            feature_cols = ['SMA_20', 'EMA_20', 'RSI', 'Volume']  # Adding Volume for normalization

            train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
            test_df[feature_cols] = scaler.transform(test_df[feature_cols])

            # Saving transformed data
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
    transformer.initiate_data_transformation('artifacts/train.csv', 'artifacts/test.csv')
