import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method")
        try:
            # Load the raw Tesla stock data
            df = pd.read_csv('notebook/data/tesla_stock_data.csv')
            logging.info('Dataset read into dataframe')

            # Dropping unnecessary columns like 'Open', 'High', 'Low', etc.
            df = df[['Date', 'Close', 'Volume']]  # Keeping only essential features for forecasting
            df['Date'] = pd.to_datetime(df['Date'])  # Convert Date column to datetime format
            
            # Resetting index to ensure Date is a regular column, not the index
            df.reset_index(drop=True, inplace=True)

            # Save raw data with Date as a column
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Train-test split
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42, shuffle=False)  # Time series split

            # Save the train and test data with Date as a column
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()