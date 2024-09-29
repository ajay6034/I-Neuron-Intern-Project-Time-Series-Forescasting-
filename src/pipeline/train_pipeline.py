import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging


class DataPreprocessing:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        try:
            logging.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)

            # Basic cleanup and handling missing values
            df.dropna(inplace=True)
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def apply_technical_indicators(self, df):
        try:
            logging.info("Applying technical indicators (SMA_20, EMA_20, RSI)")

            # Add SMA (Simple Moving Average)
            df['SMA_20'] = df['Close'].rolling(window=20).mean()

            # Add EMA (Exponential Moving Average)
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

            # Add RSI (Relative Strength Index)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Drop missing values caused by technical indicators
            df.dropna(inplace=True)

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def split_data(self, df):
        try:
            logging.info("Splitting data into training and testing sets")

            # Select features and target variable
            X = df[['SMA_20', 'EMA_20', 'RSI']]
            y = df['Close']

            # Train-test split (70-30 split)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

            # Scale the features (if required)
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            return X_train_scaled, X_test_scaled, y_train, y_test, scaler

        except Exception as e:
            raise CustomException(e, sys)


class ModelTraining:
    def __init__(self):
        pass

    def train_models(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Training models")

            # Define models to train
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(),
                'Lasso Regression': Lasso(),
                'Random Forest': RandomForestRegressor(random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'SVR': SVR(),
                'XGBoost': XGBRegressor(random_state=42),
                'K-Nearest Neighbors': KNeighborsRegressor()
            }

            # Track model performance
            model_performance = {}
            best_model = None
            best_model_name = None
            best_r2_score = -float('inf')

            for model_name, model in models.items():
                logging.info(f"Training {model_name}")
                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)
                test_r2 = r2_score(y_test, y_pred_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

                logging.info(f"{model_name} - Test R2: {test_r2}, Test RMSE: {test_rmse}")

                # Track the best model
                if test_r2 > best_r2_score:
                    best_model = model
                    best_model_name = model_name
                    best_r2_score = test_r2

                model_performance[model_name] = {'r2': test_r2, 'rmse': test_rmse}

            logging.info(f"Best model: {best_model_name} with R2 score: {best_r2_score}")
            return best_model, model_performance

        except Exception as e:
            raise CustomException(e, sys)


def main():
    try:
        # Define paths
        data_path = os.path.join('data', 'tesla_stock_data.csv')
        model_path = os.path.join('artifacts', 'best_model.pkl')
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

        # Data Preprocessing
        preprocessing = DataPreprocessing(data_path)
        df = preprocessing.load_data()
        df = preprocessing.apply_technical_indicators(df)
        X_train, X_test, y_train, y_test, scaler = preprocessing.split_data(df)

        # Model Training
        trainer = ModelTraining()
        best_model, model_performance = trainer.train_models(X_train, y_train, X_test, y_test)

        # Save the best model and preprocessor
        logging.info("Saving the best model and preprocessor")
        save_object(model_path, best_model)
        save_object(preprocessor_path, scaler)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
