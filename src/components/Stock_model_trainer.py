import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', "best_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_data_path, test_data_path):
        logging.info("Starting model training")

        try:
            # Load train and test data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            # Separate features and target
            X_train = train_df[['SMA_20', 'EMA_20', 'RSI']]
            y_train = train_df['Close']
            X_test = test_df[['SMA_20', 'EMA_20', 'RSI']]
            y_test = test_df['Close']

            # Define models
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

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)
                test_r2 = r2_score(y_test, y_pred_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

                model_performance[model_name] = {
                    'test_r2': test_r2,
                    'test_rmse': test_rmse
                }

                logging.info(f"{model_name} test_r2: {test_r2}, test_rmse: {test_rmse}")

            # Select the best model based on R2 score
            best_model_name = max(model_performance, key=lambda x: model_performance[x]['test_r2'])
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with test R2 score: {model_performance[best_model_name]['test_r2']}")

            # Save the best model
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info(f"Best model saved at {self.model_trainer_config.trained_model_file_path}")

            return best_model_name, model_performance[best_model_name]

        except Exception as e:
            logging.error("Error in model training")
            raise CustomException(e, sys)

if __name__ == "__main__":
    trainer = ModelTrainer()
    best_model_name, best_model_performance = trainer.initiate_model_training('artifacts/transformed_train_data.csv', 'artifacts/transformed_test_data.csv')
    print(f"Best Model: {best_model_name}, Performance: {best_model_performance}")
