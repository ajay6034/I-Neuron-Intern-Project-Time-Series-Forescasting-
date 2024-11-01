import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features, forecast_days):
        try:
            model_path = os.path.join("artifacts", "best_model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # If a preprocessor exists, apply it to the initial input features
            if preprocessor:
                data_scaled = preprocessor.transform(features)
            else:
                data_scaled = features

            # Initialize a list to store predictions
            predictions = []

            # Loop through each forecast day
            for _ in range(forecast_days):
                # Predict for the current day
                pred = model.predict(data_scaled)
                predictions.append(pred[0])  # Append the prediction (assuming it's a single value)

                # Update the features for the next day's prediction
                # Here we assume the model only takes current indicators as features
                # You may need to modify this to fit your specific model and data structure
                new_data = {
                    "SMA_20": [predictions[-1]],  # Use the latest prediction as a new SMA_20
                    "EMA_20": [self.calculate_new_ema(predictions)],  # Update EMA_20 if necessary
                    "RSI": [features['RSI'].iloc[0]],  # Keep RSI the same
                    "Volume": [features['Volume'].iloc[0]]  # Keep Volume the same
                }

                # Convert new data to DataFrame and scale it again
                data_scaled = preprocessor.transform(pd.DataFrame(new_data))

            return predictions
        except Exception as e:
            raise CustomException(e, sys)

    def calculate_new_ema(self, predictions):
        # Placeholder method to calculate updated EMA based on past predictions
        # Adjust according to the desired formula for EMA calculation
        return sum(predictions) / len(predictions) 

class CustomData:
    def __init__(self, sma_20: float, ema_20: float, rsi: float, volume: float, start_date: str = None, forecast_days: int = None):
        # Initialize with user inputs for technical indicators
        self.sma_20 = sma_20
        self.ema_20 = ema_20
        self.rsi = rsi
        self.volume = volume
        self.start_date = start_date
        self.forecast_days = forecast_days

    def get_data_as_data_frame(self):
        try:
            # Create a dictionary with only the features used for training
            custom_data_input_dict = {
                "SMA_20": [self.sma_20],
                "EMA_20": [self.ema_20],
                "RSI": [self.rsi],
                "Volume": [self.volume]
            }
            # Return the dictionary as a DataFrame
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
