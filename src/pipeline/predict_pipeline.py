import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Define paths to model and preprocessor
            model_path = os.path.join("artifacts", "best_model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")  # If you have one

            print("Before Loading the model and preprocessor")

            # Load the model
            model = load_object(file_path=model_path)
            
            # If you have preprocessing steps (scaling, encoding), load the preprocessor
            preprocessor = None
            if os.path.exists(preprocessor_path):
                preprocessor = load_object(file_path=preprocessor_path)
                print("Preprocessor Loaded")

            print("After Loading the model")

            # If preprocessor exists, apply it to the features
            if preprocessor:
                data_scaled = preprocessor.transform(features)
            else:
                data_scaled = features  # If no preprocessing is needed

            # Make prediction
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, sma_20: float, ema_20: float, rsi: float):
        # Initialize with user inputs for technical indicators
        self.sma_20 = sma_20
        self.ema_20 = ema_20
        self.rsi = rsi

    def get_data_as_data_frame(self):
        try:
            # Create a dictionary with only the features used for training
            custom_data_input_dict = {
                "SMA_20": [self.sma_20],
                "EMA_20": [self.ema_20],
                "RSI": [self.rsi],
            }

            # Return the dictionary as a DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
