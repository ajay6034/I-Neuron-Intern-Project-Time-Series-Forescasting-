import os
import pickle
import sys
from src.exception import CustomException
from src.logger import logging

file_path = "artifacts/best_model.pkl"
def save_object(file_path, obj):
    try:
        # Ensure the directory for the file path exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the object to the specified file path
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at {file_path}")
    
    except Exception as e:
        logging.error("Error in saving object")
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        # Load the object from the specified file path
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        logging.error(f"Error in loading object from {file_path}")
        raise CustomException(e, sys)
