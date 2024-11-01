import logging
import os 
from datetime import datetime

# Define the log file name and path
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_dir = os.path.join(os.getcwd(), "logs")  # Create only the logs directory
os.makedirs(logs_dir, exist_ok=True)          # Ensure the logs directory exists

LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)  # Join logs directory with the log file

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging has started")
