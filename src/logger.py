import logging
import os 
from datetime import datetime

LOF_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOF_FILE)
os.makedirs(logs_path,exist_ok=True)

LOF_FILE_PATH=os.path.join(logs_path,LOF_FILE)

logging.basicConfig(

    filename=LOF_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,

)

if __name__=="__main__":
    logging.info("Logging has started")