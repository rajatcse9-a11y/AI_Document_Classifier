import json
import logging
import joblib
from src.config import MODEL_DIR,REPORT_DIR

def get_logger(name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname) | %(message)s",
    )
    return logging.getLogger(name)

def save_model(model,path):
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,path)

def load_model(path):
    return joblib.load(path)

def save_json(obj,filename):
    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    with open(REPORT_DIR/filename,"w") as f:
        json.dump(obj,f,indent=2)