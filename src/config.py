from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent.parent
MODEL_DIR=BASE_DIR / "models"
REPORT_DIR=BASE_DIR /"reports"
MODEL_PATH=MODEL_DIR /"doc_classifier.joblib"

RANDOM_STATE=42
VAL_SIZE=0.2
MIN_WORDS=3

MAX_FEATURES=50000
NGRAM_RANGE=(1,2)
MIN_DF=2