import numpy as np
from src.config import MODEL_PATH
from src.data_preprocessing import clean_text
from src.utils import load_model

_model=None

def _get_model():
    global _model
    if _model is None:
        _model=load_model(MODEL_PATH)
    return _model

def predict_text(text:str,top_k:int=3):
    model=_get_model()
    proba=model.predict_proba([clean_text(text)])[0]
    top=np.argsort(proba)[::-1][:top_k]
    classes=model.classes_
    return {
        "label":classes[top[0]],
        "confidence":round(float(proba[top[0]]),4),
        "top_k":[{"label":classes[i],"prob":round(float(proba[i]),4)} for i in top],
    }

if __name__=="__main__":
    print(predict_text("The new GPU has 16GB VRAM and supports ray tracing"))
    