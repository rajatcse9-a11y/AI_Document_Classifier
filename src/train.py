from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import MODEL_PATH,RANDOM_STATE,VAL_SIZE
from src.data_loaders import load_data
from src.feature_engineering import build_vectorizer
from src.data_preprocessing import preprocess_df
from src.utils import get_logger,save_json,save_model


logger=get_logger(__name__)

def get_models():
    return {
        "logger":LogisticRegression(max_iter=1000,C=10),
        "Complement_nb":ComplementNB(alpha=0.3),
        "linear_svc":CalibratedClassifierCV(LinearSVC(),cv=3),

    }

def main():
    train_df,_ =load_data()
    train_df=preprocess_df(train_df)
    logger.info("Train docs after cleaning:%d",len(train_df))
    tr,val=train_test_split(train_df,test_size=VAL_SIZE,stratify=train_df['label'],random_state=RANDOM_STATE,)
    results,best_name,best_score,best_pipe={},None,-1,None
    for name,clf in get_models().items():
        pipe=Pipeline([("tfidf",build_vectorizer()),("clf",clf)])
        pipe.fit(tr['text'],tr['label'])
        preds=pipe.predict(val['text'])
        score= f1_score(val["label"],preds,average="macro")
        results[name]=round(score,4)
        logger.info("%s-> val macro F1: 4f",name,score)

        if score >best_score:
            best_name,best_score,best_pipe=name,score,pipe
    logger.info("Best Model:%s(%.4f)",best_name,best_score)
    final=clone(best_pipe)
    final.fit(train_df['text'],train_df['label'])

    save_model(final,MODEL_PATH)
    save_json({"val_macro_f1":results,"best_model":best_name},"train_results.json")
    logger.info("Model saved: %s",MODEL_PATH)
if __name__=="__main__":
    main()
                

    
