import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay,accuracy_score,classification_report,f1_score)
from src.config import MODEL_PATH,REPORT_DIR
from src.data_loaders import load_data
from src.data_preprocessing import preprocess_df
from src.utils import get_logger,load_model,save_json

logger=get_logger(__name__)
def evaluate(model,df):
    y_true=df['label']
    y_pred=model.predict(df['text'])
    metrics={
        "accuracy":round(accuracy_score(y_true,y_pred),4),
        "macro_f1":round(f1_score(y_true,y_pred,average='macro'),4),
        "weighted_f1":round(f1_score(y_true,y_pred,average='weighted'),4),
    }
    report=classification_report(y_true,y_pred,output_dict=True)
    print(classification_report(y_true,y_pred))

    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(14,14))
    ConfusionMatrixDisplay.from_predictions(
        y_true,y_pred,ax=ax,xticks_rotation=90,colorbar=False
    )
    plt.tight_layout()
    fig.savefig(REPORT_DIR/"confusion_matrix.png",dpi=120)

    save_json({"metrics":metrics,"per_class":report},"test_results.json")
    return metrics

def main():
    model=load_model(MODEL_PATH)
    _,test_df=load_data()
    test_df=preprocess_df(test_df)
    metrics=evaluate(model,test_df)
    logger.info("Test metrics:%s",metrics)
if __name__=="__main__":
    main() 



