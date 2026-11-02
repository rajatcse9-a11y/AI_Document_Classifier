from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import MAX_FEATURES,NGRAM_RANGE,MIN_DF

def build_vectorizer():
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        stop_words="english",
        sublinear_tf=True,
    )