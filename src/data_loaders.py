import pandas as pd
from sklearn.datasets import fetch_20newsgroups

REMOVE=("headers","footers","quotes")

def _to_df(bunch):
    return pd.DataFrame({
        "text":bunch.data,
        "label":[bunch.target_names[i] for i in bunch.target],
    })
def load_data():
    train=fetch_20newsgroups(subset='train',remove=REMOVE)
    test=fecth=fetch_20newsgroups(subset="test",remove=REMOVE)
    return _to_df(train),_to_df(test)    
