import re
import pandas as  pd
from src.config import MIN_WORDS


URL_RE=re.compile(r"http\S+|www\.\www\.\S+")
EMAIL_RE=re.compile(r"\S+@\S+")
NON_ALPHA_RE=re.compile(r"[^a-z\s]")
SPACE_RE=re.compile(r"\s+")

def clean_text(text:str)->str:
    text=text.lower()
    text=URL_RE.sub(" ",text)
    text=EMAIL_RE.sub(" ",text)
    text=NON_ALPHA_RE.sub(" ",text)
    text=SPACE_RE.sub(" ",text).strip()
    return text

def preprocess_df(df:pd.DataFrame)->pd.DataFrame:
    df=df.copy()
    df['text']=df['text'].apply(clean_text)
    df=df[df['text'].str.split().str.len()>=MIN_WORDS]
    return df.reset_index(drop=True)
    

