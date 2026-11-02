from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_text

app=FastAPI(title="Document Classsification API")

class Doc(BaseModel):
    text:str
    top_k:int=3

@app.get("/")
def home():
    return {"status":"ok"}

@app.post("/predict")
def predict(doc:Doc):
    return predict_text(doc.text,doc.top_k)