#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from predict import HeartAttackPredictor

app = FastAPI()

# Создаём объект предсказателя один раз при запуске сервиса
predictor = HeartAttackPredictor(model_path='catboost_model.joblib')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = predictor.predict_from_file(temp_file)
    finally:
        os.remove(temp_file)

    return JSONResponse(content=result)







