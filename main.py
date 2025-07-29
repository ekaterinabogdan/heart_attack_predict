#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from predict import make_prediction

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = make_prediction(temp_file)
    finally:
        os.remove(temp_file)

    return JSONResponse(content=result)







