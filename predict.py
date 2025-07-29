#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import pandas as pd
import numpy as np
import joblib


def add_new_features(df):
    df = df.copy()
    df['log_blood_sugar'] = np.log1p(df['Blood sugar'])
    df['log_trig'] = np.log1p(df['Triglycerides'])
    df['log_chol'] = np.log1p(df['Cholesterol'])
    df['bp_ratio'] = df['Systolic blood pressure'] / (df['Diastolic blood pressure'] + 1)
    return df

FEATURES = [
    'Income', 'bp_ratio', 'Exercise Hours Per Week', 'log_trig', 'log_chol', 'BMI',
    'Systolic blood pressure', 'Sedentary Hours Per Day', 'Heart rate', 'Age',
    'Diastolic blood pressure', 'log_blood_sugar', 'Stress Level',
    'Physical Activity Days Per Week', 'Sleep Hours Per Day'
]


BEST_THRESHOLD = 0.33  


model = joblib.load('catboost_model.joblib')


def make_prediction(data_path: str):
    df = pd.read_csv(data_path)

   
    ids = df['id'] if 'id' in df.columns else np.arange(len(df))
    
    df = df.drop(columns=['Unnamed: 0', 'id'], errors='ignore')
    df = add_new_features(df)
    df = df[FEATURES]

    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= BEST_THRESHOLD).astype(int)

    return {'predictions': preds.tolist(), 'probabilities': probs.tolist(), 'ids': ids.tolist()}

