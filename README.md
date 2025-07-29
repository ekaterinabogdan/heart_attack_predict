```python
# Heart Attack Prediction

## Описание
Проект по предсказанию риска сердечного приступа на данных пациентов. Модель обучена с использованием CatBoostClassifier. Основная цель — максимально точно выявить пациентов с высоким риском.



## Состав проекта
- `heart_attack.ipynb` — исследование и обучение моделей
- `main.py` — FastAPI приложение
- `predict.py` — логика предсказания
- `catboost_model.joblib` — сохранённая модель
- `submission.csv` — предсказания для тестовой выборки
- `requirements.txt` — зависимости

## Входной CSV-файл должен содержать следующие признаки:

Income

Systolic blood pressure

Diastolic blood pressure

BMI

Age

Heart rate

Blood sugar

Cholesterol

Triglycerides

Exercise Hours Per Week

Sedentary Hours Per Day

Physical Activity Days Per Week

Sleep Hours Per Day

Stress Level

Дополнительно создаются лог-преобразованные и производные признаки (log_blood_sugar, log_chol, log_trig, bp_ratio).

## Запуск приложения
1. Установить зависимости:
    pip install -r requirements.txt
2. Запустить FastAPI приложение:
    uvicorn main:app --reload
        
3.Отправить POST-запрос с файлом:
    curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_test_file.csv"

```
