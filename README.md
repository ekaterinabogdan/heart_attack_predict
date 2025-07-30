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

## Описание классов и методов
Класс HeartAttackPredictor
Предназначен для загрузки модели и выполнения предсказаний на новых данных.

__init__(self, model_path: str, threshold: float = 0.33)
Загружает обученную модель из файла.

__init__(model_path: str)
Загружает модель из файла.

add_new_features(df: pd.DataFrame) -> pd.DataFrame

Добавляет признаки:

Логарифмы: log_blood_sugar, log_trig, log_chol
Соотношение давлений: bp_ratio

predict_from_file(data_path: str) -> dict
Загружает CSV, применяет трансформации и возвращает:

predictions: метки 0/1

probabilities: вероятности

ids: ID пациентов


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

Удалённый доступ (через Render или другой хостинг):
curl -X POST "https://heart-attack-predict-duhc.onrender.com/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_test_file.csv"
```
