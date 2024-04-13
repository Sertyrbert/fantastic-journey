import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузка данных
data = pd.read_csv('train.csv')  # Замените 'your_dataset.csv' на путь к вашему файлу данных

# Замените 'target' на имя вашей целевой переменной
target_column = 'age'

# Предполагается, что целевая переменная находится в столбце с именем 'target'
X = data.drop(target_column, axis=1)  # Призна
y = data[target_column]  # Целевая переменная

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание датасетов для LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Параметры модели LightGBM
params = {
    'boosting_type': 'gbdt',  # Тип градиентного бустинга
    'objective': 'binary',     # Бинарная классификация
    'metric': 'binary_error',  # Метрика для оценки качества
    'num_leaves': 31,          # Максимальное количество листьев в дереве
    'learning_rate': 0.05,     # Скорость обучения
    'feature_fraction': 0.9,   # Доля признаков, используемых при построении каждого дерева
    'bagging_fraction': 0.8,    # Доля обучающих данных, используемых при построении каждого дерева
    'bagging_freq': 5,         # Частота использования bagging
    'verbose': 0               # Уровень вывода информации
}

# Обучение модели LightGBM с ранней остановкой
model = lgb.train(params,
                  train_data,
                  num_boost_round=1000,  # Установите достаточно большое число итераций
                  valid_sets=[test_data],
                  early_stopping_rounds=10)  # Установите количество итераций для ранней остановки

# Предсказание на тестовом наборе данных
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]  # Преобразование вероятностей в бинарные предсказания

# Оценка качества модели
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy}')

# Важность признаков
feature_importance = pd.DataFrame()
feature_importance['feature'] = X.columns
feature_importance['importance'] = model.feature_importance()
print(feature_importance.sort_values(by='importance', ascending=False))