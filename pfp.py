import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import lightgbm as lgb
import os

randA = 1
randB = 1000
repeats = 30
old_score = 0
train_File = 'test.csv'
sample_File = 'sample_submission.csv'
model_file = "trained_model2.txt"

#нужно сделать чтобы через простенький интерфейс заполнялись данные repeats & train_File
#вот этот у нужно сравнить с file.txt который получается в ходе работы программы, потому что F1 метрика слишком ебейшая какая-то
#если получится не зафакапить всё, разбить всё на модули
#Определение ненужных столбцов, соответственно их удаление
#Вывод удобной статистики в виде какой-нибудь диограммы или какого-то иного простого к пониманию объекта



with open('all_model_info.txt', 'r+') as file:
    file.truncate(0)
with open('file.txt', 'r+') as file:
    file.truncate(0)
# Загрузка данных
data = pd.read_csv(train_File)
data2 = pd.read_csv(sample_File)
# Удаление ненужных столбцов
data.drop(['slctn_nmbr', 'client_id', 'npo_account_id', 'frst_pmnt_date', 'lst_pmnt_date_per_qrtr'], axis=1,
          inplace=True)

# Заполнение пропущенных значений
data.fillna(0, inplace=True)

# Преобразование всех категориальных столбцов к строковому типу
categorical_cols = ['pmnts_type', 'year', 'quarter', 'gender', 'phone_number', 'email', 'lk', 'assignee_npo',
                    'assignee_ops', 'postal_code', 'region', 'citizen', 'fact_addrss', 'appl_mrkr', 'evry_qrtr_pmnt']
for col in categorical_cols:
    data[col] = data[col].astype(str)

# Кодирование категориальных признаков
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Разделение данных на признаки и целевую переменную
data = data.merge(data2[['churn']], left_index=True, right_index=True)
X = data.drop('churn', axis=1)
y = data['churn']


gt = random.randint(randA, randB)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=gt)

# Масштабирование признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
clf = LGBMClassifier(num_leaves=2, max_depth=4, learning_rate=0.1, n_estimators=repeats)

if os.path.exists(model_file):
        gt = random.randint(randA, randB)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=gt)
        booster = lgb.Booster(model_file=model_file, )
        booster.refit(X_train, y_train)
        #booster.save_model(model_file)
        print('I\'m here!')
        wtw = True
else:
        # Создание и обучение модели
        clf.fit(X_train, y_train)
        clf.booster_.save_model(model_file)
        wtw = False

    # Прогнозирование на тестовой выборке
if wtw:
        y_pred = booster.predict(X_test)
else:
        y_pred = clf.predict(X_train)


param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7]
    }

    # Создание объекта GridSearchCV
grid_search = GridSearchCV(LGBMClassifier(), param_grid, cv=5, n_jobs=4, scoring='f1')

    # Обучение объекта GridSearchCV
grid_search.fit(X_train, y_train)

    # Оценка производительности на отложенном тестовом наборе
f1_score = grid_search.best_score_

print('F1 score:', f1_score.mean())
if f1_score.mean() > old_score:
    booster.save_model(model_file)
    print("Сохранение модели")
    print()
    old_score = f1_score.mean()
    with open('all_model_info.txt', 'a', encoding='utf-8') as f:
        f.writelines("Новый F1 показатель = " + f"{f1_score:.10f}" + '\n')



print(y_pred)
with open('file.txt', 'a', encoding='utf-8') as f:
    for item in y_pred:
        f.write("{:.0f}".format(item) + '\n')



