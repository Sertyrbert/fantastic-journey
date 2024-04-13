import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import os

nk = 1.59447899

# Загрузка данных
data = pd.read_csv('суд.csv')

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
X = data.drop('churn', axis=1)
y = data['churn']

gt = random.randint(1, 100)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=gt)

# Масштабирование признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model_file = "trained_model2.txt"
clf = LGBMClassifier()
for i in range(5):

    if os.path.exists(model_file):
        gt = random.randint(1, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=gt)
        booster = lgb.Booster(model_file=model_file)
        booster.refit(X_train, y_train)
        #booster.save_model(model_file)
        print('I\'m here!')
        wtw = True
    else:
        # Создание и обучение модели
        clf.fit(X_train, y_train)
        clf.booster_.save_model(model_file)
        wtw = False

    # Оценка модели с использованием метрики fair
    def fair_loss(y_true, y_pred):
        c = 0.2
        penalty = np.abs(y_true - y_pred)
        loss = np.mean(np.log(penalty + c))
        return loss

    scorer = make_scorer(fair_loss, greater_is_better=False)
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring=scorer)

    nn = int(np.mean(scores))
    if nn < nk:
        booster.save_model(model_file)
        nk = nn
    print(f'Потери fair: {np.mean(scores)}')

    # Прогнозирование на тестовой выборке
    if wtw:
        y_pred = booster.predict(X_test)
    else:
        y_pred = clf.predict(X_train)

#1.5943872861676442 стартовое значение
#1.594351724714977
#1.594453417949637
#1.594453417949637
#1.5944796212499683
print(y_pred)
with open('file.txt', 'a', encoding='utf-8') as f:
    data = y_pred
    f.writelines(str(data))



