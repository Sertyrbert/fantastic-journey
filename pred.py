import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier
import lightgbm as lgb
import os

train_File = 'суд.csv'
model_file = "trained_model2.txt"

# Загрузка данных
data = pd.read_csv(train_File)

# Удаление ненужных столбцов
data.drop(['slctn_nmbr', 'client_id', 'npo_account_id', 'frst_pmnt_date', 'lst_pmnt_date_per_qrtr'], axis=1, inplace=True)

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

# Масштабирование признаков
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Загрузка ранее обученной модели
if os.path.exists(model_file):
    booster = lgb.Booster(model_file=model_file)
else:
    print("Модель не найдена, убедитесь, что она была обучена ранее.")

# Прогнозирование на данных
if 'booster' in locals():
    y_pred = booster.predict(X)
    print(y_pred)
    with open('file2.txt', 'w', encoding='utf-8') as f:
        for item in y_pred:
            f.write("{:.0f}".format(item) + '\n')