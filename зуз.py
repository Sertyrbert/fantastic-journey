import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error

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

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Построение нейронной сети
def build_model():
    model = Sequential([
        Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Обучение модели несколько раз
n_epochs = 50
n_runs = 50
losses = []

for i in range(n_runs):
    print(f"Training run {i + 1}/{n_runs}")

    model = build_model()

    # Обучение модели
    history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=32, validation_split=0.2, verbose=0)

    # Оценка модели на тестовом наборе данных
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy for run {i + 1}: {accuracy}")
    losses.append(loss)

# Вывод среднего значения потерь
print(f"Average loss: {np.mean(losses)}")
