import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Загрузка данных
data = pd.read_csv('train.csv')

print(data)

data.columns = data.columns.str.replace(' ', '_').str.lower()

# Создание экземпляра LabelEncoder
label_encoder = LabelEncoder()

# Применение Label Encoding к столбцам client_id и npo_account_id
data['client_id'] = label_encoder.fit_transform(data['client_id'])
data['npo_account_id'] = label_encoder.fit_transform(data['npo_account_id'])
data['quarter'] = label_encoder.fit_transform(data['quarter'])

# Преобразование дат в числовой формат (количество дней с начала 2000 года)
data['frst_pmnt_date'] = (pd.to_datetime(data['frst_pmnt_date']) - pd.Timestamp('2000-01-01')).dt.days
data['lst_pmnt_date_per_qrtr'] = (pd.to_datetime(data['lst_pmnt_date_per_qrtr']) - pd.Timestamp('2000-01-01')).dt.days

# Преобразование region с использованием One-Hot Encoding
data = pd.get_dummies(data, columns=['region'])


# Удаление столбца с индексом, если он есть
if 'Unnamed: 0' in data.columns:
    data.drop('Unnamed: 0', axis=1, inplace=True)

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.drop('churn', axis=1)
y = data['churn']

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Подготовка данных для LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Определение параметров модели LightGBM
params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 15,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 15,
    'lambda_l2': 15,
    'verbose': 100,

}

# Обучение модели LightGBM
num_round = 1000
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])

# Предсказание вероятности ухода из НПФ за полгода
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
auc = roc_auc_score(y_test, y_pred)
print(f'AUC на тестовом наборе: {auc}')

# Пример вывода причины ухода человека из НПФ за полгода
feature_importance = pd.DataFrame()
feature_importance['feature'] = X.columns
feature_importance['importance'] = bst.feature_importance()
feature_importance = feature_importance.sort_values(by='importance', ascending=False)
print(feature_importance)