import pandas as pd

# Загрузка данных из CSV файла
data = pd.read_csv('суд.csv')

# Выбор определенного столбца по его названию
column_name = 'churn'
column_data = data[column_name]

# Подсчет уникальных значений в столбце
unique_values = column_data.value_counts()

with open('file2.txt', 'r') as file:
    lines = file.readlines()
    count_zeros = 0
    count_ones = 0

    # Перебираем каждую строку
    for line in lines:
        # Удаляем лишние пробелы и символы переноса строки, если они есть
        cleaned_line = line.strip()

        # Если строка равна '0', увеличиваем счетчик нулей
        if cleaned_line == '0':
            count_zeros += 1
        else:
            count_ones += 1

# Вывод результата
print(unique_values)
print(count_zeros)
print(count_ones)