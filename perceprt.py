import numpy as np
import torch as to

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation_function(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

# Предположим, что у нас есть некоторые входные данные и соответствующие метки
# Входные данные могут содержать различные параметры, например, количество лет существования НПО, количество членов, финансовые показатели и т. д.
# Метки могут быть 1 или 0, где 1 означает, что договор был сохранен, а 0 - что договор был расторгнут

training_inputs = np.array([[30, 10], [50, 2000], [20, 500], [70, 3000]])
labels = np.array([1, 1, 0, 1])

# Создаем экземпляр перцептрона и обучаем его
perceptron = Perceptron(input_size=2)
perceptron.train(training_inputs, labels)

# Предсказываем вероятность сохранения договора для нового временного интервала
new_data_point = np.array([4, 150])
prediction = perceptron.predict(new_data_point)
print("Вероятность сохранения договора на новом временном интервале:", prediction)