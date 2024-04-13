import tkinter as tk
from tkinter import *
from tkinter import messagebox

def count():
   dataset = (dataset_tf.get())
   repits = int(repits_tf.get())


   if repits == 1:
      answer = ('Дохуя')
      messagebox.showinfo('Результат', f'Вероятность закрытия счета: {answer}' )
   if repits == 0:
      answer = ('Нихуя')
      messagebox.showinfo('Результат', f'Вероятность закрытия счета: {answer}')

window = Tk() #Создаём окно приложения.
window.title("Расчет вероятности") #Добавляем название приложения.


window.geometry('400x300')

frame = Frame(
   window, #Обязательный параметр, который указывает окно для размещения Frame.
   padx = 10, #Задаём отступ по горизонтали.
   pady = 10 #Задаём отступ по вертикали.
)
frame.pack(expand=True) #Не забываем позиционировать виджет в окне. Здесь используется метод pack. С помощью свойства expand=True указываем, что Frame заполняет весь контейнер, созданный для него.

repits_lb = Label(
   frame,
   text="Введите Повторы  "
)
repits_lb.grid(row=3, column=1)

dataset_lb = Label(
   frame,
   text="Введите DataSet ",
)
dataset_lb.grid(row=4, column=1)

repits_tf = Entry(
   frame, #Используем нашу заготовку с настроенными отступами.
)
repits_tf.grid(row=3, column=2)

dataset_tf = Entry(
   frame,
)
dataset_tf.grid(row=4, column=2, pady=5)

cal_btn = Button(
   frame, #Заготовка с настроенными отступами.
   text='Рассчитать', #Надпись на кнопке.
   command=count
)
cal_btn.grid(row=5, column=2) #Размещаем кнопку в ячейке, расположенной ниже, чем наши надписи, но во втором столбце, то есть под ячейками для ввода информации.
window.mainloop()


