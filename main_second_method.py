# pip install numpy matplotlib

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Данные из задачи (вставить свои!)
default_y = np.array([45, 25, 48, 52, 54, 51, 59, 60, 62, 69, 72, 78, 76, 80, 82, 85, 81, 90, 93])
default_x = np.array([30, 35, 31, 38, 41, 48, 50, 55, 51, 58, 60, 59, 65, 73, 78, 71, 79, 80, 81])


# Отображение заданной матрицы
def display_matrix(matrix, title):
    """
    Параметры:
    matrix - Исходная матрица
    title - Название матрицы
    """

    calculations_tab_frame = tk.Frame(master=calculations_tab)
    calculations_tab_frame.pack(fill=tk.X)

    # Название матрицы
    label_title = ttk.Label(
        calculations_tab_frame, text=title, font=("Arial", 18, "bold")
    )
    label_title.grid(row=0, column=0, pady=3)

    # Дополнительный фрейм для удобства вывода матрицы
    calculations_tab_matrix_frame = tk.Frame(
        master=calculations_tab_frame)
    calculations_tab_matrix_frame.grid(row=1, column=0, pady=5, sticky=tk.W)

    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            # Пояснение - :.3f позволяет вывести число только с тремя знаками после запятой, не используя round()
            label = tk.Label(
                calculations_tab_matrix_frame,
                text=f"{val:.2f}",
                width=8,
            )
            label.grid(row=i, column=j, padx=1, pady=2)


# Создание полей ввода данных
def create_data_entries():
    try:
        num_entries = int(entry_data_input_tab.get())
        if num_entries <= 0:
            label_error.config(text="Введите число больше, чем 0")
            return

    except ValueError:
        label_error.config(text="Введите число")
        return
    # Очищаем поле с текстом об ошибке
    label_error.config(text="")

    # Очищаем старые элементы 2 фрейма вкладки получения данных
    for widget in data_input_tab_frame2.winfo_children():
        widget.destroy()

    # Списки для хранения ссылок на Entry матрицы входных значений
    global entries_x, entries_y
    entries_x = []
    entries_y = []

    ttk.Label(data_input_tab_frame2, text="Y").grid(row=0, column=0, padx=5, pady=5)
    ttk.Label(data_input_tab_frame2, text="X").grid(row=0, column=1, padx=5, pady=5)

    for i in range(num_entries):
        entry_y = ttk.Entry(data_input_tab_frame2, width=5)
        entry_y.grid(row=i + 1, column=0, padx=5, pady=5)
        entries_y.append(entry_y)

        entry_x = ttk.Entry(data_input_tab_frame2, width=5)
        entry_x.grid(row=i + 1, column=1, padx=5, pady=5)
        entries_x.append(entry_x)


# Расчёт и вывод графика
def calculate_and_draw():
    # Очищаем поле с текстом об ошибке
    label_error.config(text="")

    if data_source.get() == "default":
        # Используем предопределённые данные
        x = default_x
        y = default_y
    else:
        # Используем введённые данные.
        # Получаем данные из Entry и закидываем их в массивы numpy
        try:
            x_values = [float(entry.get()) for entry in entries_x]
            y_values = [float(entry.get()) for entry in entries_y]
        except ValueError:
            label_error.config(text="Введите коректные числа")
            return
        except NameError:
            label_error.config(
                text="Сначала создайте поля для ввода данных"
            )
            return

        x = np.array(x_values)
        y = np.array(y_values)

    # Добавление столбца единиц к x для расчета β0
    X = np.vstack([np.ones(len(x)), x]).T

    # Оценка коэффициентов β
    X_T = X.T  # Получаем транспонированную матрицу X
    X_T_X = X_T @ X  # Умножаем транспонированную матрицу X на начальную матрицу X
    X_T_X_inv = np.linalg.inv(
        X_T_X
    )  # Делаем обратной матрицу, которая является предыдущим произведением

    coefs = X_T_X_inv @ X_T @ y  # Произведение обратной матрицы(X'*X) * X' * вектор y
    beta_0, beta_1 = coefs  # Получаем коэффициенты

    # Функция построения линейной регрессии
    def linear_regression(x):
        return beta_0 + beta_1 * x

    # Отображение промежуточных вычислений.
    # Очищаем вкладку с вычислениями от старых элементов
    for widget in calculations_tab.winfo_children():
        widget.destroy()

    # Вывод промежуточных вычислений
    display_matrix(X, "Матрица X")
    display_matrix(X_T, "Транспонированная X'")
    display_matrix(X_T_X, "Произведение (X' * X)")
    display_matrix(X_T_X_inv, "Обратная (X' * X)")

    # Вывод коэффициентов
    info_beta_0 = f"Коэффициент β0: {beta_0:.3f}"
    display_label(info_beta_0, calculations_tab)

    info_beta_1 = f"Коэффициент β1: {beta_1:.3f}"
    display_label(info_beta_1, calculations_tab)

    # Построение графика
    ax.clear()  # Очистка предыдущего графика
    ax.set_xlabel("Ось X")  # Название для оси X
    ax.set_ylabel("Ось Y")  # Название для оси Y
    ax.set_title("График регрессии")  # Название графика

    # Построение точек на графике
    ax.scatter(x, y, color="orange", label="Точки", zorder=4)

    # Построение линии
    ax.plot(
        x,
        linear_regression(x),
        color="black",
        label=f"y = {beta_0:.3f} + {beta_1:.3f}x",
        linewidth=1,
        zorder=2,
    )

    # Добавление панели с дополнительной информацией
    ax.legend(
        loc="lower right",
        fontsize="large",
        facecolor="black",
        edgecolor="black",
        framealpha=0.09
    )

    ax.grid(zorder=0)  # Добавление сетки

    # Обновление Canvas
    canvas.draw()

    # Отображение авторов работы
    # ВСТАВИТЬ СВОЕ
    author_1_info = ("==================================================\n\nФ.И.О, группа, за что был в первую очередь "
                     "ответственен.")
    display_label(author_1_info, author_tab_first)

    author_2_info = ("Ф.И.О, группа, за что был в первую очередь "
                     "ответственен.\n\n==================================================")
    display_label(author_2_info, author_tab_first)


# Отображение дополнительной информации
def display_label(info, tab):
    calculations_tab_label_frame = tk.Frame(master=tab)
    calculations_tab_label_frame.pack(fill=tk.X)

    ttk.Label(calculations_tab_label_frame, text=info, font=("Arial", 20, "bold")).grid(
        row=0, column=0, columnspan=2, pady=(5, 0)
    )


root = tk.Tk()
root.title("Проект по математике и Python")

# СОЗДАНИЕ ВКЛАДОК
tabs = ttk.Notebook(root)
data_input_tab = ttk.Frame(tabs)
calculations_tab = ttk.Frame(tabs)
result_tab = ttk.Frame(tabs)
author_tab = ttk.Frame(tabs)

tabs.add(data_input_tab, text="Данные для решения")
tabs.add(calculations_tab, text="Математические вычисления")
tabs.add(result_tab, text="График регрессии")
tabs.add(author_tab, text="Авторы")

tabs.pack()

# Фрейм 1: Для ввода размера матрицы вводных данных
data_input_tab_frame1 = tk.Frame(master=data_input_tab)
data_input_tab_frame1.pack()

ttk.Label(data_input_tab_frame1, text="Введите колличество переменных по X и Y: ").grid(
    column=0, row=0, padx=5, pady=5
)

entry_data_input_tab = ttk.Entry(data_input_tab_frame1, width=4)
entry_data_input_tab.grid(column=0, row=1, padx=5)

btn_count = ttk.Button(
    data_input_tab_frame1, text="Создать поля для ввода значений", command=create_data_entries
)
btn_count.grid(column=0, row=2, padx=5, pady=5)

# Фрейм 2: Для полей ввода данных (Entries)
# Элементы этого фрейма добавляются автоматически, если нажата кнопка
data_input_tab_frame2 = tk.Frame(master=data_input_tab)
data_input_tab_frame2.pack()

# Фрейм 3: Для варианта получения данных (из полей ввода или из предопределенных значений)
data_input_tab_frame3 = tk.Frame(master=data_input_tab)
data_input_tab_frame3.pack()

# Переключатель для выбора источника входных данных(предопределённые/введённые)
data_source = tk.StringVar(value="default")

default_radiobutton = tk.Radiobutton(
    data_input_tab_frame3,
    text="Использовать данные из задачи",
    variable=data_source,
    value="default",
)
default_radiobutton.pack(anchor=tk.CENTER)

custom_radiobutton = tk.Radiobutton(
    data_input_tab_frame3,
    text="Использовать введённые данные",
    variable=data_source,
    value="custom",
)
custom_radiobutton.pack(anchor=tk.CENTER)

btn_result = ttk.Button(
    data_input_tab_frame3, text="Вычислить и построить график", command=calculate_and_draw
)
btn_result.pack(pady=10)

# Вывод ошибки
label_error = ttk.Label(root, text="", foreground="red", font=("Arial", 10, "bold"))
label_error.pack(side="bottom")

# ВКЛАДКА 2: ВЫЧИСЛЕНИЯ ---
# Появляется автоматически при вызове функции calculate_and_draw()

# ВКЛАДКА 3: РЕЗУЛЬТАТ
# Создание графика matplotlib
fig, ax = plt.subplots()

# Размещение графика matplotlib внутри фрейма Tkinter
canvas = FigureCanvasTkAgg(fig, master=result_tab)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# ВКЛАДКА 4: АВТОРЫ
author_tab_first = tk.Frame(master=author_tab)
author_tab_first.pack()

author_tab_second = tk.Frame(master=author_tab)
author_tab_second.pack()

root.mainloop()
