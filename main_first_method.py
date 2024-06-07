# pip install numpy matplotlib

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Данные из задачи (вставить свои!)
default_y = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]
default_x = [10, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]


# Создание полей ввода данных
def create_data_entries():
    try:
        num_entries = int(entry_tab1_input.get())
    except ValueError:
        label_error.config(text="Пожалуйста, введите число")
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
            label_error.config(text="Пожалуйста, введите корректные числа")
            return
        except NameError:
            label_error.config(
                text="Пожалуйста, сначала создайте поля для ввода данных"
            )
            return

        x = x_values
        y = y_values

    # Вычисление средних значений
    _x = np.mean(x)
    _y = np.mean(y)

    # Вычисление средних квадратов
    _x2 = np.mean([num**2 for num in x])
    _y2 = np.mean([num**2 for num in y])

    # Вычисление среднего произведения x и y
    xy = np.mean([num_x * num_y for num_x, num_y in zip(x, y)])

    kxy = round(xy - _x * _y, 3)
    qx = round(np.sqrt(_x2 - _x**2), 3)
    qy = round(np.sqrt(_y2 - _y**2), 3)

    r_xy = round(kxy / (qx * qy), 3)

    # Создание левой и правой стороны уравнения для нахождения коэффициентов a и b
    left_side = np.array(
        [[np.sum(x), len(x)], [np.sum([num**2 for num in x]), np.sum(x)]]
    )
    right_side = np.array(
        [[np.sum(y)], [np.sum([num_x * num_y for num_x, num_y in zip(x, y)])]]
    )

    # Нахождение коэффициентов a и b
    result = np.linalg.inv(left_side).dot(right_side)
    a = round(result[0][0], 3)
    b = round(result[1][0], 3)

    # Функция построения линейной регрессии
    def linear_regression(x):
        return b + a * x

    # Отображение промежуточных вычислений.
    # Очищаем вкладку с вычислениями от старых элементов
    for widget in calculations_tab.winfo_children():
        widget.destroy()

    # Вывод промежуточных вычислений
    coefs_info = f"Коэффициенты: \na = {a}\nb = {b}"
    display_label(coefs_info)

    correlation_info = f"kxy = {kxy}\nqx = {qx}\nqy = {qy}"
    display_label(correlation_info)

    coef_of_correlation = f"Коэффициент корреляции = {r_xy}"
    display_label(coef_of_correlation)

    # Построение графика
    ax.clear()  # Очистка предыдущего графика
    ax.set_xlabel("ОСЬ X")  # Название для оси X
    ax.set_ylabel("ОСЬ Y")  # Название для оси Y
    ax.set_title("График регрессии")  # Название графика

    ax.scatter(
        x, y, color="red", label="Данные", zorder=3, s=10
    )  # Построение точек на графике
    ax.plot(
        x,
        linear_regression(np.array(x)),
        color="black",
        label=f"y = {b} + {a}x",
        zorder=2,
        linewidth=1,
    )  # Построение линии
    ax.legend(
        loc="lower right",
        fontsize="large",
        title_fontsize="medium",
        facecolor="black",
        edgecolor="black",
        framealpha=0.07,
    )  # Добавление панели с дополнительной информацией
    ax.grid(zorder=0)  # Добавление сетки

    # Обновление Canvas
    canvas.draw()


# Отображение дополнительной информации
def display_label(info):
    calculations_tab_label_frame = tk.Frame(master=calculations_tab)
    calculations_tab_label_frame.pack(side=tk.TOP, fill=tk.Y)

    ttk.Label(
        calculations_tab_label_frame, text=info, font=("Times New Roman", 20)
    ).grid(row=0, column=0, columnspan=2, pady=(30, 0))


root = tk.Tk()

# Задаем свое правило (функцию on_closing), которое определяет действия при закрытии программы
# Необходимо для корректного порядка закрытия сначала графика matplotlib, а потом уже только окна Tkinter
root.title("ПРОЕКТ")

# ОПИСАНИЕ СТИЛЕЙ
arial_12_style = ttk.Style()
arial_12_style.configure("Arial_9.TLabel", font=("Arial", 9))

# СОЗДАНИЕ ВКЛАДОК
tabs = ttk.Notebook(root)
data_input_tab = ttk.Frame(tabs, borderwidth=20, relief=tk.RIDGE)
calculations_tab = ttk.Frame(tabs, borderwidth=20, relief=tk.RIDGE)
result_tab = ttk.Frame(tabs, borderwidth=20, relief=tk.RIDGE)

tabs.add(data_input_tab, text="Данные")
tabs.add(calculations_tab, text="Вычисления")
tabs.add(result_tab, text="График")

tabs.pack()

# Фрейм 1: Для ввода размера матрицы вводных данных
data_input_tab_frame1 = tk.Frame(master=data_input_tab)
data_input_tab_frame1.pack()

ttk.Label(data_input_tab_frame1, text="Кол-во значений: ").grid(
    column=0, row=0, padx=5, pady=5
)

entry_tab1_input = ttk.Entry(data_input_tab_frame1, width=4)
entry_tab1_input.grid(column=0, row=2, padx=5, pady=5)

btn_count = ttk.Button(
    data_input_tab_frame1, text="Создать поля для ввода", command=create_data_entries
)
btn_count.grid(column=0, row=3, padx=5, pady=5)

# Фрейм 2: Для полей ввода данных (Entries)
# Элементы этого фрейма добавляются автоматически, если нажата кнопка
data_input_tab_frame2 = tk.Frame(master=data_input_tab)
data_input_tab_frame2.pack()

# Фрейм 3: Для варианта получения данных (из полей ввода или из предопределенных значений)
tab1_frame3 = tk.Frame(master=data_input_tab)
tab1_frame3.pack()

# Переключатель для выбора источника входных данных(предопределённые/введённые)
data_source = tk.StringVar(value="default")

default_radiobutton = tk.Radiobutton(
    tab1_frame3,
    text="Использовать данные из задачи",
    variable=data_source,
    value="default",
)
default_radiobutton.pack(anchor=tk.CENTER)

custom_radiobutton = tk.Radiobutton(
    tab1_frame3,
    text="Использовать введённые данные",
    variable=data_source,
    value="custom",
)
custom_radiobutton.pack(anchor=tk.CENTER)

btn_result = ttk.Button(
    tab1_frame3, text="Вычислить и построить график", command=calculate_and_draw
)
btn_result.pack(pady=10)

# Вывод ошибки
label_error = ttk.Label(root, text="", foreground="red", style="Arial_9.TLabel")
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

root.mainloop()
