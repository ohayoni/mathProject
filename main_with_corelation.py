# pip install numpy matplotlib

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Данные из задачи (вставить свои!)
default_y = np.array(
    [0.27, 0.40, 0.36, 0.42, 0.45, 0.51, 0.55, 0.58, 0.61, 0.64, 0.68, 0.72, 0.76, 0.78, 0.82, 0.88, 0.95, 1.20]
)
default_x = np.array(
    [1330, 1340, 1350, 1360, 1370, 1380, 1390, 1400, 1410, 1420, 1430, 1440, 1450, 1460, 1470, 1480, 1490, 1500]
)


# --- ФУНКЦИИ ---
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
    label_title = ttk.Label(calculations_tab_frame, text=title, font=("Arial", 12, "bold"))
    label_title.grid(row=0, column=0, sticky="w", pady=(10, 2))

    # Дополнительный фрейм для удобства вывода матрицы
    calculations_tab_matrix_frame = tk.Frame(master=calculations_tab_frame)
    calculations_tab_matrix_frame.grid(row=1, column=0, sticky="w")

    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            # Пояснение - :.3f позволяет выводит число только с тремя знаками после запятой
            label = tk.Label(
                calculations_tab_matrix_frame,
                text=f"{val:.3f}",
                borderwidth=1,
                relief="solid",
                width=10,
            )
            label.grid(row=i, column=j, padx=1, pady=1)


# Создание полей ввода данных
def create_data_entries():
    try:
        num_entries = int(n_entry_data_input_tab.get())
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
        # Используем введённые данные
        # Получаем данные из Entry и закидываем их в массивы numpy
        try:
            x_values = [float(entry.get()) for entry in entries_x]
            y_values = [float(entry.get()) for entry in entries_y]
        except ValueError:
            label_error.config(text="Пожалуйста, введите коректные числа")
            return
        except NameError:
            label_error.config(
                text="Пожалуйста, сначала создайте поля для ввода данных"
            )
            return

        x = np.array(x_values)
        y = np.array(y_values)

    # Расчет коэффициента корелляции
    _x = np.mean(x)
    _y = np.mean(y)

    k_x_y = np.mean(x * y) - _x * _y
    sigma_x = np.std(x, ddof=0)
    sigma_y = np.std(y, ddof=0)
    corelation_coef = round(abs(k_x_y / (sigma_x * sigma_y)), 3)

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

    # Отображение промежуточных вычислений
    # Очищаем вкладку с вычислениями от старых элементов
    for widget in calculations_tab.winfo_children():
        widget.destroy()

    # Вывод промежуточных вычислений
    display_matrix(X, "Исходная матрица X")
    display_matrix(X_T, "Транспонированная матрица X'")
    display_matrix(X_T_X, "Произведение матриц (X' * X)")
    display_matrix(X_T_X_inv, "Обратная матрица (X' * X)")

    # Вывод коэффициентов
    info_beta_0 = f"Коэффициент β0: {beta_0:.3f}"
    display_label(info_beta_0)

    info_beta_1 = f"Коэффициент β1: {beta_1:.3f}"
    display_label(info_beta_1)

    corelation_coef_info = f"Коэффициент кореляции = {corelation_coef}"
    display_label(corelation_coef_info)

    # Построение графика
    ax.clear()  # Очистка предыдущего графика
    ax.set_xlabel("X")  # Название для оси X
    ax.set_ylabel("Y")  # Название для оси Y
    ax.set_title("Линейная регрессия")  # Название графика

    ax.scatter(
        x, y, color="green", label="Данные", zorder=5
    )  # Построение точек на графике
    ax.plot(
        x,
        linear_regression(x),
        color="black",
        label=f"y = {beta_0:.3f} + {beta_1:.3f}x",
    )  # Построение линии
    ax.legend()  # Добавление панели с дополнительной информацией
    ax.grid()  # Добавление сетки

    # Обновление Canvas
    canvas.draw()


# Отображение дополнительной информации (разных коэффициентов)
def display_label(info):
    calculations_tab_label_frame = tk.Frame(master=calculations_tab)
    calculations_tab_label_frame.pack(fill=tk.X)

    ttk.Label(
        calculations_tab_label_frame, text=info, font=("Arial", 12, "bold")
    ).grid(row=0, column=0, columnspan=2, pady=(5, 0))


# Завершение цикла matplotlib
# Выполняется при закрытии программы
def on_closing():
    plt.close()
    root.destroy()


root = tk.Tk()

# Задаем свое правило (функцию on_closing), которое определяет действия при закрытии программы
# Необходимо для корректного порядка закрытия сначала графика matplotlib, а потом уже только окна Tkinter
root.protocol("WM_DELETE_WINDOW", on_closing)
root.title("Проект по математике")

# --- ОПИСАНИЕ СТИЛЕЙ ТЕКСТА ---
italic_style = ttk.Style()
italic_style.configure("Italic.TLabel", font=("Arial", 10, "italic"))

arial_12_style = ttk.Style()
arial_12_style.configure("Arial_12.TLabel", font=("Arial", 12))

# --- СОЗДАНИЕ ВКЛАДОК ---
tabs = ttk.Notebook(root)
data_input_tab = ttk.Frame(tabs)
calculations_tab = ttk.Frame(tabs)
result_tab = ttk.Frame(tabs)

tabs.add(data_input_tab, text="Данные на вход")
tabs.add(calculations_tab, text="Вычисления")
tabs.add(result_tab, text="График")

tabs.pack()

# --- ВКЛАДКА 1: ВХОДНЫЕ ДАННЫЕ  ---
# Фрейм 1: Для ввода размера матрицы вводных данных
data_input_tab_frame1 = tk.Frame(master=data_input_tab)
data_input_tab_frame1.pack()

ttk.Label(data_input_tab_frame1, text="Введите количество значений: ").grid(
    column=0, row=0, padx=5, pady=10
)

n_entry_data_input_tab = ttk.Entry(data_input_tab_frame1)
n_entry_data_input_tab.grid(column=1, row=0, padx=5)

btn_count = ttk.Button(
    data_input_tab_frame1, text="Создать поля ввода", command=create_data_entries
)
btn_count.grid(column=2, row=0, padx=5)

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
    text="Данные из задачи",
    variable=data_source,
    value="default",
)
default_radiobutton.pack(anchor=tk.CENTER)

custom_radiobutton = tk.Radiobutton(
    tab1_frame3,
    text="Введённые данные",
    variable=data_source,
    value="custom",
)
custom_radiobutton.pack(anchor=tk.CENTER)

btn_result = ttk.Button(
    tab1_frame3, text="Вычислить и построить график", command=calculate_and_draw
)
btn_result.pack(pady=10)

# Вывод ошибки
label_error = ttk.Label(root, text="", foreground="red", style="Arial_12.TLabel")
label_error.pack(side="bottom")

# --- ВКЛАДКА 2: ВЫЧИСЛЕНИЯ ---
# Появляется автоматически при вызове функции calculate_and_draw()

# --- ВКЛАДКА 3: РЕЗУЛЬТАТ ---
# Создание графика matplotlib
fig, ax = plt.subplots()

# Размещение графика matplotlib внутри фрейма Tkinter
canvas = FigureCanvasTkAgg(fig, master=result_tab)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

root.mainloop()
