import os
import numpy as np


class MatrixLab:
    def __init__(self, seed=42):
        # Матрицы хранятся в полях объекта (self.*)
        self.seed = seed
        self.my_array = None
        self.A = None
        self.B = None
        self.dir = os.path.dirname(os.path.abspath(__file__))

    # Вспомогательные методы вывода
    def _print_step(self, num, title):
        print("\n" + "=" * 60)
        print(f"Шаг {num}. {title}")
        print("=" * 60)

    def _shape_str(self, M):
        # Возвращает размер матрицы в виде "строки×столбцы"
        return f"{M.shape[0]}×{M.shape[1]}"

    def _format_number(self, x):
        # Форматируем число компактно, без длинного хвоста
        return f"{x:.6g}"

    def _format_vector(self, v):
        # np.ravel превращает любой массив в одномерный
        # чтобы можно было напечатать его в одну строку
        return " ".join(self._format_number(x) for x in np.ravel(v))

    def _format_matrix(self, M):
        # Печатаем матрицу строками, без квадратных скобок
        return "\n".join(
            "  ".join(self._format_number(x) for x in row)
            for row in M
        )

    def _print_matrix(self, name, M):
        print(f"{name}:")
        print(self._format_matrix(M))

    def _print_big_matrix(self, name, M):
        # Если матрица большая, печатаем только её угол
        print(f"{name} (размер {self._shape_str(M)}):")
        if M.size <= 25:
            print(self._format_matrix(M))
        else:
            print("Первые 3×3 элемента:")
            print(self._format_matrix(M[:3, :3]))
            print("...")

    # Шаги 1–12
    def step1_create_array(self):
        self._print_step(1, "Создание массива")
        # np.arange(начало, конец, шаг) создаёт последовательность чисел:
        # от 10 включительно до 70 не включая с шагом 2
        self.my_array = np.arange(10, 70, 2)
        print("Массив my_array:")
        print(self._format_vector(self.my_array))

    def step2_form_matrix_A(self):
        self._print_step(2, "Формирование матрицы A")
        # np.reshape(6, 5) раскладывает 30 чисел в 6 строк и 5 столбцов
        matrix_6x5 = self.my_array.reshape(6, 5)
        # .T — транспонирование: строки становятся столбцами, столбцы — строками
        self.A = matrix_6x5.T
        print(f"Размер матрицы A: {self._shape_str(self.A)}")
        self._print_matrix("Матрица A", self.A)

    def step3_transform_A(self):
        self._print_step(3, "Преобразование элементов A")
        # Умножаем каждый элемент на 2.5 и вычитаем 5 одной строкой
        self.A = self.A * 2.5 - 5
        self._print_matrix("Матрица A после преобразования", self.A)
        # .min() ищет минимальное значение среди всех элементов матрицы
        print(f"Минимальный элемент A: {self._format_number(self.A.min())}")

    def step4_create_B(self):
        self._print_step(4, "Создание матрицы B")
        # np.random.seed фиксирует генератор случайных чисел
        # чтобы при каждом запуске получались одинаковые числа
        np.random.seed(self.seed)
        # np.random.uniform(0, 10, size=(6, 3)) генерирует случайные числа
        # из диапазона [0, 10) и складывает их в матрицу 6×3
        self.B = np.random.uniform(0, 10, size=(6, 3))
        self._print_matrix("Матрица B", self.B)
        print(f"Размерность B: {self._shape_str(self.B)}")

    def step5_sum_vectors(self):
        self._print_step(5, "Векторы сумм")
        # .sum(axis=1) складывает элементы каждой строки матрицы A
        a = self.A.sum(axis=1)
        # .sum(axis=0) складывает элементы каждого столбца матрицы B
        b = self.B.sum(axis=0)
        print(f"Вектор a (суммы по строкам A), размер {a.shape[0]}:")
        print(self._format_vector(a))
        print(f"Вектор b (суммы по столбцам B), размер {b.shape[0]}:")
        print(self._format_vector(b))

    def step6_multiply(self):
        self._print_step(6, "Умножение матриц A · B")
        # Оператор @ выполняет матричное умножение
        # (5, 6) @ (6, 3) -> (5, 3)
        product = self.A @ self.B
        self._print_matrix("Результат A · B", product)
        print(f"Размерность результата: {self._shape_str(product)}")

    def step7_make_square(self):
        self._print_step(7, "Приведение к квадратному виду")
        # np.delete удаляет третий столбец (индекс 2) из матрицы A -> 5×5
        self.A = np.delete(self.A, 2, axis=1)
        # Снова генерируем случайные числа, теперь из [10, 20)
        extra = np.random.uniform(10, 20, size=(self.B.shape[0], 3))
        # np.hstack приклеивает новые столбцы справа к матрице B -> 6×6
        self.B = np.hstack((self.B, extra))
        print(f"Размер A после удаления столбца: {self._shape_str(self.A)}")
        print(f"Размер B после добавления столбцов: {self._shape_str(self.B)}")

    def _safe_inv(self, M, name):
        # Обратная матрица с обработкой случая вырожденной матрицы
        try:
            # np.linalg.inv вычисляет обратную матрицу
            inv = np.linalg.inv(M)
            self._print_matrix(f"Обратная матрица {name}", inv)
        except np.linalg.LinAlgError:
            print(f"Обратная матрица {name} не существует (матрица вырождена).")

    def step8_det_inv(self):
        self._print_step(8, "Определители и обратные матрицы")
        # np.linalg.det вычисляет определитель матрицы
        det_A = np.linalg.det(self.A)
        det_B = np.linalg.det(self.B)
        print(f"det(A) = {self._format_number(det_A)}")
        print(f"det(B) = {self._format_number(det_B)}")
        self._safe_inv(self.A, "A")
        self._safe_inv(self.B, "B")

    def step9_power(self):
        self._print_step(9, "Возведение в степень")
        # np.linalg.matrix_power возводит квадратную матрицу в целую степень
        # именно матричным умножением, а не поэлементно
        self.A = np.linalg.matrix_power(self.A, 6)
        self.B = np.linalg.matrix_power(self.B, 14)
        self._print_big_matrix("A^6", self.A)
        self._print_big_matrix("B^14", self.B)

    def step10_solve_system(self):
        self._print_step(10, "Решение системы уравнений (вариант 3)")
        # номер_в_группе = 18 -> (18 % 8) + 1 = 3
        # Пропущенные переменные в уравнениях дают коэффициент 0
        # np.array создаёт матрицу из списка списков
        coeff = np.array([
            [3.0, -1.2, -8.0, 8.0],
            [21.0, -19.0, 0.5, 0.0],
            [7.0, 0.0, -4.9, -2.0],
            [1.0, -2.0, 13.0, 9.0],
        ])
        rhs = np.array([20.0, -8.0, 11.0, 3.0])

        self._print_matrix("Матрица коэффициентов", coeff)
        print("Вектор правых частей:")
        print(self._format_vector(rhs))

        # np.linalg.solve решает систему линейных уравнений A x = b
        x = np.linalg.solve(coeff, rhs)
        print("Решение x:")
        print(self._format_vector(x))

        # Оператор @ вычисляет A x, затем вычитаем b — получаем невязку
        residual = coeff @ x - rhs
        # np.linalg.norm вычисляет длину (норму) вектора невязки
        norm = np.linalg.norm(residual)
        # np.allclose проверяет, что все элементы близки к нулю
        ok = np.allclose(residual, 0)
        print("Невязка:")
        print(self._format_vector(residual))
        print(f"Норма невязки: {self._format_number(norm)}")
        print(f"Невязка близка к нулю: {ok}")

    def step11_analysis(self):
        self._print_step(11, "Дополнительный анализ матриц (после п. 9)")
        # np.linalg.matrix_rank вычисляет ранг матрицы
        print(f"Ранг A: {np.linalg.matrix_rank(self.A)}")
        print(f"Ранг B: {np.linalg.matrix_rank(self.B)}")

        # .mean() вычисляет среднее арифметическое всех элементов
        print(f"Среднее A: {self._format_number(self.A.mean())}")
        print(f"Среднее B: {self._format_number(self.B.mean())}")

        # np.cov вычисляет ковариационную матрицу
        # По умолчанию каждая строка — отдельная переменная
        # поэтому для матриц A и B это именно то, что нужно
        cov_A = np.cov(self.A)
        cov_B = np.cov(self.B)
        self._print_matrix("Ковариационная матрица A", cov_A)
        self._print_matrix("Ковариационная матрица B", cov_B)

        # np.argmax возвращает индекс максимального элемента
        # в «плоском» (одномерном) представлении массива
        flat_index = np.argmax(self.A)
        # np.unravel_index превращает плоский индекс
        # в пару (строка, столбец) для формы матрицы A
        row, col = np.unravel_index(flat_index, self.A.shape)
        print(f"Индексы максимального элемента A: строка {row}, столбец {col}")

        # .ravel() превращает матрицу B в одномерный вектор
        b_vec = self.B.ravel()
        print(f"Размер одномерного вектора из B: {b_vec.shape[0]}")
        print("Первые 10 элементов вектора B:")
        print(self._format_vector(b_vec[:10]))

    def step12_save_load(self):
        self._print_step(12, "Сохранение и загрузка матриц")
        path_A = os.path.join(self.dir, "A.csv")
        path_B = os.path.join(self.dir, "B.csv")

        # np.savetxt сохраняет матрицу в текстовый файл
        # с разделителем-запятой
        np.savetxt(path_A, self.A, delimiter=",")
        np.savetxt(path_B, self.B, delimiter=",")
        print(f"Сохранено: {path_A}")
        print(f"Сохранено: {path_B}")

        # np.loadtxt загружает матрицу обратно из текстового файла
        A_loaded = np.loadtxt(path_A, delimiter=",")
        B_loaded = np.loadtxt(path_B, delimiter=",")

        # np.abs берёт модуль числа, np.max ищет максимум
        diff_A = np.max(np.abs(self.A - A_loaded))
        diff_B = np.max(np.abs(self.B - B_loaded))
        print(f"Максимальная разница для A: {self._format_number(diff_A)}")
        print(f"Максимальная разница для B: {self._format_number(diff_B)}")

    def run(self):
        # Инкапсуляция: снаружи вызывается только run()
        # все шаги идут по порядку
        self.step1_create_array()
        self.step2_form_matrix_A()
        self.step3_transform_A()
        self.step4_create_B()
        self.step5_sum_vectors()
        self.step6_multiply()
        self.step7_make_square()
        self.step8_det_inv()
        self.step9_power()
        self.step10_solve_system()
        self.step11_analysis()
        self.step12_save_load()


if __name__ == "__main__":
    lab = MatrixLab()
    lab.run()