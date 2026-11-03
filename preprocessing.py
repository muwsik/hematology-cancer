import pandas as pd
import numpy as np


def read_data(file_path):
    df = pd.read_excel(file_path, sheet_name=0, header=3)

    # Проверка уникальности номеров заявок
    numbers = df["№ заявки"].dropna()
    duplicates = numbers[numbers.duplicated(keep=False)].unique()

    if len(duplicates):
        raise ValueError(
            f"Обнаружены повторяющиеся номера заявок: "
            f"{', '.join(duplicates)}"
        )   

    # Номер заявки распространяется на все результаты одной записи
    df["№ заявки"] = df["№ заявки"].ffill()

    # Признаки, общие для всей заявки
    common_features = [
        "Дата выполнения",
        "Пол пациента",
        "Возраст",
        "Диагноз"
    ]

    # Получаем информацию об объектах
    objects = (
        df.groupby("№ заявки", sort = False)[common_features]
        .first()
    )

    # Преобразуем результаты анализов в отдельные столбцы
    tests = df.pivot_table(
        index = "№ заявки",
        columns = "Название теста",
        values = "Результат",
        aggfunc = "first",
        sort = False
    )

    # Объединяем общие признаки и результаты анализов
    result = objects.join(tests)

    result.columns.name = None
    result.index.name = "№ заявки"

    return result.reset_index()


def prepare_data(raw_data, drop_columns = None):
    if drop_columns is None:
        drop_columns = []

    # Формируем вектор классов
    y = np.where(raw_data["Диагноз"] == "C91.1", 1, -1)

    # Исключаем ненужные колонки
    columns = ["№ заявки", "Диагноз"] + drop_columns

    missing = set(columns) - set(raw_data.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")

    features = raw_data.drop(columns = columns)

    # Кодируем пол
    if "Пол пациента" in features.columns:
        features["Пол пациента"] = features["Пол пациента"].map({
            "М": 0,
            "Ж": 1
        })

    # Преобразуем остальные признаки в числа
    for column in features.columns:
        features[column] = pd.to_numeric(
            features[column].astype("string").str.replace(",", ".", regex = False),
            errors = "raise"
        )

    X = features.to_numpy(dtype = np.float64, na_value = np.nan)
    feature_names = features.columns.to_numpy()

    return X, y, feature_names


def load_dataset(drop_nan_objects = True, drop_nan_features = False):
    drop_columns = [
        "Дата выполнения",
        "Средний объем эритроцита (MCV)",
        "Среднее содержание гемоглобина в эритроците (MCH)",
        "Средняя концентрация гемоглобина в эритроците (MCHC)",
        "Тромбокрит (PCT)",
        "Гранулоциты незрелые (IG%)",
        "Нормобласты (NRBC%)",
        "Нейтрофилы (NEU%)",
        "Лимфоциты (LYM%)",
        "Моноциты (MON%)",
        "Эозинофилы (EO%)",
        "Базофилы (BA%)"
    ]

    X1, y1, features1 = prepare_data(read_data(r"data\C91.1.xlsx"), drop_columns)

    X2, y2, features2 = prepare_data(read_data(r"data\Z01.7.xlsx"), drop_columns)

    features = features1
    if not np.array_equal(features1, features2):
        raise ValueError("Наборы или порядок признаков не совпадают")

    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    # Удаление признаков с пропусками
    if drop_nan_features:
        valid = np.isfinite(X).all(axis = 0)

        X = X[:, valid]
        features = features[valid]

    # Удаление объектов с пропусками
    if drop_nan_objects:
        valid = np.isfinite(X).all(axis = 1)

        X = X[valid]
        y = y[valid]

    return X, y, features


if __name__ == "__main__":
    # Загрузка без фильтрации
    X, y, features = load_dataset(
        drop_nan_objects = False,
        drop_nan_features = False
    )

    print(f"Исходная выборка: {X.shape}")
    print(f"Класс -1: {np.sum(y == -1)}")
    print(f"Класс  1: {np.sum(y == 1)}")

    # Проверка пропусков
    nan_mask = ~np.isfinite(X)

    print(f"Объектов с пропусками: {nan_mask.any(axis = 1).sum()}")
    print(f"Признаков с пропусками: {nan_mask.any(axis = 0).sum()}")
    print(f"Всего пропусков: {nan_mask.sum()}")

    # Проверка фильтрации объектов
    X1, y1, features1 = load_dataset(
        drop_nan_objects = True,
        drop_nan_features = False
    )

    print(f"\nПосле фильтрации объектов: {X1.shape}")
    print(f"Удалено объектов: {len(y) - len(y1)}")    
    print(f"Класс -1: {np.sum(y1 == -1)}")
    print(f"Класс  1: {np.sum(y1 == 1)}")

    # Проверка фильтрации признаков
    X2, y2, features2 = load_dataset(
        drop_nan_objects = False,
        drop_nan_features = True
    )

    print(f"\nПосле фильтрации признаков: {X2.shape}")
    print(f"Удалено признаков: {len(features) - len(features2)}")
    print("Удалённые признаки:", np.setdiff1d(features, features2))    
    print(f"Класс -1: {np.sum(y2 == -1)}")
    print(f"Класс  1: {np.sum(y2 == 1)}")