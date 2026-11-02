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

    X = features.to_numpy()
    feature_names = features.columns.to_numpy()

    return X, y, feature_names



if __name__ == "__main__":
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

    print(f"""C91.1
        Objects:{X1.shape}
        Lables: {y1.shape}
    """)

    X2, y2, features2 = prepare_data(read_data(r"data\Z01.7.xlsx"), drop_columns)
    
    print(f"""Z01.7
        Objects:{X2.shape}
        Lables: {y2.shape}
    """)

    # Проверяем совпадение признаков и их порядка
    assert np.array_equal(features1, features2), \
        "Наборы или порядок признаков не совпадают"
    
    X = np.concatenate((X1, X2), axis = 0)
    y = np.concatenate((y1, y2), axis = 0)    
    print(f"""All
        Objects:{X.shape}\n{X}\n
        Lables: {y.shape}\n{y}\n
    """)