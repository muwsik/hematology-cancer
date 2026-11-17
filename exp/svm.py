import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from preprocessing import load_dataset


# Параметры эксперимента
C = 10.0
n_splits = 5
random_state = np.random.default_rng().integers(0, 2**32).item()
drop_nan_objects = False
drop_nan_features = True

# Загрузка данных
X, y, features, applications = load_dataset(
    drop_nan_objects = drop_nan_objects,
    drop_nan_features = drop_nan_features
)


if __name__ == "__main__":
    # Кросс-валидация
    cv = StratifiedKFold(
        n_splits = n_splits,
        shuffle = True,
        random_state = random_state
    )

    scoring = {
        "accuracy": [],
        "precision_+1": [],
        "recall_+1": [],
        "f1_+1": [],
        "precision_-1": [],
        "recall_-1": [],
        "f1_-1": [],
        "roc_auc": []
    }

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Масштабирование
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Обучение SVM
        model = SVC(kernel = "linear", C = C)
        model.fit(X_train, y_train)

        # Предсказания
        y_pred = model.predict(X_test)
        y_score = model.decision_function(X_test)

        # Оценка качества
        scoring["accuracy"].append(accuracy_score(y_test, y_pred))
        scoring["precision_+1"].append(precision_score(y_test, y_pred, pos_label = 1))
        scoring["recall_+1"].append(recall_score(y_test, y_pred, pos_label = 1))
        scoring["f1_+1"].append(f1_score(y_test, y_pred, pos_label = 1))
        scoring["precision_-1"].append(precision_score(y_test, y_pred, pos_label = -1))
        scoring["recall_-1"].append(recall_score(y_test, y_pred, pos_label = -1))
        scoring["f1_-1"].append(f1_score(y_test, y_pred, pos_label = -1))
        scoring["roc_auc"].append(roc_auc_score(y_test, y_score))

    # Формирование результатов
    header = [
        "method", "kernel", "C",
        "n_splits", "random_state",
        "drop_nan_objects", "drop_nan_features",
        "n_objects", "n_features",
        "class_-1", "class_1",
        "accuracy",
        "precision_1", "recall_1", "f1_1",
        "precision_-1", "recall_-1", "f1_-1",
        "roc_auc"
    ]

    values = [
        "SVM", "linear", C,
        n_splits, random_state,
        drop_nan_objects, drop_nan_features,
        len(y), X.shape[1],
        np.sum(y == -1), np.sum(y == 1)
    ]

    metrics = [
        "accuracy",
        "precision_+1", "recall_+1", "f1_+1",
        "precision_-1", "recall_-1", "f1_-1",
        "roc_auc"
    ]

    values += [
        f"{np.mean(scoring[metric]):.4f}+-{np.std(scoring[metric]):.4f}"
        for metric in metrics
    ]

    print("\t".join(header))
    print("\t".join(map(str, values)))
    print(features)




import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    # Обучение модели на всей выборке для визуализации
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel = "linear", C = C)
    model.fit(X_scaled, y)

    # Выходы классификатора
    decision_values = model.decision_function(X_scaled)
    y_pred = model.predict(X_scaled)
    cm = confusion_matrix(y, y_pred, labels = [-1, 1])

    # Заявки с ошибками
    pd.DataFrame({
        "application": applications,
        "true_class": y,
        "predicted_class": y_pred
    }).to_csv("svm_predictions.csv", index = False)


    # Параметры гистограммы
    bin_width = 0.5

    bin_min = np.floor(decision_values.min() / bin_width) * bin_width
    bin_max = np.ceil(decision_values.max() / bin_width) * bin_width

    bins = np.arange(
        bin_min,
        bin_max + bin_width,
        bin_width
    )

    # Вычисление гистограмм отдельно для каждого класса
    hist_neg, _ = np.histogram(
        decision_values[y == -1],
        bins = bins
    )

    hist_pos, _ = np.histogram(
        decision_values[y == 1],
        bins = bins
    )

    # Центры интервалов
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Коэффициенты линейного SVM
    weights = model.coef_.ravel()

    # Сортировка по абсолютному значению
    indices = np.argsort(np.abs(weights))

    # Номера заявок для каждого интервала
    apps_neg = [
        applications[
            (y == -1) &
            (decision_values >= left) &
            (decision_values < right)
        ][:10]
        for left, right in zip(bins[:-1], bins[1:])
    ]

    apps_pos = [
        applications[
            (y == 1) &
            (decision_values >= left) &
            (decision_values < right)
        ][:10]
        for left, right in zip(bins[:-1], bins[1:])
    ]

    # Создание общей фигуры
    fig = make_subplots(
        rows = 2, cols = 1,
        vertical_spacing = 0.12,
        subplot_titles = (
            "SVM: decision function distributions",
            "SVM: feature contributions"
        )
    )

    # Гистограмма распределения выходов SVM
    fig.add_trace(go.Bar(
        x = bin_centers,
        y = hist_neg,
        name = "Class Z01.7",
        marker_color = "green",
        opacity = 0.6,
        customdata = [[", ".join(map(str, apps))] for apps in apps_neg],
        hovertemplate = (
            "Decision function: %{x:.2f}<br>"
            "Objects: %{y}<br>"
            "Applications: %{customdata[0]}"
            "<extra>%{fullData.name}</extra>"
        )
    ), row = 1, col = 1)

    fig.add_trace(go.Bar(
        x = bin_centers,
        y = hist_pos,
        name = "Class C91.1",
        marker_color = "red",
        opacity = 0.75,        
        customdata = [[", ".join(map(str, apps))] for apps in apps_pos],
        hovertemplate = (
            "Decision function: %{x:.2f}<br>"
            "Objects: %{y}<br>"
            "Applications: %{customdata[0]}"
            "<extra>%{fullData.name}</extra>"
        )
    ), row = 1, col = 1)

    # Граница принятия решения
    fig.add_vline(
        x = 0,
        line_dash = "dash",
        line_color = "black",
        annotation_text = "Decision boundary",
        row = 1, col = 1
    )

    # Гистограмма коэффициентов признаков
    fig.add_trace(go.Bar(
        x = np.abs(weights[indices]),
        y = np.array(features)[indices],
        orientation = "h",
        marker_color = "steelblue",
        showlegend = False
    ), row = 2, col = 1)

    # Нулевая линия коэффициентов
    fig.add_vline(
        x = 0,
        line_dash = "dash",
        line_color = "black",
        row = 2,
        col = 1
    )

    # Общие настройки
    fig.update_layout(
        title = (
            f"SVM(C = {C}): classification results<br>"
            f"Confusion Matrix: TN = {cm[0, 0]}, FP = {cm[0, 1]}, "
            f"FN = {cm[1, 0]}, TP = {cm[1, 1]}<br>"
            f"(drop_objects: {drop_nan_objects}, drop_features: {drop_nan_features})"
        ),
        barmode = "overlay",
        bargap = 0,
        height = 1500,
        legend_title = "Class"
    )

    # Подписи осей
    fig.update_xaxes(title_text = "Decision function", row = 1, col = 1)
    fig.update_yaxes(title_text = "Objects", row = 1, col = 1)

    fig.update_xaxes(title_text = "Coefficient", row = 2, col = 1)
    fig.update_yaxes(title_text = "Feature", row = 2, col = 1)

    fig.show()