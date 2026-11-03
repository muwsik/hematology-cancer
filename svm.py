import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from preprocessing import load_dataset


# Параметры эксперимента
C = 1.0
n_splits = 5
random_state = np.random.default_rng().integers(0, 2**32).item()
drop_nan_objects = False
drop_nan_features = True

# Загрузка данных
X, y, features = load_dataset(
    drop_nan_objects = drop_nan_objects,
    drop_nan_features = drop_nan_features
)


if __name__ == "1__main__":
    # Кросс-валидация
    cv = StratifiedKFold(
        n_splits = n_splits,
        shuffle = True,
        random_state = random_state
    )

    scoring = {
        "accuracy": [],
        "precision_1": [],
        "recall_1": [],
        "precision_-1": [],
        "recall_-1": [],
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
        scoring["precision_1"].append(precision_score(y_test, y_pred, pos_label = 1))
        scoring["recall_1"].append(recall_score(y_test, y_pred, pos_label = 1))
        scoring["precision_-1"].append(precision_score(y_test, y_pred, pos_label = -1))
        scoring["recall_-1"].append(recall_score(y_test, y_pred, pos_label = -1))
        scoring["roc_auc"].append(roc_auc_score(y_test, y_score))

    # Формирование результатов
    header = [
        "method", "kernel", "C",
        "n_splits", "random_state",
        "drop_nan_objects", "drop_nan_features",
        "n_objects", "n_features",
        "class_-1", "class_1",
        "accuracy_mean", "accuracy_std",
        "precision_1_mean", "precision_1_std",
        "recall_1_mean", "recall_1_std",
        "precision_-1_mean", "precision_-1_std",
        "recall_-1_mean", "recall_-1_std",
        "roc_auc_mean", "roc_auc_std"
    ]

    values = [
        "SVM", "linear", C,
        n_splits, random_state,
        drop_nan_objects, drop_nan_features,
        len(y), X.shape[1],
        np.sum(y == -1), np.sum(y == 1),
        np.mean(scoring["accuracy"]), np.std(scoring["accuracy"]),
        np.mean(scoring["precision_1"]), np.std(scoring["precision_1"]),
        np.mean(scoring["recall_1"]), np.std(scoring["recall_1"]),
        np.mean(scoring["precision_-1"]), np.std(scoring["precision_-1"]),
        np.mean(scoring["recall_-1"]), np.std(scoring["recall_-1"]),
        np.mean(scoring["roc_auc"]), np.std(scoring["roc_auc"])
    ]

    print("\t".join(map(str, header)))
    print("\t".join(
        f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
        for value in values
    ))
    print(features)


import plotly.graph_objects as go
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

    # Визуализация
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x = bin_centers,
        y = hist_neg,
        name = "Class Z01.7",
        opacity = 0.6
    ))

    fig.add_trace(go.Bar(
        x = bin_centers,
        y = hist_pos,
        name = "Class C91.1",
        opacity = 0.6
    ))

    fig.update_layout(
        title = (
            "SVM: decision function distributions<br>"
            f"Confusion Matrix: TN = {cm[0, 0]}, FP = {cm[0, 1]}, "
            f"FN = {cm[1, 0]}, TP = {cm[1, 1]}<br>"
            f"{drop_nan_objects}, {drop_nan_features}"
        ),
        barmode = "overlay",
        bargap = 0,
        xaxis_title = "Decision function",
        yaxis_title = "Density",
        legend_title = "Class"
    )

    fig.add_vline(
        x = 0,
        line_dash = "dash",
        line_color = "black",
        annotation_text = "Decision boundary"
    )

    fig.show()

    # Коэффициенты линейного SVM
    weights = model.coef_.ravel()

    # Сортировка по абсолютному значению
    indices = np.argsort(np.abs(weights))

    fig = go.Figure(go.Bar(
        x = weights[indices],
        y = np.array(features)[indices],
        orientation = "h"
    ))

    fig.update_layout(
        title = f"SVM: feature contributions ({drop_nan_objects}, {drop_nan_features})",
        xaxis_title = "Coefficient",
        yaxis_title = "Feature",
        height = max(600, len(features) * 25)
    )

    fig.add_vline(x = 0, line_dash = "dash", line_color = "black")

    fig.show()