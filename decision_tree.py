import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold

from preprocessing import load_dataset


# Параметры эксперимента
criterion = "gini"
splitter = "best"
max_features = None
max_depth = None
min_samples_split = 25
min_samples_leaf = 10

n_splits = 5

drop_nan_objects = False
drop_nan_features = True

random_state = np.random.default_rng().integers(0, 2**32).item()

model_params = {
    "criterion": criterion,
    "splitter": splitter,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "random_state": random_state
}


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
    }

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Обучение дерева решений
        model = DecisionTreeClassifier(**model_params)
        model.fit(X_train, y_train)

        # Предсказания
        y_pred = model.predict(X_test)

        # Оценка качества
        scoring["accuracy"].append(
            accuracy_score(y_test, y_pred)
        )

        for label in [1, -1]:
            scoring[f"precision_{'+' if label == 1 else ''}{label}"].append(
                precision_score(
                    y_test, y_pred,
                    pos_label = label,
                    zero_division = 0
                )
            )

            scoring[f"recall_{'+' if label == 1 else ''}{label}"].append(
                recall_score(
                    y_test, y_pred,
                    pos_label = label,
                    zero_division = 0
                )
            )

            scoring[f"f1_{'+' if label == 1 else ''}{label}"].append(
                f1_score(
                    y_test, y_pred,
                    pos_label = label,
                    zero_division = 0
                )
            )

    # Формирование результатов
    header = [
        "method", "criterion", "splitter", "max_features",
        "max_depth", "min_samples_split", "min_samples_leaf",
        "n_splits", "random_state",
        "drop_nan_objects", "drop_nan_features",
        "n_objects", "n_features",
        "class_-1", "class_1",
        "accuracy",
        "precision_1", "recall_1", "f1_1",
        "precision_-1", "recall_-1", "f1_-1",
    ]

    values = [
        "DecisionTree", criterion, splitter, max_features,
        max_depth, min_samples_split, min_samples_leaf,
        n_splits, random_state,
        drop_nan_objects, drop_nan_features,
        len(y), X.shape[1],
        np.sum(y == -1), np.sum(y == 1)
    ]

    metrics = [
        "accuracy",
        "precision_+1", "recall_+1", "f1_+1",
        "precision_-1", "recall_-1", "f1_-1",
    ]

    values += [
        f"{np.mean(scoring[metric]):.4f}+-{np.std(scoring[metric]):.4f}"
        for metric in metrics
    ]

    print("\t".join(header))
    print("\t".join(map(str, values)))



def plot_decision_tree(fig, model, X, y, applications, features):
    tree = model.tree_

    y_pred = model.predict(X)
    leaves = model.apply(X)
    decision_path = model.decision_path(X)

    positions = {}
    leaf_position = [0]

    def set_positions(node, depth):
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left == right:
            x = leaf_position[0]
            leaf_position[0] += 1
        else:
            x_left = set_positions(left, depth + 1)
            x_right = set_positions(right, depth + 1)
            x = (x_left + x_right) / 2

        positions[node] = (x, -depth)

        return x

    set_positions(0, 0)

    edges_x = []
    edges_y = []

    for node in range(tree.node_count):
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left != right:
            x, y_pos = positions[node]

            for child in (left, right):
                child_x, child_y = positions[child]

                edges_x.extend([x, child_x, None])
                edges_y.extend([y_pos, child_y, None])

    node_x = []
    node_y = []
    node_colors = []
    node_labels = []
    hover_text = []

    for node in range(tree.node_count):
        x, y_pos = positions[node]

        node_x.append(x)
        node_y.append(y_pos)

        node_samples = decision_path[:, node].nonzero()[0]
        n_samples = len(node_samples)

        left = tree.children_left[node]
        right = tree.children_right[node]
        is_leaf = left == right

        class_counts = {
            -1: np.sum(y[node_samples] == -1),
            1: np.sum(y[node_samples] == 1)
        }

        predicted_class = int(
            model.classes_[np.argmax(tree.value[node][0])]
        )

        node_colors.append(
            "#f08080" if predicted_class == 1 else "#90c990"
        )

        # Текст внутри узла
        if is_leaf:
            title = f"Лист {node}<br>Класс: {predicted_class:+d}"
        else:
            feature = features[tree.feature[node]]
            threshold = tree.threshold[node]
            title = (
                f"Узел {node}<br>"
                f"{feature} ≤ {threshold:.3f}"
            )

        node_labels.append(
            f"<b>{title}</b><br>"
            f"<span style='color:#0'>C91.1: {class_counts[1]}</span><br>"
            f"<span style='color:#0'>Z07.1: {class_counts[-1]}</span>"
        )

        # Всплывающая подсказка
        if is_leaf:
            leaf_samples = np.where(leaves == node)[0]

            error_samples = leaf_samples[
                y[leaf_samples] != y_pred[leaf_samples]
            ]

            error_list = [
                f"{applications[i]} (класс {int(y[i]):+d})"
                for i in error_samples
            ]

            errors_text = (
                "<br>".join(error_list)
                if error_list else "Нет"
            )

            hover = (
                f"<b>Лист {node}</b><br>"
                f"Предсказанный класс: {predicted_class:+d}<br>"
                f"Объектов: {n_samples} ({class_counts[1]} | {class_counts[-1]})<br>"
                f"Правильно: {n_samples - len(error_samples)}<br>"
                f"Ошибок: {len(error_samples)}<br><br>"
                f"<b>Ошибочные заявки:</b><br>{errors_text}"
            )
        else:
            hover = (
                f"<b>Узел {node}</b><br>"
                f"Условие: {feature} ≤ {threshold:.4f}<br>"
                f"Объектов: {n_samples} ({class_counts[1]} | {class_counts[-1]})<br>"
                f"Предсказываемый класс: {predicted_class:+d}"
            )

        hover_text.append(hover)

    # Рёбра дерева
    fig.add_trace(
        go.Scatter(
            x = edges_x,
            y = edges_y,
            mode = "lines",
            line = dict(color = "gray", width = 1),
            hoverinfo = "skip",
            showlegend = False
        ),
        row = 1,
        col = 1
    )

    # Узлы дерева
    fig.add_trace(
        go.Scatter(
            x = node_x,
            y = node_y,
            mode = "markers+text",
            text = node_labels,
            textposition = "middle center",
            textfont = dict(size = 10, color = "black"),
            marker = dict(
                symbol = "square",
                color = node_colors,
                size = 65,
                line = dict(color = "black", width = 1)
            ),
            hovertext = hover_text,
            hovertemplate = "%{hovertext}<extra></extra>",
            showlegend = False
        ),
        row = 1,
        col = 1
    )

    # Легенда
    fig.add_trace(
        go.Scatter(
            x = [None],
            y = [None],
            mode = "markers",
            marker = dict(color = "#ef4141", size = 10),
            name = "C91.1 (+1)"
        ),
        row = 1,
        col = 1
    )

    fig.add_trace(
        go.Scatter(
            x = [None],
            y = [None],
            mode = "markers",
            marker = dict(color = "#46d046", size = 10),
            name = "Z07.1 (-1)"
        ),
        row = 1,
        col = 1
    )

    return leaf_position[0]

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if __name__ == "__main__":
    # Обучение дерева на всей выборке
    model = DecisionTreeClassifier(**model_params)
    model.fit(X, y)
    y_pred = model.predict(X)

    cm = confusion_matrix(y, y_pred, labels = [-1, 1])
    
    # Заявки с ошибками
    pd.DataFrame({
        "application": applications,
        "true_class": y,
        "predicted_class": y_pred
    }).to_csv("decision_tree_predictions.csv", index = False)


    # Общая фигура
    fig = make_subplots(
        rows = 2, cols = 1,
        horizontal_spacing = 0.08,
        subplot_titles = [
            (
                f"Confusion Matrix: TN = {cm[0, 0]}, FP = {cm[0, 1]}, "
                    f"FN = {cm[1, 0]}, TP = {cm[1, 1]}<br>"
                f"Depth: {model.get_depth()} | "
                f"Nodes: {model.tree_.node_count} | "
                f"Leaves: {model.get_n_leaves()}"
            ),
            "Feature importance"
        ]
    )

    # Дерево решений
    n_leaf_positions = plot_decision_tree(
        fig,
        model,
        X,
        y,
        applications,
        features
    )

    # Важность признаков
    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig.add_trace(
        go.Bar(
            x = np.abs(importances[indices]),
            y = np.array(features)[indices],
            orientation = "h",
            marker = dict(color = "steelblue"),
            showlegend = False,
            hovertemplate = (
                "%{y}<br>"
                "Importance: %{x:.4f}"
                "<extra></extra>"
            )
        ),
        row = 2, col = 1
    )

    # Настройка общей фигуры
    fig.update_layout(
        title = "Decision Tree analysis",
        width = max(1600, n_leaf_positions * 65 + 500),
        height = 1500,
        hoverlabel = dict(
            bgcolor = "white",
            font_size = 12
        ),
        showlegend = True
    )

    fig.update_xaxes(
        visible = False,
        row = 1, col = 1
    )

    fig.update_yaxes(
        visible = False,
        row = 1, col = 1
    )

    fig.update_xaxes(
        title_text = "Feature importance",
        row = 2, col = 1
    )

    fig.update_yaxes(
        title_text = "Feature",
        row = 2, col = 1
    )

    fig.show()