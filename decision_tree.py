import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

from preprocessing import load_dataset


# Параметры эксперимента
criterion = "gini"
splitter = "best"
max_features = None
max_depth = None
min_samples_split = 2
min_samples_leaf = 1

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
X, y, features = load_dataset(
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
        "precision_1": [],
        "recall_1": [],
        "precision_-1": [],
        "recall_-1": []
    }

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = DecisionTreeClassifier(**model_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        scoring["accuracy"].append(
            accuracy_score(y_test, y_pred)
        )

        scoring["precision_1"].append(
            precision_score(y_test, y_pred, pos_label = 1)
        )
        scoring["recall_1"].append(
            recall_score(y_test, y_pred, pos_label = 1)
        )

        scoring["precision_-1"].append(
            precision_score(y_test, y_pred, pos_label = -1)
        )
        scoring["recall_-1"].append(
            recall_score(y_test, y_pred, pos_label = -1)
        )

    # Формирование результатов
    header = [
        "method", "criterion", "splitter", "max_features",
        "max_depth", "min_samples_split", "min_samples_leaf",
        "n_splits", "random_state",
        "drop_nan_objects", "drop_nan_features",
        "n_objects", "n_features",
        "class_-1", "class_1",
        "accuracy_mean", "accuracy_std",
        "precision_1_mean", "precision_1_std",
        "recall_1_mean", "recall_1_std",
        "precision_-1_mean", "precision_-1_std",
        "recall_-1_mean", "recall_-1_std"
    ]

    values = [
        "DecisionTree", criterion, splitter, max_features,
        max_depth, min_samples_split, min_samples_leaf,
        n_splits, random_state,
        drop_nan_objects, drop_nan_features,
        len(y), X.shape[1],
        np.sum(y == -1), np.sum(y == 1),
        np.mean(scoring["accuracy"]), np.std(scoring["accuracy"]),
        np.mean(scoring["precision_1"]), np.std(scoring["precision_1"]),
        np.mean(scoring["recall_1"]), np.std(scoring["recall_1"]),
        np.mean(scoring["precision_-1"]), np.std(scoring["precision_-1"]),
        np.mean(scoring["recall_-1"]), np.std(scoring["recall_-1"])
    ]

    print("\t".join(map(str, header)))
    print("\t".join(
        f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
        for value in values
    ))

    
import plotly.graph_objects as go
from sklearn.tree import export_text
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
if __name__ == "2__main__":
    # Обучение дерева на всей выборке
    model = DecisionTreeClassifier(**model_params)
    model.fit(X, y)

    tree = model.tree_

    # ============================================================
    # 1. Характеристики дерева
    # ============================================================

    tree_depth = model.get_depth()
    n_nodes = tree.node_count
    n_leaves = model.get_n_leaves()

    print("\nDecision Tree characteristics")
    print("-" * 45)
    print(f"Depth:  {tree_depth}")
    print(f"Nodes:  {n_nodes}")
    print(f"Leaves: {n_leaves}")

    # ============================================================
    # 2. Правила классификации
    # ============================================================

    rules = export_text(
        model,
        feature_names = list(features)
    )

    print("\nDecision rules")
    print("-" * 45)
    print(rules)

    # ============================================================
    # 3. Визуализация структуры дерева
    # ============================================================

    # Визуализация структуры дерева
    plt.figure(figsize = (24, 12))

    plot_tree(
        model,
        feature_names = list(features),
        class_names = ["-1", "+1"],
        filled = True,
        rounded = True,
        proportion = False,
        fontsize = 8
    )

    plt.title(
        f"Decision Tree\n"
        f"Depth: {tree_depth} | "
        f"Nodes: {n_nodes} | "
        f"Leaves: {n_leaves}"
    )

    plt.tight_layout()
    plt.show()

    # ============================================================
    # 4. Важность признаков
    # ============================================================

    importances = model.feature_importances_

    indices = np.argsort(importances)

    fig_importance = go.Figure(go.Bar(
        x = importances[indices],
        y = np.array(features)[indices],
        orientation = "h",
        marker = dict(color = "steelblue")
    ))

    fig_importance.update_layout(
        title = (
            "Decision Tree: feature importance"
        ),
        xaxis_title = "Feature importance",
        yaxis_title = "Feature",
        height = max(600, len(features) * 25)
    )

    fig_importance.show()