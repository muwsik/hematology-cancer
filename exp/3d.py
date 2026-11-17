import pandas as pd
import numpy as np
import plotly.express as px
from preprocessing import load_dataset



if __name__ == "__main__":
    svm = pd.read_csv("svm_predictions.csv")
    tree = pd.read_csv("decision_tree_predictions.csv")

    results = svm.merge(
        tree,
        on = "application",
        suffixes = ("_svm", "_tree"),
        validate = "one_to_one"
    )

    assert (
        results["true_class_svm"] == results["true_class_tree"]
    ).all()

    results["true_class"] = results["true_class_svm"]

    results["agreement"] = (
        results["predicted_class_svm"] ==
        results["predicted_class_tree"]
    )

    # Оба метода предсказали одинаково
    both_correct = results[
        results["agreement"] &
        (results["predicted_class_svm"] == results["true_class"])
    ]

    both_wrong = results[
        results["agreement"] &
        (results["predicted_class_svm"] != results["true_class"])
    ]

    # Только один метод классифицировал правильно
    svm_better = results[
        (results["predicted_class_svm"] == results["true_class"]) &
        (results["predicted_class_tree"] != results["true_class"])
    ]

    tree_better = results[
        (results["predicted_class_tree"] == results["true_class"]) &
        (results["predicted_class_svm"] != results["true_class"])
    ]

    print("\nСводка:")
    print(f"Оба правильно: {len(both_correct)}")
    print(f"Оба ошиблись: {len(both_wrong)}")
    print(f"Правильно только SVM: {len(svm_better)}")
    print(f"Правильно только Decision Tree: {len(tree_better)}")

    print("\nОба метода ошиблись:")
    print(*both_wrong["application"].tolist())

    print("\nПравильно только SVM:")
    print(*svm_better["application"].tolist())

    print("\nПравильно только Decision Tree:")
    print(*tree_better["application"].tolist())


def plot_3d_features(X, y, features, applications):
    class_names = {
        "-1": "Class Z01.7",
        "1": "Class C91.1"
    }

    fig = px.scatter_3d(
        x = X[:, 0],
        y = X[:, 1],
        z = X[:, 2],
        color = y.astype(str),
        color_discrete_map = {
            "-1": "seagreen",
            "1": "red"
        },
        hover_name = applications.astype(str),
        labels = {
            "x": features[0],
            "y": features[1],
            "z": features[2],
            "color": "Класс"
        },
        category_orders = {
            "color": ["-1", "1"]
        },
        opacity = 0.7
    )

    fig.for_each_trace(
        lambda trace: trace.update(
            name = class_names[trace.name]
        )
    )

    fig.update_layout(
        scene = {
            "xaxis_title": features[0],
            "yaxis_title": features[1],
            "zaxis_title": features[2]
        },
        legend_title = "Класс"
    )

    return fig


if __name__ == "__main__":
    X, y, features, applications = load_dataset(
        drop_nan_objects = False,
        drop_nan_features = True
    )

    features_3d = [
        "Лимфоциты (LYM%)",
        "Тромбоциты (PLT)",
        "Лейкоциты (WBC)"
    ]

    indices_3d = [
        np.where(features == feature)[0][0]
        for feature in features_3d
    ]

    fig = plot_3d_features(
        X[:, indices_3d],
        y,
        features_3d,
        applications
    )

    fig.show()