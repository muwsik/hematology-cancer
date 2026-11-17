import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from plotly.subplots import make_subplots

import plotly.graph_objects as go

from preprocessing import DROP_COLUMNS, read_data, prepare_data




st.set_page_config(
    page_title = "Hematology",
    page_icon = "🔬",
    layout = "wide"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Подготовка данных",
    "Дерево решений",    
    "SVM-классификатор",
    "Согласование решений"
])


with tab1:
    # Чтение исходных данных
    read_data_cached = st.cache_data(read_data)

    raw_c = read_data_cached("data/C91.1.xlsx")
    raw_z = read_data_cached("data/Z01.7.xlsx")

    raw_data = pd.concat([raw_c, raw_z], ignore_index = True)
    raw_data["№ заявки"] = raw_data["№ заявки"].astype("int64")

    # Исходные признаки
    raw_features = raw_data.drop(
        columns = ["№ заявки", "Диагноз"]
    ).columns.to_numpy()

    # Информация об исключаемых признаках
    excluded_features = [
        column for column in DROP_COLUMNS
        if column in raw_features
    ]

    # Анализ пропусков в исходных данных
    raw_missing = raw_data[raw_features].isna()

    missing_objects = raw_missing.any(axis = 1)
    missing_features = raw_missing.any(axis = 0)

    # Подготовка матриц признаков
    X1, y1, features1, applications1 = prepare_data(
        raw_c, DROP_COLUMNS
    )

    X2, y2, features2, applications2 = prepare_data(
        raw_z, DROP_COLUMNS
    )

    if not np.array_equal(features1, features2):
        raise ValueError("Наборы или порядок признаков не совпадают")

    X_raw = np.concatenate([X1, X2], axis = 0)
    y = np.concatenate([y1, y2], axis = 0)

    features_raw = features1
    applications = np.concatenate(
        [applications1, applications2],
        axis = 0
    )

    # Удаление признаков с пропусками
    valid_features = np.isfinite(X_raw).all(axis = 0)

    missing_feature_names = features_raw[~valid_features]

    X = X_raw[:, valid_features]
    features = features_raw[valid_features]

    #
    #
    # Исходная статистика
    st.subheader("1. Исходные данные")

    class_counts = raw_data["Диагноз"].value_counts()
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Заявок всего", len(raw_data))
    col2.metric("Заявок C91.1", (raw_data["Диагноз"] == "C91.1").sum())
    col3.metric("Заявок Z07.1", (raw_data["Диагноз"] == "Z01.7").sum())
    col4.metric("Признаков", len(raw_features))
    col5.metric("Заявок с пропусками", int(missing_objects.sum()))
    col6.metric("Всего пропусков", int(raw_missing.sum().sum()))

    # Пропуски по признакам
    with st.expander("**Пропуски по признакам**"):
        missing_feature_table = (
            pd.DataFrame({
                "Признак": raw_features,
                "Количество пропусков": raw_missing.sum(axis = 0).values
            })
            .query("`Количество пропусков` > 0")
            .sort_values("Количество пропусков", ascending = False)
        )

        if not missing_feature_table.empty:
            st.dataframe(
                missing_feature_table,
                hide_index = True,
                width = "stretch"
            )
        else:
            st.success("Пропуски отсутствуют.")

    # Пропуски по заявкам    
    with st.expander("**Заявки с отсутствующими значениями**"):
        missing_rows = []

        for i in np.flatnonzero(missing_objects.to_numpy()):
            missing_rows.append({
                "№ заявки": raw_data.iloc[i]["№ заявки"],
                "Класс": raw_data.iloc[i]["Диагноз"],
                "Количество пропусков": int(raw_missing.iloc[i].sum()),
                "Отсутствующие признаки": ", ".join(
                    raw_features[raw_missing.iloc[i].to_numpy()]
                )
            })

        if missing_rows:
            st.dataframe(
                pd.DataFrame(missing_rows),
                hide_index = True,
                width = "stretch"
            )
        else:
            st.success("Заявки с пропусками отсутствуют.")


    # Результаты подготовки
    st.subheader("2. Данные после подготовки")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Исключены вручную**")
        st.caption(f"{len(excluded_features)} признаков")

        if excluded_features:
            st.markdown("\n".join(
                f"- {feature}" for feature in excluded_features
            ))
        else:
            st.write("Нет")

    with col2:
        st.write("**Исключены из-за пропусков**")
        st.caption(f"{len(missing_feature_names)} признаков")

        if len(missing_feature_names):
            st.markdown("\n".join(
                f"- {feature}" for feature in missing_feature_names
            ))
        else:
            st.write("Нет")

    with col3:
        st.write("**Оставшиеся признаки**")
        st.caption(f"{len(features)} признаков")

        st.markdown("\n".join(
            f"- {feature}" for feature in features
        ))


    st.subheader("3. Итоговая выборка")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Количество заявок", X.shape[0])
    col2.metric("Заявок C91.1", (y == 1).sum())
    col3.metric("Заявок Z01.7", (y == -1).sum())
    col4.metric("Количество признаков", X.shape[1])
    col5.metric("Количество классов", len(np.unique(y)))





def plot_decision_tree(fig, model, X, y, features):
    tree = model.tree_
    decision_path = model.decision_path(X)

    positions = {}
    leaf_position = [0]

    # Расположение узлов
    def set_positions(node, depth):
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left == right:
            x = leaf_position[0] * 2.0
            leaf_position[0] += 1
        else:
            x_left = set_positions(left, depth + 1)
            x_right = set_positions(right, depth + 1)
            x = (x_left + x_right) / 2

        positions[node] = (x, -depth * 2.0)
        return x

    set_positions(0, 0)

    edge_x = []
    edge_y = []

    # Формирование рёбер
    for node in range(tree.node_count):
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left != right:
            x, y_pos = positions[node]

            for child in (left, right):
                child_x, child_y = positions[child]
                edge_x.extend([x, child_x, None])
                edge_y.extend([y_pos, child_y, None])

    # Рёбра добавляем первыми
    fig.add_trace(
        go.Scatter(
            x = edge_x,
            y = edge_y,
            mode = "lines",
            line = dict(color = "gray", width = 1),
            hoverinfo = "skip",
            showlegend = False
        )
    )

    node_x = []
    node_y = []
    node_labels = []
    node_ids = []

    # Формирование узлов
    for node in range(tree.node_count):
        x, y_pos = positions[node]

        left = tree.children_left[node]
        right = tree.children_right[node]
        is_leaf = left == right

        # Количество объектов каждого класса
        node_samples = decision_path[:, node].nonzero()[0]

        n_c = np.sum(y[node_samples] == 1)
        n_z = np.sum(y[node_samples] == -1)

        predicted_class = int(
            model.classes_[np.argmax(tree.value[node][0])]
        )

        color = "#f08080" if predicted_class == 1 else "#90c990"

        # Содержимое узла
        if is_leaf:
            class_name = "C91.1" if predicted_class == 1 else "Z01.7"

            label = (
                f"<b>{class_name}</b><br>"
                f"C91.1: {n_c}<br>"
                f"Z01.7: {n_z}"
            )

            box_width = 1.4
            box_height = 1.4

        else:
            feature = features[tree.feature[node]]
            threshold = tree.threshold[node]
            rule = f"{feature} ≤ {threshold:.3f}"

            label = (
                f"<b>{rule}</b><br>"
                f"C91.1: {n_c}<br>"
                f"Z01.7: {n_z}"
            )

            box_width = max(1.8, len(rule) * 0.16)
            box_height = 1.2

        # Прямоугольник или квадрат
        fig.add_trace(
            go.Scatter(
                x = [
                    x - box_width / 2,
                    x + box_width / 2,
                    x + box_width / 2,
                    x - box_width / 2,
                    x - box_width / 2
                ],
                y = [
                    y_pos - box_height / 2,
                    y_pos - box_height / 2,
                    y_pos + box_height / 2,
                    y_pos + box_height / 2,
                    y_pos - box_height / 2
                ],
                mode = "lines",
                fill = "toself",
                fillcolor = color,
                line = dict(color = "black", width = 1),
                hoverinfo = "skip",
                showlegend = False
            )
        )

        node_x.append(x)
        node_y.append(y_pos)
        node_labels.append(label)
        node_ids.append(node)

    # Подписи и маркеры для выбора узлов
    fig.add_trace(
        go.Scatter(
            x = node_x,
            y = node_y,
            mode = "markers+text",
            text = node_labels,
            textposition = "middle center",
            textfont = dict(
                color = "black",
                size = 11
            ),
            marker = dict(
                size = 35,
                color = "rgba(0, 0, 0, 0.01)",
                line = dict(
                    color = "rgba(0, 0, 0, 0.01)",
                    width = 1
                )
            ),
            customdata = node_ids,
            hovertemplate = "<extra></extra>",
            showlegend = False
        )
    )

    return leaf_position[0]


with tab2:
    model_params = {
        "criterion": "gini",
        "splitter": "best",
        "max_features": None,
        "max_depth": None,
        "min_samples_split": 25,
        "min_samples_leaf": 10,
        "random_state": 42
    }

    # Обучение дерева
    model = DecisionTreeClassifier(**model_params)
    model.fit(X, y)

    tree_y_pred = model.predict(X)

    # Построение графика
    fig = go.Figure()

    n_leaf_positions = plot_decision_tree(
        fig,
        model,
        X,
        y,
        features
    )

    fig.update_layout(
        height = 650,
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        font = dict(color = "black"),
        margin = dict(t = 50, b = 30, l = 30, r = 30)
    )

    fig.update_xaxes(visible = False)
    fig.update_yaxes(
        visible = False,
        scaleanchor = "x",
        scaleratio = 1
    )

    # Отображение графика с выбором узлов
    event = st.plotly_chart(
        fig,
        width = "stretch",
        on_select = "rerun",
        selection_mode = "points",
        key = "decision_tree"
    )

    # Обработка выбранного узла
    # Обработка выбранного листа
    if event and event.selection.points:
        selected_node = int(
            event.selection.points[-1]["customdata"]
        )

        is_leaf = (
            model.tree_.children_left[selected_node]
            == model.tree_.children_right[selected_node]
        )

        if is_leaf:
            mask = model.apply(X) == selected_node

            # Основная информация о заявках
            selected_applications = pd.DataFrame({
                "№ заявки": applications[mask],
                "Класс": np.where(
                    y[mask] == 1, "C91.1", "Z01.7"
                ),
                "Предсказанный класс": np.where(
                    model.predict(X[mask]) == 1, "C91.1", "Z01.7"
                )
            })

            # Значения всех признаков
            selected_features = pd.DataFrame(
                X[mask],
                columns = features
            )

            # Объединение таблиц
            selected_applications = pd.concat(
                [
                    selected_applications.reset_index(drop = True),
                    selected_features.reset_index(drop = True)
                ],
                axis = 1
            )

            st.subheader(f"Заявки в выбраном листе")

            st.dataframe(
                selected_applications,
                width = "stretch",
                hide_index = True
            )

    #
    #
    # Влияние признаков
    feature_importance = pd.DataFrame({
        "Признак": features,
        "Важность": model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        "Важность",
        ascending = False
    )

    st.subheader("Влияние признаков")

    fig = go.Figure(go.Bar(
        x = feature_importance["Важность"],
        y = feature_importance["Признак"],
        orientation = "h",
        marker_color = "steelblue",
        hoverinfo = 'skip'
    ))

    fig.update_layout(
        height = max(400, len(features) * 22),
        xaxis_title = "Важность признака",
        yaxis_title = None,
        yaxis = dict(autorange = "reversed"),
        margin = dict(t = 20, b = 30, l = 20, r = 20)
    )

    st.plotly_chart(fig, width = "stretch")





with tab3:
    # Обучение модели на всей выборке
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel = "linear", C = 10.0)
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    svm_y_pred = y_pred.copy()
    decision_values = model.decision_function(X_scaled)

    # Основные показатели
    accuracy = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, decision_values)

    #
    #
    #
    # Гистограмма распределения выходов SVM
    st.subheader("Распределение выходов классификатора")

    bin_width = 0.5

    bin_min = np.floor(decision_values.min() / bin_width) * bin_width
    bin_max = np.ceil(decision_values.max() / bin_width) * bin_width

    bins = np.arange(
        bin_min,
        bin_max + bin_width,
        bin_width
    )

    hist_neg, _ = np.histogram(
        decision_values[y == -1],
        bins = bins
    )

    hist_pos, _ = np.histogram(
        decision_values[y == 1],
        bins = bins
    )

    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x = bin_centers,
        y = hist_neg,
        width = bin_width-0.05,
        name = "Z01.7",
        marker_color = "green",
        opacity = 0.6,
        hoverinfo = "name"
    ))

    fig.add_trace(go.Bar(
        x = bin_centers,
        y = hist_pos,
        width = bin_width-0.05,
        name = "C91.1",
        marker_color = "red",
        opacity = 0.6,
        hoverinfo = "name"
    ))

    fig.add_vline(
        x = 0,
        line_dash = "dash",
        line_color = "black",
        annotation_text = "Граница классификации"
    )

    fig.update_layout(
        barmode = "overlay",
        bargap = 0,
        xaxis_title = "Значение функции принятия решения",
        yaxis_title = "Количество заявок",
        legend_title = "Класс",
        height = 600,
        margin = dict(t = 10, b = 10, l = 10, r = 10)
    )

    event = st.plotly_chart(
        fig,
        width = "stretch",
        on_select = "rerun",
        selection_mode = "points",
        key = "svm_histogram"
    )

    selected_points = event.selection.points
    if selected_points:
        selected_x = selected_points[0]["x"]

        bin_index = np.argmin(np.abs(bin_centers - selected_x))

        left = bins[bin_index]
        right = bins[bin_index + 1]

        # Выбор заявок из интервала
        mask = (
            (decision_values >= left) &
            (
                (decision_values < right) |
                (
                    (bin_index == len(bins) - 2) &
                    (decision_values <= right)
                )
            )
        )

        selected_applications = pd.DataFrame({
            "№ заявки": applications[mask],
            "Класс": np.where(y[mask] == 1, "C91.1", "Z01.7"),
            "Предсказанный класс": np.where(
                y_pred[mask] == 1, "C91.1", "Z01.7"
            )
        })

        # Добавляем значения признаков
        selected_features = pd.DataFrame(
            X[mask],
            columns = features
        )

        selected_applications = pd.concat(
            [
                selected_applications.reset_index(drop = True),
                selected_features.reset_index(drop = True)
            ],
            axis = 1
        ).sort_values("Класс")

        st.subheader(
            f"Заявки в выделенном интервале"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Всего заявок", len(selected_applications))
        col2.metric(
            "Z01.7",
            (selected_applications["Класс"] == "Z01.7").sum()
        )
        col3.metric(
            "C91.1",
            (selected_applications["Класс"] == "C91.1").sum()
        )

        st.dataframe(
            selected_applications,
            hide_index = True,
            width = "stretch"
        )

    #
    #
    #
    # Веса признаков линейной SVM
    st.subheader("Веса признаков")
    
    weights = model.coef_.ravel()
    indices = np.argsort(np.abs(weights))

    fig_weights = go.Figure()

    fig_weights.add_trace(go.Bar(
        x = np.abs(weights[indices]),
        y = np.array(features)[indices],
        orientation = "h",
        marker_color = "steelblue",
        hoverinfo = "skip"
    ))

    fig_weights.update_layout(
        xaxis_title = "Абсолютное значение веса",
        yaxis_title = "Признак",
        height = max(400, len(features) * 25),
        showlegend = False,
        margin = dict(t = 10, b = 10, l = 10, r = 10)
    )

    st.plotly_chart(
        fig_weights,
        width = "stretch"
    )


    #
    #
    # Результаты классификации
    with st.expander("Результаты классификации"):
        st.caption("Показатели рассчитаны на обучающей выборке.")

        st.dataframe(
            pd.DataFrame({
                "Accuracy": [accuracy],
                "ROC-AUC": [roc_auc],
                "C91.1 Precision": [
                    precision_score(y, y_pred, pos_label = 1, zero_division = 0)
                ],
                "C91.1 Recall": [
                    recall_score(y, y_pred, pos_label = 1, zero_division = 0)
                ],
                "C91.1 F1": [
                    f1_score(y, y_pred, pos_label = 1, zero_division = 0)
                ],
                "Z01.7 Precision": [
                    precision_score(y, y_pred, pos_label = -1, zero_division = 0)
                ],
                "Z01.7 Recall": [
                    recall_score(y, y_pred, pos_label = -1, zero_division = 0)
                ],
                "Z01.7 F1": [
                    f1_score(y, y_pred, pos_label = -1, zero_division = 0)
                ]
            }).round(4),
            hide_index = True,
            width = "stretch"
        )

        # Матрица ошибок
        st.subheader("Матрица ошибок")

        cm = confusion_matrix(y, y_pred, labels = [-1, 1])

        st.dataframe(
            pd.DataFrame(
                cm,
                index = ["Фактически Z01.7", "Фактически C91.1"],
                columns = ["Предсказано Z01.7", "Предсказано C91.1"]
            ),
            width = "stretch"
        )

    # Кросс-валидация
    with st.expander("Дополнительная оценка качества: кросс-валидация"):
        random_state = np.random.default_rng().integers(0, 2**32).item()

        st.write(
            "Пятикратная стратифицированная кросс-валидация. "
            "Стандартизация выполняется отдельно по каждому блоку."
        )
        st.caption(f"random_state = {random_state}")

        if st.button("Запустить кросс-валидацию", key = "svm_cv_button"):
            cv = StratifiedKFold(
                n_splits = 5,
                shuffle = True,
                random_state = random_state
            )

            cv_results = {
                "Accuracy": [],
                "ROC-AUC": [],
                "C91.1 Precision": [],
                "C91.1 Recall": [],
                "C91.1 F1": [],
                "Z01.7 Precision": [],
                "Z01.7 Recall": [],
                "Z01.7 F1": []
            }

            for train_index, test_index in cv.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                fold_scaler = StandardScaler()
                X_train_scaled = fold_scaler.fit_transform(X_train)
                X_test_scaled = fold_scaler.transform(X_test)

                fold_model = SVC(kernel = "linear", C = 10.0)
                fold_model.fit(X_train_scaled, y_train)

                y_fold_pred = fold_model.predict(X_test_scaled)
                fold_decision = fold_model.decision_function(X_test_scaled)

                cv_results["Accuracy"].append(
                    accuracy_score(y_test, y_fold_pred)
                )
                cv_results["ROC-AUC"].append(
                    roc_auc_score(y_test, fold_decision)
                )

                for label, prefix in [(1, "C91.1"), (-1, "Z01.7")]:
                    cv_results[f"{prefix} Precision"].append(
                        precision_score(
                            y_test,
                            y_fold_pred,
                            pos_label = label,
                            zero_division = 0
                        )
                    )
                    cv_results[f"{prefix} Recall"].append(
                        recall_score(
                            y_test,
                            y_fold_pred,
                            pos_label = label,
                            zero_division = 0
                        )
                    )
                    cv_results[f"{prefix} F1"].append(
                        f1_score(
                            y_test,
                            y_fold_pred,
                            pos_label = label,
                            zero_division = 0
                        )
                    )

            # Сводные результаты кросс-валидации
            cv_metrics = pd.DataFrame({
                name: [
                    f"{np.mean(values):.4f} ± {np.std(values, ddof = 1):.4f}"
                ]
                for name, values in cv_results.items()
            })

            st.dataframe(
                cv_metrics,
                hide_index = True,
                width = "stretch"
            )





with tab4:
    # Формирование общей таблицы результатов
    comparison = pd.DataFrame({
        "№ заявки": applications,
        "Фактический класс": np.where(
            y == 1, "C91.1", "Z01.7"
        ),
        "SVM": np.where(
            svm_y_pred == 1, "C91.1", "Z01.7"
        ),
        "Дерево решений": np.where(
            tree_y_pred == 1, "C91.1", "Z01.7"
        )
    })

    # Совпадение решений
    comparison["Решения совпали"] = (
        comparison["SVM"] == comparison["Дерево решений"]
    )

    # Правильность решений каждого классификатора
    comparison["SVM"] = comparison["SVM"].astype(str)
    comparison["Дерево решений"] = comparison["Дерево решений"].astype(str)

    comparison["SVM: результат"] = np.where(
        svm_y_pred == y, "Верно", "Ошибка"
    )

    comparison["Дерево: результат"] = np.where(
        tree_y_pred == y, "Верно", "Ошибка"
    )

    # Разделение заявок
    agreed = comparison[
        comparison["Решения совпали"]
    ].copy()

    disagreed = comparison[
        ~comparison["Решения совпали"]
    ].copy()

    # Общая статистика
    n_total = len(comparison)
    n_agreed = len(agreed)
    n_disagreed = len(disagreed)

    col1, col2, col3 = st.columns(3)

    col1.metric("Всего заявок", n_total)
    col2.metric(
        "Совпавшие решения",
        n_agreed,
        f"{n_agreed / n_total * 100:.2f}%"
    )
    col3.metric(
        "Различающиеся решения",
        n_disagreed,
        f"{n_disagreed / n_total * 100:.2f}%"
    )

    # Совпавшие решения
    st.subheader("1. Совпавшие решения")

    st.caption(
        "Оба классификатора присвоили заявке одинаковый класс."
    )

    st.dataframe(
        agreed.drop(columns = ["Решения совпали"]),
        width = "stretch",
        hide_index = True
    )

    # Различающиеся решения
    st.subheader("2. Различающиеся решения")

    st.caption(
        "Классификаторы присвоили заявке разные классы."
    )

    if not disagreed.empty:
        st.dataframe(
            disagreed.drop(columns = ["Решения совпали"]),
            width = "stretch",
            hide_index = True
        )
    else:
        st.success("Все решения классификаторов совпадают.")