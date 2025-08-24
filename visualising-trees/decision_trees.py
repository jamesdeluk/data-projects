import marimo

__generated_with = "0.14.16"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import time
    import timeit
    import pandas as pd
    import numpy as np
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split, ShuffleSplit, RepeatedKFold, cross_validate
    from sklearn.tree import DecisionTreeRegressor, plot_tree
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score
    from skopt import BayesSearchCV
    import plotly.graph_objs as go
    import matplotlib.pyplot as plt
    import networkx as nx


@app.cell
def _():
    data = fetch_california_housing()
    return (data,)


@app.cell
def _(data):
    california_housing_df = pd.DataFrame(data.data, columns=data.feature_names)
    target = data.target_names[0]
    california_housing_df[target] = data.target
    chosen = pd.DataFrame(california_housing_df.iloc[-1]).T.reset_index(drop=True)
    california_housing_df = california_housing_df[:-1]
    return california_housing_df, chosen, target


@app.cell
def _(california_housing_df):
    california_housing_df.iloc[:5]
    return


@app.cell
def _(chosen):
    chosen
    return


@app.cell
def _(california_housing_df, target):
    X = california_housing_df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']]
    y = california_housing_df[target]
    return X, y


@app.cell
def _(y):
    y.describe()
    return


@app.cell
def _(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell
def _():
    dt_shallow = DecisionTreeRegressor(max_depth=3, random_state=42)
    dt_deep = DecisionTreeRegressor(max_depth=None, random_state=42)
    dt_prune = DecisionTreeRegressor(max_depth=None, ccp_alpha=0.005, random_state=42)
    return dt_deep, dt_prune, dt_shallow


@app.cell
def _(X, X_test, X_train, chosen, target, y_test, y_train):
    def fitpred_single(dt, timing_count=1):
        fit_time = timeit.timeit(lambda: dt.fit(X_train, y_train), number=timing_count)
        print(f"Time to fit:\t\t{fit_time/timing_count}")

        pred_time = timeit.timeit(lambda: dt.predict(X_test), number=timing_count)
        print(f"Time to predict:\t{pred_time/timing_count}")
        dt_predict = dt.predict(X_test)

        print()

        print(f"MAE:\t{mean_absolute_error(y_test, dt_predict)}")
        print(f"MAPE:\t{mean_absolute_percentage_error(y_test, dt_predict)}")
        print(f"MSE:\t{mean_squared_error(y_test, dt_predict)}")
        print(f"RMSE:\t{root_mean_squared_error(y_test, dt_predict)}")
        print(f"R2:\t\t{r2_score(y_test, dt_predict)}")

        print()

        print(f"Chosen prediction:\t{float(dt.predict(chosen.drop(columns=target))[0])}")
        print(f"Chosen actual:\t\t{chosen[target].iloc[0]}")

        plt.figure(figsize=(12,4))
        plot_tree(dt, feature_names=X.columns, label='all', filled=True, impurity=True, proportion=True, rounded=True)
        plt.show()
    return (fitpred_single,)


@app.cell
def _(dt_shallow, fitpred_single):
    fitpred_single(dt_shallow)
    return


@app.cell
def _(dt_deep, fitpred_single):
    fitpred_single(dt_deep)
    return


@app.cell
def _(X, dt_deep):
    plot_branch_vertical(dt_deep, X.iloc[0].values, feature_names=X.columns)
    return


@app.cell
def _(dt_prune, fitpred_single):
    fitpred_single(dt_prune)
    return


@app.cell
def _(X, y):
    def fitpred_loop(dt, loops=1000):
        _start = time.time()

        maes = []
        mapes = []
        mses = []
        rmses = []
        r2s = []

        for _ in mo.status.progress_bar(range(loops)):
            X_train_loop, X_test_loop, y_train_loop, y_test_loop = train_test_split(X, y)
            dt.fit(X_train_loop, y_train_loop)
            _df_predict = dt.predict(X_test_loop)
            maes.append(mean_absolute_error(y_test_loop, _df_predict))
            mapes.append(mean_absolute_percentage_error(y_test_loop, _df_predict))
            mses.append(mean_squared_error(y_test_loop, _df_predict))
            rmses.append(root_mean_squared_error(y_test_loop, _df_predict))
            r2s.append(r2_score(y_test_loop, _df_predict))

        _end = time.time()
        _time = _end - _start

        return maes, mapes, mses, rmses, r2s, _time
    return (fitpred_loop,)


@app.function
def print_info(dt_suffix):
    dt_name = f"dt_{dt_suffix}"
    dt = globals()[dt_name]
    print(f"Depth:\t\t{dt.get_depth()}")
    print(f"Leaves:\t\t{dt.get_n_leaves()}")
    print(f"Nodes:\t\t{dt.tree_.node_count}")
    print(f"Params:\t\t{dt.get_params()}")
    print()
    maes = globals()[f"maes_{dt_suffix}"]
    print(f"MAE mean:\t{np.mean(maes):.3f}, std: {np.std(maes):.3f}")
    mapes = globals()[f"mapes_{dt_suffix}"]
    print(f"MAPE mean:\t{np.mean(mapes):.3f}, std: {np.std(mapes):.3f}")
    mses = globals()[f"mses_{dt_suffix}"]
    print(f"MSE mean:\t{np.mean(mses):.3f}, std: {np.std(mses):.3f}")
    rmses = globals()[f"rmses_{dt_suffix}"]
    print(f"RMSE mean:\t{np.mean(rmses):.3f}, std: {np.std(rmses):.3f}")
    r2s = globals()[f"r2s_{dt_suffix}"]
    print(f"R2 mean:\t{np.mean(r2s):.3f}, std: {np.std(r2s):.3f}")
    print()
    time = globals()[f"time_{dt_suffix}"]
    print(f"Time:\t\t{time:.1f}s")


@app.cell
def _(dt_shallow, fitpred_loop):
    maes_shallow, mapes_shallow, mses_shallow, rmses_shallow, r2s_shallow, time_shallow = fitpred_loop(dt_shallow)
    return (
        maes_shallow,
        mapes_shallow,
        mses_shallow,
        r2s_shallow,
        rmses_shallow,
    )


@app.cell
def _():
    print_info("shallow")
    return


@app.cell
def _(dt_deep, fitpred_loop):
    maes_deep, mapes_deep, mses_deep, rmses_deep, r2s_deep, time_deep = fitpred_loop(dt_deep)
    return maes_deep, mapes_deep, mses_deep, r2s_deep, rmses_deep


@app.cell
def _():
    print_info("deep")
    return


@app.function(hide_code=True)
def plot_branch_vertical(tree, X_sample, feature_names=None):
    plt.figure(figsize=(4, 12))
    tree_ = tree.tree_
    node_indicator = tree.decision_path([X_sample])
    node_index = node_indicator.indices

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(tree_.n_features)]

    G = nx.DiGraph()
    for i, node_id in enumerate(node_index[:-1]):
        next_node_id = node_index[i+1]
        if tree_.children_left[node_id] == tree_.children_right[node_id]:
            label = f"Leaf {node_id}\nValue: {tree_.value[node_id][0]}"
        else:
            feature = feature_names[tree_.feature[node_id]]
            threshold = tree_.threshold[node_id]
            label = f"Node {node_id}\n{feature} <= {threshold:.2f}\nSample: {X_sample[tree_.feature[node_id]]:.2f}"
        G.add_node(node_id, label=label)
        G.add_edge(node_id, next_node_id)

    leaf_id = node_index[-1]
    label = f"Leaf {leaf_id}\nValue: {tree_.value[leaf_id][0]}"
    G.add_node(leaf_id, label=label)

    try:
        pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
    except:
        print("graphviz_layout not available, falling back to spring_layout")
        pos = nx.spring_layout(G)

    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=False, arrows=True)
    nx.draw_networkx_labels(G, pos, labels)
    plt.show()


@app.cell
def _(dt_prune, fitpred_loop):
    maes_prune, mapes_prune, mses_prune, rmses_prune, r2s_prune, time_prune = fitpred_loop(dt_prune)
    return maes_prune, mapes_prune, mses_prune, r2s_prune, rmses_prune


@app.cell
def _():
    print_info("prune")
    return


@app.cell
def _(
    maes_deep,
    maes_prune,
    maes_shallow,
    mapes_deep,
    mapes_prune,
    mapes_shallow,
    mses_deep,
    mses_prune,
    mses_shallow,
    r2s_deep,
    r2s_prune,
    r2s_shallow,
    rmses_deep,
    rmses_prune,
    rmses_shallow,
):
    _trace_maes_shallow = go.Box(y=maes_shallow, name='MAEs, Shallow Tree', marker_color='blue', width=0.6)
    _trace_maes_deep = go.Box(y=maes_deep, name='MAEs, Deep Tree', marker_color='green', width=0.6)
    _trace_maes_prune = go.Box(y=maes_prune, name='MAEs, Prune Tree', marker_color='red', width=0.6)
    _trace_mapes_shallow = go.Box(y=mapes_shallow, name='MAPEs, Shallow Tree', marker_color='blue', width=0.6)
    _trace_mapes_deep = go.Box(y=mapes_deep, name='MAPEs, Deep Tree', marker_color='green', width=0.6)
    _trace_mapes_prune = go.Box(y=mapes_prune, name='MAPEs, Prune Tree', marker_color='red', width=0.6)
    _trace_mses_shallow = go.Box(y=mses_shallow, name='MSEs, Shallow Tree', marker_color='blue', width=0.6)
    _trace_mses_deep = go.Box(y=mses_deep, name='MSEs, Deep Tree', marker_color='green', width=0.6)
    _trace_mses_prune = go.Box(y=mses_prune, name='MSEs, Prune Tree', marker_color='red', width=0.6)
    _trace_rmses_shallow = go.Box(y=rmses_shallow, name='RMSEs, Shallow Tree', marker_color='blue', width=0.6)
    _trace_rmses_deep = go.Box(y=rmses_deep, name='RMSEs, Deep Tree', marker_color='green', width=0.6)
    _trace_rmses_prune = go.Box(y=rmses_prune, name='RMSEs, Prune Tree', marker_color='red', width=0.6)
    _trace_r2s_shallow = go.Box(y=r2s_shallow, name='R2s, Shallow Tree', marker_color='blue', width=0.6)
    _trace_r2s_deep = go.Box(y=r2s_deep, name='R2s, Deep Tree', marker_color='green', width=0.6)
    _trace_r2s_prune = go.Box(y=r2s_prune, name='R2s, Prune Tree', marker_color='red', width=0.6)

    _data = [_trace_maes_shallow, _trace_maes_deep, _trace_maes_prune, _trace_mapes_shallow, _trace_mapes_deep, _trace_mapes_prune, _trace_mses_shallow, _trace_mses_deep, _trace_mses_prune, _trace_rmses_shallow, _trace_rmses_deep, _trace_rmses_prune, _trace_r2s_shallow, _trace_r2s_deep, _trace_r2s_prune]

    _layout = go.Layout(
        title='Comparison of Metrics for Decision Tree Variants',
        boxmode='group',
        xaxis=dict(tickangle=90),
    )

    go.Figure(data=_data, layout=_layout)
    return


@app.cell
def _(maes_deep, maes_prune, maes_shallow):
    _maes_shallow_norm = np.array(maes_shallow) - np.mean(maes_shallow)
    _maes_deep_norm = np.array(maes_deep) - np.mean(maes_deep)
    _maes_prune_norm = np.array(maes_prune) - np.mean(maes_prune)

    _trace_shallow = go.Box(y=_maes_shallow_norm, name='Shallow Tree', marker_color='blue', width=0.6)
    _trace_deep = go.Box(y=_maes_deep_norm, name='Deep Tree', marker_color='green', width=0.6)
    _trace_prune = go.Box(y=_maes_prune_norm, name='Prune Tree', marker_color='red', width=0.6)

    _data = [_trace_shallow, _trace_deep, _trace_prune]

    _layout = go.Layout(
        title='Normalised Comparison of MAEs for Decision Tree Variants',
        yaxis=dict(title='Mean Absolute Error (Normalised, Mean = 0)'),
        boxmode='group',
        # xaxis=dict(tickangle=90),
        width=400,
        height=400
    )

    go.Figure(data=_data, layout=_layout)
    return


@app.cell
def _(maes_deep, maes_prune, maes_shallow):
    _trace_shallow = go.Histogram(x=maes_shallow, name='Shallow', opacity=0.5, marker_color='blue')
    _trace_deep = go.Histogram(x=maes_deep, name='Deep', opacity=0.5, marker_color='green')
    _trace_prune = go.Histogram(x=maes_prune, name='Pruned', opacity=0.5, marker_color='red')

    _data = [_trace_shallow, _trace_deep, _trace_prune]

    _layout = go.Layout(
        barmode='overlay',
        title='Overlapping Histogram of MAE Distributions',
        xaxis=dict(title='Mean Absolute Error'),
        yaxis=dict(title='Count'),
        width=800
    )

    go.Figure(data=_data, layout=_layout)
    return


@app.cell
def _(fitpred_loop, fitpred_single):
    dt_split = DecisionTreeRegressor(max_depth=10, min_samples_split=0.2, random_state=42)
    fitpred_single(dt_split)
    maes_split, mapes_split, mses_split, rmses_split, r2s_split, time_split = fitpred_loop(dt_split)
    print_info("split")
    return


@app.cell
def _(fitpred_loop, fitpred_single):
    dt_leaf = DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, random_state=42)
    fitpred_single(dt_leaf)
    maes_leaf, mapes_leaf, mses_leaf, rmses_leaf, r2s_leaf, time_leaf = fitpred_loop(dt_leaf)
    print_info("leaf")
    return


@app.cell
def _(fitpred_loop, fitpred_single):
    dt_max = DecisionTreeRegressor(max_depth=10, max_leaf_nodes=100, random_state=42)
    fitpred_single(dt_max)
    maes_max, mapes_max, mses_max, rmses_max, r2s_max, time_max = fitpred_loop(dt_max)
    print_info("max")
    return (dt_max,)


@app.cell
def _(dt_max):
    _tree = dt_max.tree_
    _is_leaf = (_tree.children_left == -1) & (_tree.children_right == -1)
    _leaf_samples = _tree.n_node_samples[_is_leaf]
    _total_samples = _tree.n_node_samples[0]  # Root node has all samples
    _percent_per_leaf = _leaf_samples / _total_samples * 100
    _leaf_indices = np.where(_is_leaf)[0]
    for _idx, _s, _perc in zip(_leaf_indices, _leaf_samples, _percent_per_leaf):
        print(f"Leaf node {_idx}: {_s} samples ({_perc:.2f}%)")
    return


@app.cell
def _(X_train, y_train):
    search_space = {
        'max_depth': (1, 100),
        'min_samples_split': (2, 100),
        'min_samples_leaf': (1, 100),
        'max_features': (0.1, 1.0, 'uniform'),
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
        'ccp_alpha': (0.0, 1.0, 'uniform'),
    }

    opt = BayesSearchCV(
        DecisionTreeRegressor(random_state=42),
        search_spaces=search_space,
        n_iter=250,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_absolute_error',
        verbose=0,
        random_state=42
    )

    opt.fit(X_train, y_train)

    print("Best Parameters:", opt.best_params_)
    print("Best MAE:", -opt.best_score_)
    return


@app.cell
def _(X_train, y_train):
    # dt_bscv = opt.best_estimator_

    dt_bscv = DecisionTreeRegressor(ccp_alpha=0.0, criterion='squared_error', max_depth=100, max_features=0.9193546958301854, min_samples_leaf=15, min_samples_split=24)
    dt_bscv.fit(X_train, y_train)
    return (dt_bscv,)


@app.cell
def _(chosen, dt_bscv):
    dt_bscv.predict(chosen.drop(columns='MedHouseVal'))
    return


@app.cell
def _(dt_bscv, fitpred_loop):
    maes_bscv, mapes_bscv, mses_bscv, rmses_bscv, r2s_bscv, time_bscv = fitpred_loop(dt_bscv)
    return maes_bscv, mapes_bscv, mses_bscv, r2s_bscv, rmses_bscv


@app.cell
def _():
    print_info("bscv")
    return


@app.cell
def _(
    maes_bscv,
    maes_deep,
    maes_prune,
    maes_shallow,
    mapes_bscv,
    mapes_deep,
    mapes_prune,
    mapes_shallow,
    mses_bscv,
    mses_deep,
    mses_prune,
    mses_shallow,
    r2s_bscv,
    r2s_deep,
    r2s_prune,
    r2s_shallow,
    rmses_bscv,
    rmses_deep,
    rmses_prune,
    rmses_shallow,
):
    _trace_maes_shallow = go.Box(y=maes_shallow, name='MAEs, Shallow Tree', marker_color='blue', width=0.6)
    _trace_maes_deep = go.Box(y=maes_deep, name='MAEs, Deep Tree', marker_color='green', width=0.6)
    _trace_maes_prune = go.Box(y=maes_prune, name='MAEs, Prune Tree', marker_color='red', width=0.6)
    _trace_maes_bscv = go.Box(y=maes_bscv, name='MAEs, Bayes Tree', marker_color='black', width=0.6)
    _trace_mapes_shallow = go.Box(y=mapes_shallow, name='MAPEs, Shallow Tree', marker_color='blue', width=0.6)
    _trace_mapes_deep = go.Box(y=mapes_deep, name='MAPEs, Deep Tree', marker_color='green', width=0.6)
    _trace_mapes_prune = go.Box(y=mapes_prune, name='MAPEs, Prune Tree', marker_color='red', width=0.6)
    _trace_mapes_bscv = go.Box(y=mapes_bscv, name='MAPEs, Bayes Tree', marker_color='black', width=0.6)
    _trace_mses_shallow = go.Box(y=mses_shallow, name='MSEs, Shallow Tree', marker_color='blue', width=0.6)
    _trace_mses_deep = go.Box(y=mses_deep, name='MSEs, Deep Tree', marker_color='green', width=0.6)
    _trace_mses_prune = go.Box(y=mses_prune, name='MSEs, Prune Tree', marker_color='red', width=0.6)
    _trace_mses_bscv = go.Box(y=mses_bscv, name='MSEs, Bayes Tree', marker_color='black', width=0.6)
    _trace_rmses_shallow = go.Box(y=rmses_shallow, name='RMSEs, Shallow Tree', marker_color='blue', width=0.6)
    _trace_rmses_deep = go.Box(y=rmses_deep, name='RMSEs, Deep Tree', marker_color='green', width=0.6)
    _trace_rmses_prune = go.Box(y=rmses_prune, name='RMSEs, Prune Tree', marker_color='red', width=0.6)
    _trace_rmses_bscv = go.Box(y=rmses_bscv, name='RMSEs, Bayes Tree', marker_color='black', width=0.6)
    _trace_r2s_shallow = go.Box(y=r2s_shallow, name='R2s, Shallow Tree', marker_color='blue', width=0.6)
    _trace_r2s_deep = go.Box(y=r2s_deep, name='R2s, Deep Tree', marker_color='green', width=0.6)
    _trace_r2s_prune = go.Box(y=r2s_prune, name='R2s, Prune Tree', marker_color='red', width=0.6)
    _trace_r2s_bscv = go.Box(y=r2s_bscv, name='R2s, Bayes Tree', marker_color='black', width=0.6)

    _data = [
        _trace_maes_shallow, _trace_maes_deep, _trace_maes_prune, _trace_maes_bscv, 
        _trace_mapes_shallow, _trace_mapes_deep, _trace_mapes_prune, _trace_mapes_bscv, 
        _trace_mses_shallow, _trace_mses_deep, _trace_mses_prune, _trace_mses_bscv, 
        _trace_rmses_shallow, _trace_rmses_deep, _trace_rmses_prune, _trace_rmses_bscv, 
        _trace_r2s_shallow, _trace_r2s_deep, _trace_r2s_prune, _trace_r2s_bscv
    ]

    _layout = go.Layout(
        title='Comparison of Errors for Decision Tree Variants',
        yaxis=dict(title='Error'),
        boxmode='group',
        xaxis=dict(tickangle=90),
    )

    go.Figure(data=_data, layout=_layout)
    return


@app.cell
def _():
    scoring = ['neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error', 'r2']
    return (scoring,)


@app.cell
def _(X, dt_bscv, scoring, y):
    _cv = ShuffleSplit(n_splits=1000, test_size=0.2, random_state=42)
    sscv_results = cross_validate(dt_bscv, X, y, cv=_cv, scoring=scoring, return_train_score=False)
    sscv_results
    return


@app.cell
def _(X, dt_bscv, scoring, y):
    _cv = RepeatedKFold(n_splits=5, n_repeats=200, random_state=42)  # 5*200=1000 splits
    rkf_results = cross_validate(dt_bscv, X, y, cv=_cv, scoring=scoring, return_train_score=False)
    rkf_results
    return


if __name__ == "__main__":
    app.run()
