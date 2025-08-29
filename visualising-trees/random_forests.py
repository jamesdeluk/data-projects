import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import time
    import timeit
    import pandas as pd
    import numpy as np 
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor, plot_tree
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score, make_scorer
    from skopt import BayesSearchCV
    import plotly.graph_objs as go
    import matplotlib.pyplot as plt


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
def _(california_housing_df, target):
    features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    X = california_housing_df[features]
    y = california_housing_df[target]
    return X, features, y


@app.cell
def _(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell
def _():
    rf = RandomForestRegressor(random_state=42)
    return (rf,)


@app.cell
def _(X_test, X_train, chosen, target, y_test, y_train):
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
    return (fitpred_single,)


@app.cell
def _(X_test, X_train, chosen, target, y_test, y_train):
    def fitpred_list(dt_list, timing_count=1):
        all_results = []
        for dt in dt_list:
            results = {}
            fit_time = timeit.timeit(lambda: dt.fit(X_train, y_train), number=timing_count)
            print(f"Time to fit:\t\t{fit_time/timing_count}")
            results['fit_time'] = fit_time/timing_count

            pred_time = timeit.timeit(lambda: dt.predict(X_test), number=timing_count)
            print(f"Time to predict:\t{pred_time/timing_count}")
            results['fit_pred'] = pred_time/timing_count
            dt_predict = dt.predict(X_test)

            print()

            print(f"MAE:\t{mean_absolute_error(y_test, dt_predict)}")
            print(f"MAPE:\t{mean_absolute_percentage_error(y_test, dt_predict)}")
            print(f"MSE:\t{mean_squared_error(y_test, dt_predict)}")
            print(f"RMSE:\t{root_mean_squared_error(y_test, dt_predict)}")
            print(f"R2:\t\t{r2_score(y_test, dt_predict)}")
            results['MAE'] = mean_absolute_error(y_test, dt_predict)
            results['MAPE'] = mean_absolute_percentage_error(y_test, dt_predict)
            results['MSE'] = mean_squared_error(y_test, dt_predict)
            results['RMSE'] = root_mean_squared_error(y_test, dt_predict)
            results['R2'] = r2_score(y_test, dt_predict)

            print()

            print(f"Chosen prediction:\t{float(dt.predict(chosen.drop(columns=target))[0])}")
            results['Chosen prediction'] = float(dt.predict(chosen.drop(columns=target))[0])
            print(f"Chosen actual:\t\t{chosen[target].iloc[0]}")

        return all_results
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
    return


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


@app.function
def print_info_cv(cv):
    print(f"MAE mean:\t{np.mean(cv['test_neg_mean_absolute_error']):.3f}, std: {np.std(cv['test_neg_mean_absolute_error']):.3f}")
    print(f"MAPE mean:\t{np.mean(cv['test_neg_mean_absolute_percentage_error']):.3f}, std: {np.std(cv['test_neg_mean_absolute_percentage_error']):.3f}")
    print(f"MSE mean:\t{np.mean(cv['test_neg_mean_squared_error']):.3f}, std: {np.std(cv['test_neg_mean_squared_error']):.3f}")
    print(f"RMSE mean:\t{np.mean(cv['test_root_mean_squared_error']):.3f}, std: {np.std(cv['test_root_mean_squared_error']):.3f}")
    print(f"R² mean:\t{np.mean(cv['test_r2']):.3f}, std: {np.std(cv['test_r2']):.3f}")
    print()
    print(f"Time:\t\t{cv['fit_time'].sum():.1f}s")


@app.cell
def _(fitpred_single, rf):
    fitpred_single(rf)
    return


@app.cell
def _(rf):
    for i, tree in enumerate(rf.estimators_[:1]):
        print(tree.get_depth())
        print(tree.get_n_leaves())
        print(tree.tree_.node_count)
        plt.figure(figsize=(20, 10))
        plot_tree(tree, filled=True, fontsize=10)
        plt.title(f"Decision Tree {i+1}")
        plt.show()
    return


@app.cell
def _(chosen, features, rf, target):
    tree_predictions = [tree.predict(chosen[features].values) for tree in rf.estimators_]  # Indiviudal trees don't store feature names
    mean_prediction = np.mean(tree_predictions)

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(tree_predictions)), tree_predictions, alpha=0.7, label='Individual Predictions')
    plt.axhline(mean_prediction, color='red', linestyle='--', label='Mean Prediction')
    plt.axhline(chosen[target].iloc[0], color='blue', linestyle='--', label='Actual')
    plt.title('Individual Tree Predictions for the Chosen Row')
    plt.xlabel('Tree Index')
    plt.ylabel('Prediction')
    plt.legend()
    plt.show()
    return


@app.cell
def _(features, rf):
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xticks(rotation=45)
    plt.title("Feature Importances")
    plt.show()
    return


@app.cell
def _(X, rf, y):
    tree_errors = []

    for _i, _tree in enumerate(rf.estimators_):
        _y_pred = _tree.predict(X)
        _mse = mean_squared_error(y, _y_pred)
        tree_errors.append(_mse)

    best_tree = np.argmin(tree_errors)
    worst_tree = np.argmax(tree_errors)

    print(f"Best Tree: {best_tree} with MSE {tree_errors[best_tree]:.4f}")
    print(f"Worst Tree: {worst_tree} with MSE {tree_errors[worst_tree]:.4f}")

    plt.figure(figsize=(10, 5))
    plt.scatter(range(1, len(tree_errors) + 1), tree_errors, alpha=0.7)
    plt.xlabel("Tree Index")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Mean Squared Error of Each Tree in the Random Forest")
    plt.axhline(np.mean(tree_errors), color="red", linestyle="--", label="Mean Error")
    plt.legend()
    plt.show()
    return


@app.cell
def _(rf):
    print(rf.estimators_[32].get_depth())
    print(rf.estimators_[32].get_n_leaves())
    print(rf.estimators_[32].tree_.node_count)
    return


@app.cell
def _(rf):
    print(rf.estimators_[74].get_depth())
    print(rf.estimators_[74].get_n_leaves())
    print(rf.estimators_[74].tree_.node_count)
    return


@app.cell
def _(features, rf):
    imp_32 = pd.DataFrame({
        'Feature': features,
        'Importance': rf.estimators_[32].feature_importances_
    }).sort_values(by='Feature')

    imp_74 = pd.DataFrame({
        'Feature': features,
        'Importance': rf.estimators_[74].feature_importances_
    }).sort_values(by='Feature')

    imp_all = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values(by='Feature')

    # assert list(imp_32['Feature']) == list(imp_74['Feature'])

    combined_imp = pd.DataFrame({
        'Feature': imp_all['Feature'],
        'Overall': imp_all['Importance'],
        'Tree 32': imp_32['Importance'],
        'Tree 74': imp_74['Importance']
    }).sort_values(by='Overall', ascending=False)

    x = np.arange(len(combined_imp['Feature']))  # Positions for bars
    width = 0.2  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, combined_imp['Overall'], width, label='Overall')  # Shift left
    bars2 = ax.bar(x, combined_imp['Tree 32'], width, label='Tree 32')          # Centre
    bars3 = ax.bar(x + width, combined_imp['Tree 74'], width, label='Tree 74')  # Shift right

    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importances')
    ax.set_xticks(x)
    ax.set_xticklabels(combined_imp['Feature'], rotation=45, ha='right')
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    rf_shallow = RandomForestRegressor(max_depth=3, random_state=42)
    rf_deep = RandomForestRegressor(max_depth=None, random_state=42)
    rf_prune = RandomForestRegressor(max_depth=None, ccp_alpha=0.005, random_state=42)
    rf_split = RandomForestRegressor(max_depth=10, min_samples_split=10, random_state=42)
    rf_leaf = RandomForestRegressor(max_depth=10, min_samples_leaf=10, random_state=42)
    rf_max = RandomForestRegressor(max_depth=10, max_leaf_nodes=100, random_state=42)
    return rf_deep, rf_leaf, rf_max, rf_prune, rf_shallow, rf_split


@app.cell
def _(
    fitpred_single,
    rf_deep,
    rf_leaf,
    rf_max,
    rf_prune,
    rf_shallow,
    rf_split,
):
    print('Shallow')
    fitpred_single(rf_shallow)
    print("---")
    print('Deep')
    fitpred_single(rf_deep)
    print("---")
    print('ccp_alpha')
    fitpred_single(rf_prune)
    print("---")
    print('min_samples_split')
    fitpred_single(rf_split)
    print("---")
    print('min_samples_leaf')
    fitpred_single(rf_leaf)
    print("---")
    print('max_leaf_nodes')
    fitpred_single(rf_max)
    return


@app.cell
def _(fitpred_single):
    rf_parallel = RandomForestRegressor(n_jobs=-1, max_depth=None, random_state=42)
    fitpred_single(rf_parallel)
    return (rf_parallel,)


@app.cell
def _(fitpred_single):
    rf_thou = RandomForestRegressor(n_estimators=1000, max_depth=None, n_jobs=-1, random_state=42)
    fitpred_single(rf_thou)
    return (rf_thou,)


@app.cell
def _(X, rf_parallel, y):
    rf_cv_rkf_100 = cross_validate(rf_parallel, X, y, cv=RepeatedKFold(n_splits=5, n_repeats=20, random_state=42), n_jobs=-1, scoring={
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
        'neg_mean_absolute_percentage_error': 'neg_mean_absolute_percentage_error',
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'root_mean_squared_error': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False),
        'r2': 'r2'
    })
    rf_cv_rkf_100
    return (rf_cv_rkf_100,)


@app.cell
def _(rf_cv_rkf_100):
    print_info_cv(rf_cv_rkf_100)
    return


@app.cell
def _(X, rf_thou, y):
    rf_cv_rkf_1000 = cross_validate(rf_thou, X, y, cv=RepeatedKFold(n_splits=5, n_repeats=20, random_state=42), n_jobs=-1, scoring={
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
        'neg_mean_absolute_percentage_error': 'neg_mean_absolute_percentage_error',
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'root_mean_squared_error': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False),
        'r2': 'r2'
    })
    rf_cv_rkf_1000
    return (rf_cv_rkf_1000,)


@app.cell
def _(rf_cv_rkf_1000):
    print_info_cv(rf_cv_rkf_1000)
    return


@app.cell
def _(X_train, y_train):
    search_spaces = {
        'n_estimators': (50, 500),
        'max_depth': (1, 100),
        'min_samples_split': (2, 100),
        'min_samples_leaf': (1, 100),
        'max_features': (0.1, 1.0, 'uniform'),
        'criterion': ['squared_error'],
        'bootstrap': [True, False],
        'ccp_alpha': (0.0, 1.0, 'uniform'),
    }

    opt = BayesSearchCV(
        RandomForestRegressor(n_jobs=-1, random_state=42),
        search_spaces=search_spaces,
        n_iter=200,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_absolute_error',
        verbose=1,
        random_state=42
    )

    opt.fit(X_train, y_train)

    print("Best Parameters:", opt.best_params_)
    print("Best MAE:", -opt.best_score_)
    return (opt,)


@app.cell
def _(fitpred_single, opt):
    fitpred_single(opt.best_estimator_)
    return


@app.cell
def _(X, opt, y):
    rf_cv_rkf_bscv = cross_validate(opt.best_estimator_, X, y, cv=RepeatedKFold(n_splits=5, n_repeats=20, random_state=42), n_jobs=-1, scoring={
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
        'neg_mean_absolute_percentage_error': 'neg_mean_absolute_percentage_error',
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'root_mean_squared_error': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False),
        'r2': 'r2'
    })
    rf_cv_rkf_bscv
    return (rf_cv_rkf_bscv,)


@app.cell
def _(rf_cv_rkf_bscv):
    print_info_cv(rf_cv_rkf_bscv)
    return


@app.cell
def _():
    dt_bscv = DecisionTreeRegressor(ccp_alpha=0.0, criterion='squared_error', max_depth=100, max_features=0.9193546958301854, min_samples_leaf=15, min_samples_split=24)
    return (dt_bscv,)


@app.cell
def _(X, dt_bscv, y):
    dt_cv_rkf = cross_validate(dt_bscv, X, y, cv=RepeatedKFold(n_splits=5, n_repeats=20, random_state=42), scoring={
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
        'neg_mean_absolute_percentage_error': 'neg_mean_absolute_percentage_error',
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'root_mean_squared_error': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False),
        'r2': 'r2'
    }, return_train_score=False)
    dt_cv_rkf
    return (dt_cv_rkf,)


@app.cell
def _(dt_cv_rkf, rf_cv_rkf_100, rf_cv_rkf_bscv):
    _data = [
        go.Box(y=abs(dt_cv_rkf['test_neg_mean_absolute_error']), name='MAEs, Bayes DT', marker_color='black', width=0.6),
        go.Box(y=abs(rf_cv_rkf_100['test_neg_mean_absolute_error']), name='MAEs, RF', marker_color='pink', width=0.6),
        # go.Box(y=rf_cv_rkf_1000['test_neg_mean_absolute_error'], name='MAEs, RF 1000', marker_color='red', width=0.6),
        go.Box(y=abs(rf_cv_rkf_bscv['test_neg_mean_absolute_error']), name='MAEs, Bayes RF', marker_color='green', width=0.6),
        go.Box(y=abs(dt_cv_rkf['test_neg_mean_absolute_percentage_error']), name='MAPEs, Bayes DT', marker_color='black', width=0.6),
        go.Box(y=abs(rf_cv_rkf_100['test_neg_mean_absolute_percentage_error']), name='MAPEs, RF', marker_color='pink', width=0.6),
        # go.Box(y=rf_cv_rkf_1000['test_neg_mean_absolute_percentage_error'], name='MAPEs, RF 1000', marker_color='red', width=0.6),
        go.Box(y=abs(rf_cv_rkf_bscv['test_neg_mean_absolute_percentage_error']), name='MAPEs, Bayes RF', marker_color='green', width=0.6),
        go.Box(y=abs(dt_cv_rkf['test_neg_mean_squared_error']), name='MSEs, Bayes DT', marker_color='black', width=0.6),
        go.Box(y=abs(rf_cv_rkf_100['test_neg_mean_squared_error']), name='MSEs, RF', marker_color='pink', width=0.6),
        # go.Box(y=rf_cv_rkf_1000['test_neg_mean_squared_error'], name='MSEs, RF 1000', marker_color='red', width=0.6),
        go.Box(y=abs(rf_cv_rkf_bscv['test_neg_mean_squared_error']), name='MSEs, Bayes RF', marker_color='green', width=0.6),
        go.Box(y=abs(dt_cv_rkf['test_root_mean_squared_error']), name='RMSEs, Bayes DT', marker_color='black', width=0.6),
        go.Box(y=abs(rf_cv_rkf_100['test_root_mean_squared_error']), name='RMSEs, RF', marker_color='pink', width=0.6),
        # go.Box(y=rf_cv_rkf_1000['test_root_mean_squared_error'], name='RMSEs, RF 1000', marker_color='red', width=0.6),
        go.Box(y=abs(rf_cv_rkf_bscv['test_root_mean_squared_error']), name='RMSEs, Bayes RF', marker_color='green', width=0.6),
        go.Box(y=dt_cv_rkf['test_r2'], name='R2s, Bayes DT', marker_color='black', width=0.6),
        go.Box(y=rf_cv_rkf_100['test_r2'], name='R2s, RF', marker_color='pink', width=0.6),
        # go.Box(y=rf_cv_rkf_1000['test_r2'], name='R2s, RF 1000', marker_color='red', width=0.6),
        go.Box(y=rf_cv_rkf_bscv['test_r2'], name='R2s, Bayes RF', marker_color='green', width=0.6)
    ]

    _layout = go.Layout(
        title='Comparison of Errors for Tree Variants',
        yaxis=dict(title='Error'),
        boxmode='group',
        xaxis=dict(tickangle=90),
    )

    go.Figure(data=_data, layout=_layout)
    return


if __name__ == "__main__":
    app.run()
