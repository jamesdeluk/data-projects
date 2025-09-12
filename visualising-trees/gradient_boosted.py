import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import time
    import timeit
    import pandas as pd
    import numpy as np 
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate
    from sklearn.tree import DecisionTreeRegressor, plot_tree
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score, make_scorer
    from skopt import BayesSearchCV
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objs as go


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
def _(X_test, X_train, chosen, target, y_test, y_train):
    def fitpred_single(dt, timing_count=10):
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
    def fitpred_multiple(models, timing_count=10):
        results = []

        for name, model in models.items():
            print(name)
            fit_time = timeit.timeit(lambda: model.fit(X_train, y_train), number=timing_count) / timing_count

            pred_time = timeit.timeit(lambda: model.predict(X_test), number=timing_count) / timing_count
            predictions = model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            mape = mean_absolute_percentage_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)

            chosen_pred = float(model.predict(chosen.drop(columns=target))[0])

            results.append({
                "Model": name,
                "Fit Time (s)": fit_time,
                "Predict Time (s)": pred_time,
                "MAE": mae,
                "MAPE": mape,
                "MSE": mse,
                "RMSE": rmse,
                "R²": r2,
                "Chosen Prediction": chosen_pred,
                "Chosen Error": abs(chosen_pred - chosen[target].iloc[0])
            })

        results_df = round(pd.DataFrame(results),3)
        return results_df
    return (fitpred_multiple,)


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
def _(fitpred_multiple):
    gb = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, random_state=42)

    gb_deep = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=None, random_state=42)
    gb_ten = GradientBoostingRegressor(max_depth=10, random_state=42)
    gb_split = GradientBoostingRegressor(max_depth=10, min_samples_split=10, random_state=42)
    gb_leaf = GradientBoostingRegressor(max_depth=10, min_samples_leaf=10, random_state=42)
    gb_max = GradientBoostingRegressor(max_depth=10, max_leaf_nodes=100, random_state=42)

    gb_fastlearn = GradientBoostingRegressor(learning_rate=0.5, n_estimators=100, random_state=42)
    gb_slowlearn = GradientBoostingRegressor(learning_rate=0.01, n_estimators=100, random_state=42)

    models_originals_dict = {
        'Deep': gb_deep,
        '10': gb_ten,
        'Leaf': gb_leaf,
        'Split': gb_split,
        'Max': gb_max,
    }

    models_new_dict = {
        'Default': gb,
        'Fast': gb_fastlearn,
        'Slow': gb_slowlearn,
    }

    fitpred_multiple(models_originals_dict, timing_count=1)
    fitpred_multiple(models_new_dict, timing_count=1)
    return (gb,)


@app.cell
def _(fitpred_single, gb):
    fitpred_single(gb)
    return


@app.cell
def _(X, gb):
    plt.figure(figsize=(12,4))
    plot_tree(gb.estimators_[0, 0], feature_names=X.columns, label='all', filled=True, impurity=True, proportion=True, rounded=True)
    plt.show()
    return


@app.cell
def _(chosen, features, gb):
    _preds = []
    for _i, _pred in enumerate(gb.staged_predict(chosen[features])):
        _preds.append((_i,_pred[0]))
    _preds = pd.DataFrame(_preds)
    _preds.columns = ['Iterations','Prediction']
    px.scatter(_preds, x='Iterations', y='Prediction')
    return


@app.cell
def _(chosen, features, gb, target):
    _errs = []
    for _i, _pred in enumerate(gb.staged_predict(chosen[features])):
        _errs.append((_i,_pred[0]-chosen[target][0]))
    _errs = pd.DataFrame(_errs)
    _errs.columns = ['Iterations','Error']
    px.scatter(_errs, x='Iterations', y='Error')
    return


@app.cell
def _(X_test, gb):
    _preds = []
    for _i, _y_pred in enumerate(gb.staged_predict(X_test)):
        for _j, _val in enumerate(_y_pred):
            _preds.append((_i, _j, _val))

    _preds = pd.DataFrame(_preds, columns=["Iteration", "Sample", "Prediction"])

    px.line(
        _preds, 
        x="Iteration", 
        y="Prediction", 
        color="Sample", 
        line_group="Sample", 
        hover_data=["Sample"]
    )
    return


@app.cell
def _(X_test, gb, y_test):
    errors = []
    for _i, _y_pred in enumerate(gb.staged_predict(X_test)):
        for _j, _val in enumerate(_y_pred):
            _err = _val - y_test.iloc[_j]
            errors.append((_i, _j, _err))

    errors = pd.DataFrame(errors, columns=["Iteration", "Sample", "Error"])

    px.line(
        errors,
        x="Iteration",
        y="Error",
        color="Sample",
        line_group="Sample",
        hover_data=["Sample"]
    )
    return (errors,)


@app.cell
def _(errors):
    px.box(errors[errors['Iteration'] == 99],  y='Error', labels={'y': 'Error'})
    return


@app.cell
def _(fitpred_multiple):
    gb_thou = GradientBoostingRegressor(learning_rate=0.1, n_estimators=1000, random_state=42)
    gb_thou_es = GradientBoostingRegressor(learning_rate=0.1, n_estimators=1000, n_iter_no_change=5, validation_fraction=0.1, tol=0.005, random_state=42)

    models_es_dict = {
        'Default': gb_thou,
        'ES': gb_thou_es,
    }

    fitpred_multiple(models_es_dict, timing_count=1)
    return (models_es_dict,)


@app.cell
def _(chosen, features, models_es_dict, target):
    _preds = []

    for _name, _model in models_es_dict.items():
        for _i, _pred in enumerate(_model.staged_predict(chosen[features])):
            _preds.append((_i, _pred[0], _name))

    _preds = pd.DataFrame(_preds, columns=["Iteration", "Prediction", "Model"])

    _fig = px.scatter(
        _preds,
        x="Iteration",
        y="Prediction",
        color="Model",
    )
    _fig.add_hline(y=chosen[target].iloc[0], line_dash="dash", line_color="black", 
                  annotation_text="True value", annotation_position="top left")
    _fig
    return


@app.cell
def _(fitpred_multiple):
    _iters = 500

    gb_more = GradientBoostingRegressor(learning_rate=0.1, n_estimators=_iters, random_state=42)
    gb_fastlearn_more = GradientBoostingRegressor(learning_rate=0.5, n_estimators=_iters, random_state=42)
    gb_slowlearn_more = GradientBoostingRegressor(learning_rate=0.01, n_estimators=_iters, random_state=42)

    models_more_dict = {
        'DefaultMore': gb_more,
        'FastMore': gb_fastlearn_more,
        'SlowMore': gb_slowlearn_more,
    }

    fitpred_multiple(models_more_dict, timing_count=1)
    return (models_more_dict,)


@app.cell
def _(chosen, features, models_more_dict, target):
    _errs = []

    for _name, _model in models_more_dict.items():
        for _i, _pred in enumerate(_model.staged_predict(chosen[features])):
            _errs.append((_i, _pred[0]-chosen[target][0], _name))

    _errs = pd.DataFrame(_errs, columns=["Iteration", "Errors", "Model"])

    _fig = px.scatter(
        _errs,
        x="Iteration",
        y="Errors",
        color="Model",
    )
    _fig.add_hline(y=0, line_dash="dash", line_color="black")
    _fig
    return


@app.cell
def _():
    # search_spaces = {
    #     'n_estimators': (50, 1000),
    #     'learning_rate': (0.01, 0.5),
    #     'max_depth': (1, 100),
    #     'min_samples_split': (2, 100),
    #     'min_samples_leaf': (1, 100),
    #     'max_leaf_nodes': (2, 20000),
    #     'max_features': (0.1, 1.0, 'uniform'),
    # }

    # opt = BayesSearchCV(
    #     GradientBoostingRegressor(random_state=42),
    #     search_spaces=search_spaces,
    #     n_iter=100,
    #     cv=5,
    #     n_jobs=-1,
    #     scoring='neg_mean_absolute_error',
    #     verbose=1,
    #     random_state=42
    # )

    # opt.fit(X_train, y_train)

    # print("Best Parameters:", opt.best_params_)
    # print("Best MAE:", -opt.best_score_)
    return


@app.cell
def _():
    gb_bscv = GradientBoostingRegressor(learning_rate=0.04345459461297153, max_depth=13, max_features=0.4993693929975871, max_leaf_nodes=20000, min_samples_leaf=1, min_samples_split=83, n_estimators=325)
    return (gb_bscv,)


@app.cell
def _(X, gb_bscv, y):
    gb_cv_rkf_100 = cross_validate(gb_bscv, X, y, cv=RepeatedKFold(n_splits=5, n_repeats=20, random_state=42), n_jobs=-1, scoring={
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
        'neg_mean_absolute_percentage_error': 'neg_mean_absolute_percentage_error',
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'root_mean_squared_error': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False),
        'r2': 'r2'
    })
    return (gb_cv_rkf_100,)


@app.cell
def _(gb_cv_rkf_100):
    print_info_cv(gb_cv_rkf_100)
    return


@app.cell
def _(X, y):
    dt = DecisionTreeRegressor(ccp_alpha=0.0, criterion='squared_error', max_depth=100, max_features=0.9193546958301854, min_samples_leaf=15, min_samples_split=24)

    dt_cv = cross_validate(dt, X, y, cv=RepeatedKFold(n_splits=5, n_repeats=20, random_state=42), scoring={
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
        'neg_mean_absolute_percentage_error': 'neg_mean_absolute_percentage_error',
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'root_mean_squared_error': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False),
        'r2': 'r2'
    })
    return (dt_cv,)


@app.cell
def _(X, y):
    rf = RandomForestRegressor(bootstrap=False, ccp_alpha=0.0, criterion='squared_error', max_depth=39, max_features=0.4863711682589259, max_leaf_nodes=20000, min_samples_leaf=1, min_samples_split=2, n_estimators=380)

    rf_cv = cross_validate(rf, X, y, cv=RepeatedKFold(n_splits=5, n_repeats=20, random_state=42), n_jobs=-1, scoring={
        'neg_mean_absolute_error': 'neg_mean_absolute_error',
        'neg_mean_absolute_percentage_error': 'neg_mean_absolute_percentage_error',
        'neg_mean_squared_error': 'neg_mean_squared_error',
        'root_mean_squared_error': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False),
        'r2': 'r2'
    })
    return (rf_cv,)


@app.cell
def _(dt_cv, gb_cv_rkf_100, rf_cv):
    _data = [
        go.Box(y=abs(dt_cv['test_neg_mean_absolute_error']), name='MAEs, DT', marker_color='black', width=0.6),
        go.Box(y=abs(rf_cv['test_neg_mean_absolute_error']), name='MAEs, RF', marker_color='pink', width=0.6),
        go.Box(y=abs(gb_cv_rkf_100['test_neg_mean_absolute_error']), name='MAEs, GBT', marker_color='green', width=0.6),
        go.Box(y=abs(dt_cv['test_neg_mean_absolute_percentage_error']), name='MAPEs, DT', marker_color='black', width=0.6),
        go.Box(y=abs(rf_cv['test_neg_mean_absolute_percentage_error']), name='MAPEs, RF', marker_color='pink', width=0.6),
        go.Box(y=abs(gb_cv_rkf_100['test_neg_mean_absolute_percentage_error']), name='MAPEs, GBT', marker_color='green', width=0.6),
        go.Box(y=abs(dt_cv['test_neg_mean_squared_error']), name='MSEs, DT', marker_color='black', width=0.6),
        go.Box(y=abs(rf_cv['test_neg_mean_squared_error']), name='MSEs, RF', marker_color='pink', width=0.6),
        go.Box(y=abs(gb_cv_rkf_100['test_neg_mean_squared_error']), name='MSEs, GBT', marker_color='green', width=0.6),
        go.Box(y=abs(dt_cv['test_root_mean_squared_error']), name='RMSEs, DT', marker_color='black', width=0.6),
        go.Box(y=abs(rf_cv['test_root_mean_squared_error']), name='RMSEs, RF', marker_color='pink', width=0.6),
        go.Box(y=abs(gb_cv_rkf_100['test_root_mean_squared_error']), name='RMSEs, GBT', marker_color='green', width=0.6),
        go.Box(y=dt_cv['test_r2'], name='R²s, DT', marker_color='black', width=0.6),
        go.Box(y=rf_cv['test_r2'], name='R²s, RF', marker_color='pink', width=0.6),
        go.Box(y=gb_cv_rkf_100['test_r2'], name='R²s, GBT', marker_color='green', width=0.6),
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
