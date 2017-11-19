from functools import partial

import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.benchmark import benchmark_models


def impute_logistic(data, target, scale, categorical, scoring):
    from sklearn.preprocessing import StandardScaler
    # Prepare data
    from copy import deepcopy
    prediction_params = scale + categorical
    training_cond = data[target].notnull()
    y_train = data[training_cond][target]

    _data = deepcopy(data)
    _data[scale] = StandardScaler().fit_transform(_data[scale])
    _data = pd.get_dummies(_data, columns=categorical, drop_first=True)
    prediction_params = [param for param in _data.columns if
                         any(param.startswith(dummy) for dummy in prediction_params)]
    X_train = _data[training_cond][prediction_params]
    classifiers = [LogisticRegression( multi_class='multinomial', solver='lbfgs')]
    m = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    m.fit(X_train, y_train)

    X_test = _data[~training_cond][prediction_params]
    data.loc[~training_cond, target] = m.predict(X_test)
    return data


def impute_linear(data, target, scale, categorical, scoring):
    from sklearn.preprocessing import StandardScaler
    # Prepare data
    from copy import deepcopy
    prediction_params = scale + categorical
    training_cond = data[target].notnull()
    y_train = data[training_cond][target]
    _data = deepcopy(data)
    _data[scale] = StandardScaler().fit_transform(_data[scale])
    _data = pd.get_dummies(_data, columns=categorical, drop_first=True)
    prediction_params = [param for param in _data.columns if
                         any(param.startswith(dummy) for dummy in prediction_params)]
    X_train = _data[training_cond][prediction_params]
    benchmark = benchmark_models( X_train, y_train, scoring=scoring)
    m=benchmark[benchmark.idxmax('Mean')]
    m.fit(X_train, y_train)

    X_test = _data[~training_cond][prediction_params]
    data.loc[~training_cond, target] = m.predict(X_test)
    return data
