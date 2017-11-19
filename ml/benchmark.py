from contextlib import contextmanager
from itertools import chain
from operator import itemgetter
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, ParameterSampler

from estimators import classifiers, regressors, TunableEstimator


def benchmark_models(X, y, scoring, n_iter=5):
    """ Cookie cutter method, does a lot of things, is expensive."""
    print(f'Predicting {y.name} ', end='')
    plt.figure()
    if y.dtype != np.number or len(y.unique()) < 15:
        print(f'and it is a classification problem')
        sns.countplot(y)
        models = classifiers
    else:
        print(f'and it is a regression problem')
        sns.distplot(y)
        models = regressors
    results = pd.DataFrame(score_model(model, X, y, scoring, n_iter) for model in models)
    return results


def score_model(model: TunableEstimator, X: pd.DataFrame, y: pd.Series, scoring, n_iter=5) -> pd.Series:
    """ Evaluates different parametrizations for a model, including the default parametrization and returns the best"""
    defaults = {}
    p = chain([defaults], ParameterSampler(param_distributions=model.hyperparameters, n_iter=n_iter))
    print(f'Evaluating {model.__name__}', end='...')
    best = max([score_parametrization(model(**hyper_params), X, y, scoring) for hyper_params in p], key=itemgetter(0))
    print(f'... best is {best[0]}.')
    return pd.Series(best, index=['Mean', 'Std', 'Model'])


def score_parametrization(model, X_train: pd.DataFrame, y_train, scoring) -> Tuple[float, float, BaseEstimator]:
    """ Returns the performance of an estimator parametrization along with the tuned model"""
    with suppress_and_log():
        result = cross_val_score(model, X=X_train, y=y_train, cv=3, n_jobs=-1, scoring=scoring)
        print(f'{result.mean():4.2f}', end=' ')
        return result.mean(), result.std(), model
    return -1000, 0, model


@contextmanager
def suppress_and_log(exceptions=[]):
    try:
        yield
    except Exception:
        print(f'failed', end=' ')
