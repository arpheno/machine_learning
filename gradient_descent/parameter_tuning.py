""" Tunes hyperparameters of estimators via the fmin_slsqp optimizer. Needs some more work."""
from functools import partial
from typing import Any, Dict

from numpy import mean
from scipy.optimize import fmin_slsqp
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


def wrap(f):
    def inner(**kwargs):
        return f(**kwargs)


def tune_params_gradient_descent(estimator_cls, x0: Dict[str, Any], X, y, scoring: str):
    initial = list(x0.values())

    def f(x):
        params = dict(zip(x0, x))
        estimator = estimator_cls(**params)
        score =-cross_val_score(estimator, X=X, y=y, scoring=scoring,cv=10).mean()
        print(f'evaluating {x} at {score}')
        return score

    best_params = fmin_slsqp(f, initial,epsilon=mean(initial)/100,iter=1000,iprint=5,acc=10e-10)
    return {param: value for param, value in zip(x0, best_params)}


if __name__ == '__main__':
    iris = datasets.load_iris()
    # Take the first two features. We could avoid this by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target
    best_params = tune_params_gradient_descent(partial(SVC,probability=True), dict(C=10), X, y, 'neg_log_loss')
    print(best_params)
