from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, learning_curve

from swozny_overrides import manual_overrides


def benchmark_models(model_classes: List[type], X_train, y_train, scoring='neg_mean_absolute_error', verbose=False):
    results = pd.DataFrame()
    max_len = max(len(f'Evaluating {model_cls.__name__}...') for model_cls in model_classes)
    for model_cls in model_classes:
        try:
            print(f'Evaluating {model_cls.__name__:{max_len}s}', end='...')
            result = cross_val_score(model_cls(), X=X_train, y=y_train, cv=4, n_jobs=4, scoring=scoring)
            results = results.append(pd.Series(dict(Mean=result.mean(), Std=result.std(), Algorithm=model_cls)),
                                     ignore_index=True, )
            print(f' Score {result.mean():4.2f}')
        except Exception as e:
            if verbose == True:
                print(f' failed. with{e}')
            else:
                print(f' failed')
    return results


def plot_benchmark(cv_res: pd.DataFrame, xlabel="Mean error"):
    g = sns.barplot("Mean", "Algorithm", data=cv_res, palette="Set3", orient="h", **{'xerr': cv_res.Std})
    g.set_xlabel(xlabel)
    g = g.set_title("Cross validation scores")


def tune_params(model_classes: List[type], X_train, y_train, scoring='neg_mean_absolute_error'):
    model_classes = dict((cls(), manual_overrides[cls]) for cls in model_classes)
    best_models = []
    for model, hypers in model_classes.items():
        print(f'Tuning {model.__class__.__name__} with {hypers}...')
        clf = GridSearchCV(model, hypers, scoring=scoring, n_jobs=4, cv=4)
        clf.fit(X_train, y_train)
        best_models.append(clf.best_estimator_)
        print(clf.best_score_)  # ,clf.best_params_)
    return best_models


def linear_ensemble(regressors, X_train, y_train, scoring='neg_mean_absolute_error'):
    ensemble_results = pd.concat(
        [pd.Series(regressor.predict(X_train), name=regressor.__class__.__name__) for regressor in regressors], axis=1)
    g = sns.heatmap(ensemble_results.corr(), annot=True)
    g = LinearRegression()
    print(cross_val_score(g, X=ensemble_results, y=y_train, cv=5, n_jobs=4, scoring=scoring).mean())
    g.fit(X_train, y_train)
    return g


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_absolute_error')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def pairplot(data, target):
    plt.figure()
    sns.pairplot(data, hue=target, vars=data.select_dtypes(include=[np.number]))
