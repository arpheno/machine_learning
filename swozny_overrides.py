from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, \
    GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier, LogisticRegression, \
    HuberRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

LEARNING_RATE = [ 0.01, 0.1]
GENERIC_FLOAT = [0.0, 0.5, 0.66666666666666663, 0.33333333333333331, 0.16666666666666666, 0.83333333333333326, 1.0,
                 1.1666666666666665, 1.3333333333333333, 1.5]
GENERIC_IMPURITY = [0.0, 0.5, 0.66666666666666663, 0.33333333333333331, 0.16666666666666666, 0.83333333333333326, 1.0,
                    1.1666666666666665, 1.3333333333333333, 1.5, None]
LEAF_SAMPLES = [0.0, 1, 2, 0.16666666666666666, 100, 5, 0.33333333333333331, 0.5, 0.66666666666666663,
                0.83333333333333326, 10,
                1.1666666666666665, 1.5, 50, 20, 1.3333333333333333]
ALPHA = [1e-11, 2e-11, 3e-11, 4e-11, 4.9999999999999995e-11, 6e-11, 6.999999999999999e-11, 8e-11, 9e-11,
         9.999999999999999e-11, 1.1e-10, 1.2e-10, 1.3e-10, 1.3999999999999998e-10, 1.5e-10, 1.6e-10,
         1.6999999999999998e-10, 1.8e-10, 1.9e-10, 1.9999999999999998e-10, 2.1e-10, 2.2e-10, 2.2999999999999998e-10,
         2.4e-10, 2.4999999999999996e-10, 2.6e-10, 2.7e-10, 2.7999999999999996e-10, 2.9e-10, 1e-10]
TREE_DEPTH = [1, 2, 100, 5, 10, 50, 20, 3]
MAX_FEATURES = ['log2', 'sqrt', 'auto', None]
EPSILON = [1.05, 1.1500000000000001, 1.25, 1.35, 1.4500000000000002, 1.55]
ESTIMATORS = [10, 100]
C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
_manual_overrides = {AdaBoostClassifier: {'algorithm': ['SAMME.R', 'SAMME'],
                                          'learning_rate': LEARNING_RATE,
                                          'n_estimators': ESTIMATORS},
                     AdaBoostRegressor: {'algorithm': ['SAMME', 'SAMME.R'],
                                         'base_estimator__criterion': ['gini', 'entropy'],
                                         'base_estimator__splitter': ['best', 'random'],
                                         'learning_rate': LEARNING_RATE,
                                         'n_estimators': ESTIMATORS},
                     BaggingClassifier: {'n_estimators': ESTIMATORS},
                     DecisionTreeClassifier: {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random']},
                     ExtraTreeClassifier: {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random']},
                     ExtraTreesClassifier: {'criterion': ['gini', 'entropy'], },
                     GaussianProcessRegressor: {'alpha': ALPHA},
                     GradientBoostingClassifier: {
                         # 'criterion': ['mse', 'mae'],
                                              'learning_rate': LEARNING_RATE,
                                                  # 'loss': ['exponential', 'deviance'],
                                              'n_estimators': ESTIMATORS,
                                                  },
                     GradientBoostingRegressor: {'learning_rate': LEARNING_RATE,
                                                 'loss': ['ls', 'lad', 'huber', 'quantile'],
                                                 'n_estimators': ESTIMATORS},
                     HuberRegressor: {'epsilon': EPSILON},
                     KNeighborsClassifier: {'n_neighbors': [1, 2, 100, 5, 10, 50, 20], },
                     LogisticRegression: {'C': np.linspace(0.1, 2, num=10), },
                     PassiveAggressiveClassifier: {'C': np.linspace(0.1, 2, num=10)},
                     RandomForestClassifier: {'criterion': ['gini', 'entropy'], 'n_estimators': ESTIMATORS},
                     RandomForestRegressor: {'n_estimators': ESTIMATORS, },
                     RidgeClassifier: {'alpha': GENERIC_FLOAT},
                     SGDClassifier: {'loss': ['hinge', 'squared_hinge', 'epsilon_insensitive', 'squared_loss',
                                              'huber', 'modified_huber', 'perceptron', 'log',
                                              'squared_epsilon_insensitive']},
                     SVC: {'C': np.linspace(0.1, 2, num=10),
                           'kernel': ['linear', 'sigmoid', 'poly', 'rbf']},
                     SVR: {'C': C, 'kernel': ['rbf', 'linear']}}
manual_overrides = defaultdict(lambda: defaultdict(list))
manual_overrides.update(_manual_overrides)

import itertools
if __name__ == '__main__':
    for key, value in manual_overrides.items():
        print(f'{key.__name__} has {len(list(itertools.product(*itertools.chain(value.values()))))} combinations')
