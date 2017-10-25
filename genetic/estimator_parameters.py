from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor, GradientBoostingClassifier, \
    RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, \
    BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import HuberRegressor, SGDClassifier, RidgeClassifier, LogisticRegression, \
    PassiveAggressiveClassifier, LinearRegression, PassiveAggressiveRegressor, TheilSenRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from genetic.evolution.parameters import FloatParam, CategoricalParam, IntParam

params = {
    # Regressors
    AdaBoostRegressor: dict(
        algorithm=(CategoricalParam, ['SAMME', 'SAMME.R']),
        learning_rate=(FloatParam, 0.1), n_estimators=(IntParam, 1)
    ),
    BaggingRegressor: dict(
        n_estimators=(IntParam, 1)
    ),
    GaussianProcessRegressor: dict(
        alpha=(FloatParam, 1e-10)
    ),
    GradientBoostingRegressor:
        dict(
            criterion=(CategoricalParam, ['mse', 'mae']),
            learning_rate=(FloatParam, 0.1),
            loss=(CategoricalParam, ['exponential', 'deviance']),
            max_features=(CategoricalParam, ['log2', 'sqrt', 'auto', None]),
            n_estimators=(IntParam, 1)
        ),
    HuberRegressor: dict(
        epsilon=(FloatParam, 1.35),
        alpha=(FloatParam, 0.001)
    ),
    LinearRegression: dict(),
    PassiveAggressiveRegressor: dict(
        C=(FloatParam, 1)
    ),
    RandomForestRegressor: dict(
        criterion=(CategoricalParam, ['mse', 'mae']),
        n_estimators=(IntParam, 1),
        max_features=(CategoricalParam, ['log2', 'sqrt', 'auto', None]),
    ),
    SVR: dict(
        C=(FloatParam, 1),
        kernel=(CategoricalParam, ['rbf', 'linear']),
    ),
    TheilSenRegressor: dict(),
    # Classifiers
    AdaBoostClassifier: dict(
        algorithm=(CategoricalParam, ['SAMME', 'SAMME.R']),
        learning_rate=(FloatParam, 0.1), n_estimators=(IntParam, 1)
    ),
    BaggingClassifier: dict(
        n_estimators=(IntParam, 1)
    ),
    DecisionTreeClassifier: dict(
        criterion=(CategoricalParam, ['gini', 'entropy']),
        max_features=(CategoricalParam, ['log2', 'sqrt', 'auto', None]),
        splitter=(CategoricalParam, ['best', 'random'])
    ),
    ExtraTreeClassifier: dict(
        criterion=(CategoricalParam, ['gini', 'entropy']),
        max_features=(CategoricalParam, ['log2', 'sqrt', 'auto', None]),
        splitter=(CategoricalParam, ['best', 'random'])
    ),
    ExtraTreesClassifier: dict(
        criterion=(CategoricalParam, ['gini', 'entropy']),
        max_features=(CategoricalParam, ['log2', 'sqrt', 'auto', None]),
        n_estimators=(IntParam, 1)
    ),
    GradientBoostingClassifier:
        dict(
            criterion=(CategoricalParam, ['mse', 'mae']),
            learning_rate=(FloatParam, 0.1),
            loss=(CategoricalParam, ['exponential', 'deviance']),
            max_features=(CategoricalParam, ['log2', 'sqrt', 'auto', None]),
            n_estimators=(IntParam, 1)
        ),
    KNeighborsClassifier: dict(
        leaf_size=(IntParam, 3),
        n_neighbors=(IntParam, 5)
    ),
    LogisticRegression: dict(
        C=(FloatParam, 1)
    ),
    PassiveAggressiveClassifier: dict(
        C=(FloatParam, 1)
    ),
    RandomForestClassifier: dict(
        criterion=(CategoricalParam, ['gini', 'entropy']),
        max_features=(CategoricalParam, ['log2', 'sqrt', 'auto', None]),
        n_estimators=(IntParam, 1),
    ),
    RidgeClassifier: dict(
        alpha=(FloatParam, 1.0)
    ),
    SGDClassifier: dict(
        loss=(CategoricalParam, ['hinge', 'squared_hinge', 'epsilon_insensitive', 'squared_loss',
                                 'huber', 'modified_huber', 'perceptron', 'log',
                                 'squared_epsilon_insensitive']),
    ),
    SVC: dict(
        C=(FloatParam, 1),
        kernel=(CategoricalParam, ['rbf', 'linear', 'poly', 'sigmoid']),
    ),

}
