# noinspection PyUnresolvedReferences
from functools import partial
# noinspection PyUnresolvedReferences
from itertools import chain
from pprint import pprint as print
from typing import List

import numpy as np
import pandas as pd
import sklearn

print(f'numpy {np.__version__} pandas {pd.__version__} sklearn {sklearn.__version__}')

# noinspection PyUnresolvedReferences
import matplotlib as mpl
# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
import seaborn as sns
# noinspection PyUnresolvedReferences
from jupyterthemes import jtplot
# noinspection PyUnresolvedReferences
from scipy.stats import norm, skew
# noinspection PyUnresolvedReferences
from scipy import stats

from sklearn.model_selection import GridSearchCV

# noinspection PyUnresolvedReferences
from ml.data_cleaning import *
from ml.univariate import *
from ml.benchmark import *


def tune_params(model_classes: List[type], X_train, y_train, scoring):
    """ Tunes parameters with gridsearch, really just a convenience wrapper"""
    best_models = []
    for model in model_classes.items():
        print(f'Tuning {model.__name__} with {hypers}...')
        clf = GridSearchCV(model(), model.hypers, scoring=scoring, n_jobs=-1, cv=5)
        clf.fit(X_train, y_train)
        best_models.append(clf.best_estimator_)
        print(clf.best_score_, clf.best_params_)
    return best_models


def neg_log_mean_squared_error_inv_boxcox(y, y_pred):
    y = np.log(inv_boxcox(y))
    y_pred = np.log(inv_boxcox(pd.Series(y_pred, name='SalePrice', index=y.index)))
    error = -np.sqrt(np.mean((y - y_pred) ** 2))
    return error


__all__ = list(globals().keys())
