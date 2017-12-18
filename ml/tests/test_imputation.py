import random
from unittest.mock import patch, Mock

import pandas as pd
import pytest
from numpy import nan
from sklearn.linear_model import LinearRegression

from ml.imputation import impute_linear


@pytest.fixture()
def data():
    numerical = pd.DataFrame(pd.np.random.randint(5, size=(50, 4)), columns=['a', 'b', 'c', 'd'])
    categorical = pd.Series(['a', 'b', 'c', 'd', 'e'] * 10, name='name')
    target = pd.Series(list(range(30)) + [10] * 20, name='target').replace(10, nan)
    return pd.concat([numerical, categorical, target], axis=1)


def test_impute_linear(data):
    r=lambda *_,**__:pd.Series([random.randint(1, 5), 0, LinearRegression()], index=['Mean', 'Std', 'Model'])
    with patch('ml.benchmark.score_model',r):
        impute_linear(data, 'target', ['a', 'b', 'c', 'd'], ['name'], scoring='neg_mean_squared_error')
