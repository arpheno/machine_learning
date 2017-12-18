import pandas as pd
import pytest
from sklearn import datasets

from estimators import regressors
from ml.benchmark import score_model

rows = [5 + x * 10 for x in range(35)]


@pytest.fixture
def X_boston():
    return datasets.load_boston().data[rows, :2]


@pytest.fixture
def y_boston():
    return pd.Series(datasets.load_boston().target[rows], name='target')


@pytest.mark.parametrize("model", regressors)
def test_regressor(model, X_boston, y_boston):
    score_model(model, X_boston, y_boston, 'neg_mean_squared_error', 5)
