import pytest
from sklearn import datasets

from estimators import classifiers
from ml.benchmark import score_model


@pytest.fixture
def X_iris():
    return datasets.load_iris().data[:100,:2]

@pytest.fixture
def y_iris():
    return datasets.load_iris().target[:100]

@pytest.mark.parametrize("model", classifiers)
def test_classifier(model, X_iris, y_iris):
    score_model(model, X_iris, y_iris, 'accuracy', 1)

