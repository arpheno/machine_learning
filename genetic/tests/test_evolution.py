from itertools import cycle
from unittest.mock import patch

from sklearn import datasets

from estimators.linear import LogisticRegression
from genetic.evolution.individual_base import Individual
from genetic.evolution.parameters.base import BaseParam
from genetic.parameter_optimization import tune_params_genetic


def test_individual_mutate():
    a = Individual(dict(a=BaseParam(5.3), b=BaseParam(3)))
    with patch('genetic.evolution.individual_base.random.random', lambda: 1):
        assert (~a).inner == a.inner
    with patch('genetic.evolution.individual_base.random.random', lambda: 0):
        assert (~a).inner != a.inner


def test_individual_mate():
    with patch('genetic.evolution.individual_base.random.random', lambda: 1):
        a = Individual(dict(a=BaseParam(5), b=BaseParam(1)))
        b = Individual(dict(a=BaseParam(10), b=BaseParam(2)))
        c, d = a + b
        assert c.inner == b.inner
        assert d.inner == a.inner
        assert c.inner['a'] != a.inner['a']
        assert d.inner['a'] != b.inner['a']
    with patch('genetic.evolution.individual_base.random.random', side_effect=cycle([0, 1])):
        a = Individual(dict(a=BaseParam(5), b=BaseParam(1)))
        b = Individual(dict(a=BaseParam(10), b=BaseParam(2)))
        c, d = a + b
        assert c.inner['a'] == a.inner['a']
        assert d.inner['a'] == b.inner['a']


def test_tune_params():
    iris = datasets.load_iris()
    # Take the first two features. We could avoid this by using a two-dim dataset
    X = iris.data[:100, :2]
    y = iris.target[:100]
    tune_params_genetic([LogisticRegression], X, y, 'neg_log_loss', 5)
