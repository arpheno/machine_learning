from functools import partial
from itertools import cycle
from unittest.mock import patch

from deap.tools import History, selTournament
from sklearn import datasets
from sklearn.svm import SVC

from evolution.core import sample_population, algorithm
from evolution.individual_base import Individual
from evolution.parameters import FloatParam, IntParam, CategoricalParam
from evolution.selection import harsh_winter
from evolution.utils import statsa
from halloffame import ObjectiveFunctionHallOfFame


def test_individual_mutate():
    a = Individual(dict(a=FloatParam(default=5), b=IntParam(default=1)))
    with patch('evolution.individual_base.random.random', lambda: 1):
        assert (~a).inner == a.inner
    with patch('evolution.individual_base.random.random', lambda: 0):
        assert (~a).inner != a.inner


def test_individual_mate():
    with patch('evolution.individual_base.random.random', lambda: 1):
        a = Individual(dict(a=FloatParam(default=5), b=IntParam(default=1)))
        b = Individual(dict(a=FloatParam(default=10), b=IntParam(default=2)))
        c, d = a + b
        assert c.inner == b.inner
        assert d.inner == a.inner
        assert c.inner['a'] != a.inner['a']
        assert d.inner['a'] != b.inner['a']
    with patch('evolution.individual_base.random.random', side_effect=cycle([0, 1])):
        a = Individual(dict(a=FloatParam(default=5), b=IntParam(default=1)))
        b = Individual(dict(a=FloatParam(default=10), b=IntParam(default=2)))
        c, d = a + b
        assert c.inner['a'] == a.inner['a']
        assert d.inner['a'] == b.inner['a']
def test_evolve():
    select=partial(harsh_winter,count=10)
    history = History()
    stats = statsa()
    hof = ObjectiveFunctionHallOfFame(maxsize=15)
    iris = datasets.load_iris()
    # Take the first two features. We could avoid this by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target
    params = dict(C=(FloatParam, 1), kernel=(CategoricalParam, ['rbf', 'linear', 'poly', 'sigmoid']))
    population=sample_population(SVC,params,X, y)
    algorithm(population,  select,  stats, history, hof)



