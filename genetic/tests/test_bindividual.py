from sklearn import datasets
from sklearn.svm import SVC

from estimator import Estimator
# import some data to play with
from evolution.parameters import FloatParam, CategoricalParam

iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target


def test_bindividual():
    params = dict(C=FloatParam(default=1), kernel=CategoricalParam(default=['rbf', 'linear', 'poly', 'sigmoid']))
    some = Estimator(estimator_cls=SVC, params=params, X=X, y=y)
    for _ in range(10):
        some = ~some
        print(some.inner)
        print(some.evaluate())
