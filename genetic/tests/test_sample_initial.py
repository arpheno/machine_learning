from collections import defaultdict

from sklearn.svm import SVC

from genetic.parameter_optimization import sample_population


def test_sample_initial():

    sample_population(SVC, inner, X, y, popsize=10, scoring='accuracy')
