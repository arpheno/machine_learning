from functools import partial

from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score

from genetic.estimator import Estimator
from genetic.estimator_parameters import params
from genetic.evolution.core import algorithm
from genetic.evolution.selection import harsh_winter
from genetic.evolution.utils import timestuff


def sample_population(estimator_cls, params, X, y, popsize=10,scoring='accuracy'):
    population = set()
    while len(population) < popsize:
        population.add(Estimator(estimator_cls=estimator_cls, params=params, X=X, y=y,scoring=scoring))
        print(len(population))
    return population


def tune_params_genetic(clss, X, y,scoring='accuracy',first_gen=10,second_gen=20):
    best_estimators = []
    for estimator_cls in clss:
        hyper_parameters = params[estimator_cls]
        with timestuff("Base evaluation"):
            default_score = cross_val_score(estimator_cls(), X=X, y=y,scoring=scoring,n_jobs=4).mean()
            print(f"Default configuration scores {default_score}")

        population = sample_population(estimator_cls, hyper_parameters, X, y,scoring=scoring)
        select = partial(harsh_winter, count=10)
        hof = algorithm(population, select, first_gen)
        print("Done with first stage")
        population = {ind for ind in hof.inner if ind.fitness.valid}
        for ind in population:
            ind.set_fine()
        hof = algorithm(population, select, second_gen)
        return best_estimators.append(hof.best_estimator)
    return best_estimators


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target

    best = tune_params_genetic([AdaBoostClassifier], X=X, y=y,scoring='neg_log_loss')
