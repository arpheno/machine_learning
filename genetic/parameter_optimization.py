from functools import partial

from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import cross_val_score

from genetic.estimator import Estimator
from genetic.evolution.core import algorithm
from genetic.evolution.parameters.base import BaseParam
from genetic.evolution.selection import harsh_winter
from genetic.evolution.utils import timestuff


def sample_population(estimator_cls, params, X, y, popsize=25, scoring='accuracy'):
    population = set()
    p = iter(ParameterSampler(param_distributions=params, n_iter=100))
    while len(population) < popsize:
        try:
            inner = {attr: BaseParam(param) for attr, param in (next(p)).items()}
            individual = Estimator(estimator_cls=estimator_cls, inner=inner, X=X, y=y, scoring=scoring)
            individual.evaluate()
            population.add(individual)
        except:
            print(f"Failed to draft initial {inner}")
    return population


def tune_params_genetic(clss, X, y, scoring, first_gen=30):
    best_estimators = []
    for estimator_cls in clss:
        with timestuff("Base evaluation"):
            default_score = cross_val_score(estimator_cls(), X=X, y=y, scoring=scoring, n_jobs=4).mean()
            print(f"Default configuration scores {default_score}")

        population = sample_population(estimator_cls, estimator_cls.hyperparameters, X, y, scoring=scoring)
        select = partial(harsh_winter, count=10)
        hof = algorithm(population, select, first_gen)
        print("Done with first stage")
        best_estimators.append(hof.best_estimator)
    return best_estimators
