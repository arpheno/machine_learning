import logging

from sklearn.model_selection import cross_val_score

from .evolution.individual_base import Individual

logger = logging.getLogger(__name__)


class Estimator(Individual):
    def __init__(self, *args, X, y, estimator_cls, scoring, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimator_cls = estimator_cls
        self.X = X
        self.y = y
        self.scoring = scoring

    @property
    def objective(self):
        return sum(self.fitness.values)

    def evaluate(self):
        try:
            estimator = self.estimator_cls(**{attr: param.value for attr, param in self.inner.items()})
            result = (cross_val_score(estimator, self.X, self.y, cv=3, n_jobs=-1, scoring=self.scoring).mean(),)
        except Exception as e:
            print(f'Instantiating {self} failed. with {e}')
            result = (-1000,)
        self.fitness.values = result
