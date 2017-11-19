from sklearn.base import is_classifier, is_regressor, BaseEstimator

from .ensemble import *
from .gaussian import *
from .linear import *
from .svm import *


class TunableEstimator(BaseEstimator):
    hyperparameters = {}

    def __call__(self, *args, **kwargs) -> BaseEstimator:
        pass


regressors = [cls for cls in locals().values() if is_regressor(cls)]
classifiers = [cls for cls in locals().values() if is_classifier(cls)]
__all__ = locals().values()
