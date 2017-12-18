from sklearn.base import is_classifier, is_regressor, BaseEstimator

from .dummy import *
from .linear import *
from .gaussian import *
from .svm import *
from .ensemble import *


class TunableEstimator(BaseEstimator):
    hyperparameters = {}

    def __call__(self, *args, **kwargs) -> BaseEstimator:
        pass


regressors = [cls for cls in locals().values() if is_regressor(cls)]
classifiers = [cls for cls in locals().values() if is_classifier(cls)]
__all__ = locals().values()
