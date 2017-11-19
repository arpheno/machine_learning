from scipy.stats import expon
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier

GaussianProcessRegressor.hyperparameters = {'alpha': expon(0, 1e-10)}
GaussianProcessClassifier.hyperparameters = dict()
