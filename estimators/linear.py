from scipy.stats import expon
from sklearn.linear_model import ElasticNet, BayesianRidge, HuberRegressor, LinearRegression, \
    LogisticRegression, PassiveAggressiveClassifier, PassiveAggressiveRegressor, RidgeClassifier, TheilSenRegressor
from sklearn.preprocessing import RobustScaler

c_base = dict(C=expon(0.8, 0.5))
alpha_lambda_base = dict(
    alpha_1=expon(0, 1e-6),
    alpha_2=expon(0, 1e-6),
    lambda_1=expon(0, 1e-6),
    lambda_2=expon(0, 1e-6),
    normalize=[True],
)

ElasticNet.hyperparameters = dict(
    alpha=expon(0, 0.3),
    normalize=[True],
)


class RobustElasticNet(ElasticNet):
    def fit(self, X, y, check_input=True):
        self._scaler = RobustScaler()
        X_ = self._scaler.fit_transform(X)
        return super().fit(X_, y, check_input)

    def predict(self, X):
        X_ = self._scaler.transform(X)
        return super().predict(X_)


HuberRegressor.hyperparameters = dict(
    alpha=expon(0,0.0001),
    epsilon=expon(1, 0.35),
)

# ARDRegression.hyperparameters = dict(**alpha_lambda_base)
BayesianRidge.hyperparameters = dict(**alpha_lambda_base)

LinearRegression.hyperparameters = dict(normalize=[True])
LogisticRegression.hyperparameters = dict(**c_base)

PassiveAggressiveRegressor.hyperparameters = dict(**c_base, tol=[1e-3], max_iter=[1000])
PassiveAggressiveClassifier.hyperparameters = dict(**c_base, tol=[1e-3], max_iter=[1000])

RidgeClassifier.hyperparameters = dict(
    alpha=expon(0, 0.3),
    normalize=[True],
)
TheilSenRegressor.hyperparameters = dict(
    max_subpopulation=[1e3, 5e3, 1e4, 5e4],
)
