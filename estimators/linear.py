from scipy.stats import expon
from sklearn.linear_model import ElasticNet, BayesianRidge, HuberRegressor, LogisticRegression, RidgeClassifier, Lasso
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
Lasso.hyperparameters = dict(
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


class RobustLasso(Lasso):
    def fit(self, X, y, check_input=True):
        self._scaler = RobustScaler()
        X_ = self._scaler.fit_transform(X)
        return super().fit(X_, y, check_input)

    def predict(self, X):
        X_ = self._scaler.transform(X)
        return super().predict(X_)


HuberRegressor.hyperparameters = dict(
    alpha=expon(0, 0.0001),
    epsilon=expon(1, 0.35),
)

# ARDRegression.hyperparameters = dict(**alpha_lambda_base)
BayesianRidge.hyperparameters = dict(**alpha_lambda_base)

LogisticRegression.hyperparameters = dict(**c_base)

RidgeClassifier.hyperparameters = dict(
    alpha=expon(0, 0.3),
    normalize=[True],
)
