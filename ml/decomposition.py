from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class XPCA(BaseEstimator):
    def __init__(self, scaler=StandardScaler, n_components=0.9999999):
        self.estimator = make_pipeline(scaler(), PCA(n_components=n_components))

    def fit(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)

    def transform(self,data, target, *args, **kwargs):
        _data = data.drop(target, axis=1)
        _y = data[target]
        _data[_data.columns] = self.estimator.transform(X=_data)
        _data[target] = _y
        return _data

    def fit_transform(self, data, target, *args, **kwargs):
        _data = data.drop(target, axis=1)
        _y = data[target]
        _data[_data.columns] = self.estimator.fit_transform(X=_data)
        _data[target] = _y
        return _data


