from pprint import pprint as print

import pandas
import seaborn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def linear_ensemble(regressors, X_train, y_train, scoring='neg_mean_absolute_error'):
    ensemble_results = pd.concat(
        [pd.Series(regressor.predict(X_train), name=regressor.__class__.__name__) for regressor in regressors], axis=1)
    g = sns.heatmap(ensemble_results.corr(), annot=True)
    g = LinearRegression()
    print(cross_val_score(g, X=ensemble_results, y=y_train, cv=5, n_jobs=4, scoring=scoring).mean())
    g.fit(X_train, y_train)
    return g