from typing import Tuple, Callable

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
import warnings


def init_boxcox() -> Tuple[Callable, Callable]:
    from scipy.special import inv_boxcox1p as inverse
    from scipy.stats import boxcox as forward
    lambdas = dict()

    def myforward(raw_data: pd.Series,lambdas = lambdas):
        with warnings.catch_warnings():
            raw_data = raw_data.copy()
            data = raw_data.dropna()
            transformed, lambdas[data.name] = forward(1 + data)
            if abs(lambdas[data.name]) < 0.01:
                lambdas[data.name] = 0
                transformed = np.log1p(data)
            transformed = pd.Series(transformed, index=data.index)
            after, before = skew(transformed), skew(data)
            if abs(after) < abs(before):
                print(f'Reduced skew from {before:.4f} to {after:.4f} for {raw_data.name}')
                raw_data.update(transformed)
            else:
                print(f'Failed to reduce skew with BoxCox for {raw_data.name}')
            return raw_data

    def myinverse(raw_data: pd.Series):
        raw_data = raw_data.copy()
        data = raw_data.dropna()
        if lambdas[data.name] == 0:
            transformed = np.expm1(data)
        else:
            transformed = inverse(data, lambdas[data.name])
        transformed = pd.Series(transformed, index=data.index)
        before, after = skew(transformed), skew(data)
        if abs(after) < abs(before):
            raw_data.update(transformed)
        return raw_data

    return myforward, myinverse


boxcox, inv_boxcox = init_boxcox()


def skewednessplots(data, variable):
    fig, ax = plt.subplots(1, 4, figsize=(35, 5))
    data = data[data[variable].notnull()].copy()

    sns.distplot(data[variable], fit=norm, ax=ax[0], fit_kws=dict(color='white'))

    (mu, sigma) = norm.fit(data[variable])
    skew_ = skew(data[variable])
    ax[0].legend([fr'$\beta=$ {skew_:.2f} $\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f}'], loc='best')
    ax[0].set_title(f'{variable} distribution')
    stats.probplot(data[variable], plot=ax[1])

    transformed = boxcox(data[variable])
    sns.distplot(transformed, fit=norm, ax=ax[2], fit_kws=dict(color='white'))
    (mu, sigma) = norm.fit(transformed)
    skew_ = skew(transformed)
    ax[2].legend([fr'$\beta=$ {skew_:.2f} $\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f}'], loc='best')
    ax[2].set_title(f'Transformed {variable} distribution')
    stats.probplot(transformed, plot=ax[3])


def skewedness_numerical(data):
    """Check the skew of all numerical features"""
    numerical = data.select_dtypes(include=[np.number])
    skewed_feats = data[numerical.columns].apply(lambda x: skew(x)).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feats})
    return skewness
