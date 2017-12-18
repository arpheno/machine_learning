import pandas as pd

from ml.decomposition import XPCA
from ml.univariate import init_boxcox


def test_boxcox():
    boxcox, inv_boxcox = init_boxcox()
    original = pd.Series(dict(enumerate(range(100))), name='lol')
    result = boxcox(original)
    reversed = inv_boxcox(result)
    print(reversed)
    print(original)
    assert (reversed - original).apply(lambda x: pd.np.isclose(x, 0)).all()


def test_xpca():
    data = pd.DataFrame([[1, 2, 15], [3, 4, -1, ], [30, 1, 22]], columns=['a', 'b', 'c'])
    transformed = XPCA().fit_transform(data, 'c')
    assert (transformed['c'] == pd.Series([15, -1, 22])).all()
