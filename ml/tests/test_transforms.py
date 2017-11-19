import pandas as pd

from ml.univariate import init_boxcox


def test_boxcox():
    boxcox, inv_boxcox = init_boxcox()
    original = pd.Series(dict(enumerate(range(100))), name='lol')
    result = boxcox(original)
    reversed = inv_boxcox(result)
    print(reversed)
    print(original)
    assert (reversed-original).apply(lambda x: pd.np.isclose(x,0)).all()
