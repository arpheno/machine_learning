import random
from copy import copy


class BaseParam(object):
    def __init__(self, value, sigma=1 / 3, *args, **kwargs):
        self.kind = type(value)
        self.value = value
        self.sigma = sigma

    def mutate(self):
        if self.kind in [int, float]:
            self.value = self.kind(random.gauss(self.value, self.value * self.sigma))

    def __repr__(self):
        return f'{self.value}'

    def __eq__(self, other):
        return self.value == other.value

    def __deepcopy__(self, memodict={}):
        obj = copy(self)
        obj.value = self.value
        return obj
