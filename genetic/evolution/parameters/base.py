from copy import copy


class BaseParam(object):
    def __deepcopy__(self, memodict={}):
        obj = copy(self)
        obj.value = self.value
        return obj

    def __init__(self, default=None, *args, **kwargs):
        self.default = default
        if self.default == 0:
            self.default = 1
        self.value = None
        self.mutate()

    def mutate(self):
        pass

    def __repr__(self):
        return f'{self.value}'

    def __eq__(self, other):
        return self.value == other.value


