import random
from copy import deepcopy, copy

import names
from deap.base import Fitness

from genetic.conf import partitions


class FitnessMax(Fitness):
    weights = tuple([1 for _ in range(partitions)])


class Individual:
    mate = lambda *x: x
    mutate = lambda x: x

    @property
    def objective(self):
        return 0

    def set_sigma(self, sigma):
        for param in self.inner.values():
            param.sigma = sigma

    def __init__(self, inner, *args, **kwargs):
        self.name = names.get_full_name()
        self.fitness = FitnessMax()
        self.inner = inner

    def __deepcopy__(self, memodict={}):
        obj = copy(self)
        obj.fitness = deepcopy(self.fitness)
        obj.inner = deepcopy(self.inner)
        obj.name = names.get_full_name()
        return obj

    def __add__(self, other):
        child1, child2 = deepcopy(self), deepcopy(other)
        splitter = [True if random.random() < 0.5 else False for _ in range(len(child1))]
        _ = dict(x if c else y for (c, x, y) in zip(splitter, child1.items(), child2.items()))
        child2.inner = dict(x if not c else y for (c, x, y) in zip(splitter, child1.items(), child2.items()))
        child1.inner = _
        del child1.fitness.values
        del child2.fitness.values
        return child1, child2

    def __invert__(self):
        mutant = deepcopy(self)
        for attr in mutant.inner:
            if random.random() < 0.5:
                mutant.inner[attr].mutate()
        del mutant.fitness.values
        return mutant

    def __lt__(self, other):
        return self.objective < other.objective

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(tuple(hash((attr, param.value)) for attr, param in self.inner.items()))

    def __len__(self):
        return len(self.inner)

    def items(self):
        return self.inner.items()

    def __repr__(self):
        return f"{list(self.fitness.values)} {self.objective} {self.name} {self.inner}"
