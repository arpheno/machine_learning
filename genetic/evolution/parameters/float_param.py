import random

from genetic.evolution.parameters.base import BaseParam


class FloatParam(BaseParam):
    def mutate(self):
        self.value += self.default * random.gauss(0, 2)