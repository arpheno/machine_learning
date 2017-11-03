import random

from genetic.evolution.parameters.base import BaseParam


class IntParam(BaseParam):
    def mutate(self):
        self.value += int(self.default * random.gauss(0, 2))