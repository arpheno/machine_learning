import random

from genetic.evolution.parameters.base import BaseParam


class CategoricalParam(BaseParam):
    def mutate(self):
        if not self.value:
            self.value = random.choice(self.default)