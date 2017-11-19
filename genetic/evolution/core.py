import random

from deap.tools import History
from termcolor import colored

from genetic.conf import cxpb, mutpb
from .halloffame import ObjectiveFunctionHallOfFame
from .utils import log_stuff, statsa


def algorithm(population, select, generations) -> ObjectiveFunctionHallOfFame:
    hof = ObjectiveFunctionHallOfFame(maxsize=15)
    history = History()
    stats = statsa()
    for g in range(1, generations + 1):
        for ind in population:
            ind.set_sigma((g / generations) * 1 / 3 + (generations - g) / generations * 1 / 10)
        log_stuff(g, history, hof, population, stats)
        offspring = breed(population)
        mutants = mutate(population)
        for ind in offspring + mutants:
            if not ind.fitness.valid:
                ind.evaluate()
                print('.', end='')
        population = select(set(offspring) | set(mutants) | population)
    return hof


def breed(population):
    offspring = []
    for index, (parent1, parent2) in enumerate(zip(sorted(population)[::2], sorted(population)[1::2])):
        if random.random() < cxpb * index / 4:
            child1, child2 = parent1 + parent2
            offspring.append(child1)
            offspring.append(child2)
    print(colored(len(offspring), 'green') + colored(f" children have been born.", 'blue'))
    return offspring


def mutate(population):
    mutants = []
    for index, individual in enumerate(sorted(population), 1):
        if random.random() < mutpb * index / 4:
            mutants.append(~individual)
    print(colored(len(mutants), 'green') + colored(f" individuals have mutated.", 'blue'))
    return mutants
