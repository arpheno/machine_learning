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
    for g in range(generations):
        log_stuff(g, history, hof, population, stats)
        print(colored(f"It's breeding season, we're expecting new members of the tribe...", 'blue'))
        offspring = breed(population)
        print(colored(f"Radiation and toxic waste are causing mutations in the population...", 'blue'))
        mutants = mutate(population)
        print(colored(f"Summer is here, evaluating our new arrivals...", 'blue'))

        for ind in offspring + mutants:
            if not ind.fitness.valid:
                ind.evaluate()
                print('.', end='')
        survivors = select(set(offspring) | set(mutants) | population)
        population = survivors

    return hof


def breed(population):
    offspring = []
    while len(offspring) < len(population) * cxpb:
        parent1, parent2 = random.sample(population, 2)
        child1, child2 = parent1 + parent2
        offspring.append(child1)
        offspring.append(child2)
    print(colored(len(offspring), 'green') + colored(f" children have been born.", 'blue'))
    return offspring


def mutate(population):
    mutants = []
    for individual in population:
        if random.random() < mutpb:
            mutants.append(~individual)
    print(colored(len(mutants), 'green') + colored(f" individuals have mutated.", 'blue'))
    return mutants
