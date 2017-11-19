import random
from typing import Set

from termcolor import colored

from genetic.evolution.individual_base import Individual


def harsh_winter(population: Set[Individual], count: int) -> Set[Individual]:
    """ Selects `popsize` many individuals from the current population."""
    elitist_count = int(count * 0.3)
    new_population = set(sorted(population)[::-1][:elitist_count])
    difference = population - new_population
    new_population |= set(random.sample(difference, count - elitist_count))
    # tournament(count, difference, new_population)
    print(colored("\n\nWinter has come, weeding out the unworthy.", 'blue'))
    for ind in sorted(new_population)[::-1][:5]:
        print(ind)
    return new_population


def tournament(count, difference, new_population):
    while len(new_population) < count:
        tournament_winner = sorted(random.sample(difference, 3))[-1]
        difference.remove(tournament_winner)
        new_population.add(tournament_winner)
