import random
from operator import attrgetter
from typing import Iterable, Set

from termcolor import colored

from .individual_base import Individual


def harsh_winter(population: Set[Individual], count: int) -> Set[Individual]:
    """ Selects `popsize` many individuals from the current population."""
    elitist_count = int(count * 0.3)
    elites = select_elites(population, elitist_count)
    survivors = elites
    difference = population - survivors
    if difference:
        rest = set(random.sample(difference, count - len(survivors)))
        population = survivors | rest
    else:
        rest = set()
        population = survivors
    log_stuff(elites, rest, set())
    return population


def select_elites(individuals: Iterable[Individual], count: int):
    elites = set(sorted(individuals, key=attrgetter('objective'), reverse=True)[:count])
    return elites


def log_stuff(elites, rest: Set, specialists):
    print(colored("\n\nWinter has come, weeding out the unworthy.", 'blue'))
    print(f"{len(elites)} Elites will survive, they're currently the strongest:")
    for elite in sorted(elites, key=attrgetter('objective'), reverse=True):
        print(elite)
    print(f"{len(specialists)} Specialists will survive, they're the best in their domain:")
    for specialist in specialists:
        print(specialist)
    print(f"Some other have fought their way through:")
    for r in random.sample(rest, len(rest) // 5):
        print(r)
    print(colored('...', 'grey'))
