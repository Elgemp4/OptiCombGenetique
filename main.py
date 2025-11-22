from email.policy import default

import click

from crossover import two_point_crossover, one_point_crossover, uniform_crossover
from genetic import genetic
from initiation import initiate_randomly
from mutate import mutate_local_search_addition, mutate
from parser import read_file
from select_population import roulette_selection, select_replacement


@click.command()
@click.argument("file")
def enter_point(file):
    best = genetic(file=file,
            select_reproduction=roulette_selection,
            select_replacement=select_replacement,
            duration=15,
            crossover=uniform_crossover,
            initiate_population=initiate_randomly,
            mutate=mutate,
            reproduce_count=10000,
            select_count=2500,
            initial_count=100000)


    print(best.get_W())
    print(best.get_H())
    print(best.score)




if __name__ == '__main__':
    enter_point()


