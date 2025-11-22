from email.policy import default

import click

from crossover import two_point_crossover, one_point_crossover, uniform_crossover
from genetic import genetic
from initiation import initiate_randomly, initiate_algo
from mutate import mutate_local_search_addition, mutate, gradient_mutation, sparse_mutation
from parser import read_file
from select_population import roulette_selection, select_replacement
from utils import solutionIsFeasible


@click.command()
@click.argument("file")
def enter_point(file):
    best = genetic(file=file,
            select_reproduction=roulette_selection,
            select_replacement=select_replacement,
            duration=15,
            crossover=uniform_crossover,
            initiate_population=initiate_algo,
            mutate=sparse_mutation,
            reproduce_count=20,
            select_count=100,
            initial_count=100)


    X, m, n, rank, lower_w, upper_w, lower_h, upper_h = read_file(file)

    print(best.get_W())
    print(best.get_H())
    print(best.score)
    best.compute_score(X)
    print(best.score)

    print(solutionIsFeasible(best.get_W(), best.get_H(), rank, lower_w, upper_w, lower_h, upper_h))




if __name__ == '__main__':
    enter_point()


