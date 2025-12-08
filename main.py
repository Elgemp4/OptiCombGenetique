import click
from pathlib import Path

from crossover import  uniform_crossover
from genetic import genetic
from initiation import  initiate_algo
from mutate import stochastic_hill_climbing, \
    gradient_mutation
from parser import read_file, write_output
from plot import plot_images_comparison
from select_population import roulette_selection, select_replacement
from utils import solutionIsFeasible


@click.command()
@click.argument("file")
def enter_point(file):

    best = genetic(file=file,
                   select_reproduction=roulette_selection,
                   select_replacement=select_replacement,
                   duration=10,
                   crossover=uniform_crossover,
                   initiate_population=initiate_algo,
                   mutate_search=gradient_mutation,
                   mutate_intensify=stochastic_hill_climbing,
                   reproduce_count=14,
                   select_count=200,
                   initial_count=300)


    X, m, n, rank, lower_w, upper_w, lower_h, upper_h = read_file(file)

    print(best.get_W())
    print(best.get_H())
    print(best.score)
    best.compute_score(X)
    print(best.score)

    print(solutionIsFeasible(best.get_W(), best.get_H(), rank, lower_w, upper_w, lower_h, upper_h))

    write_output(f"./output/{Path(file).stem}.out.txt", best)

    x_recontructed = best.get_W() @ best.get_H()
    plot_images_comparison(X, x_recontructed)


if __name__ == '__main__':
    enter_point()


