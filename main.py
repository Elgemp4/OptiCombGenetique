import click
from pathlib import Path

from crossover import  uniform_crossover
from genetic import genetic
from initiation import  generate_smart_solution
from mutate import stochastic_hill_climbing, \
    nnls_mutation
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
                   initiate_population=generate_smart_solution,
                   mutate_search=nnls_mutation,
                   mutate_intensify=stochastic_hill_climbing,
                   reproduce_count=200,
                   select_count=250,
                   initial_count=300)


    X, m, n, rank, lower_w, upper_w, lower_h, upper_h = read_file(file)

    best.compute_score(X)
    output = f"./output/{Path(file).stem}.out.txt"
    print("================= Best solution found ! ========================")
    print(f"Score : {best.score}")
    print(f"Is feasible : {solutionIsFeasible(best.get_W(), best.get_H(), rank, lower_w, upper_w, lower_h, upper_h)}")
    print(f"Output file : {output}")
    write_output(output, best)

    x_recontructed = best.get_W() @ best.get_H()
    plot_images_comparison(X, x_recontructed)


if __name__ == '__main__':
    enter_point()


