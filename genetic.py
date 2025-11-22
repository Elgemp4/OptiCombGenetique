import random
import time
from typing import Callable

from parser import read_file
from solution import Solution


def genetic(file, duration,
            select_reproduction: Callable[[list[Solution]],list[Solution]],
            crossover: Callable[[Solution, Solution],tuple[Solution, Solution]],
            mutate,
            select_replacement,
            initiate_population):
    X, m, n, rank, lower_w, upper_w, lower_h, upper_h = read_file(file)
    thousand_iteration = 0
    iteration = 0
    population = initiate_population(rank, lower_w, lower_h, upper_w, upper_h)
    limit_sec = duration * 60
    start_time = time.perf_counter()

    best_solution = None

    while True:
        reproducing_population = select_reproduction(population)

        shuffled_population = reproducing_population[:]

        random.shuffle(shuffled_population)

        for i in range(len(shuffled_population) -1):
            (child1, child2) = crossover(shuffled_population[i], shuffled_population[i+1])
            child1.compute_score(X)
            child2.compute_score(X)
            population.append(child1)
            population.append(child2)


        new_childrens = mutate(new_childrens)
        population.append(new_childrens)

        population = select_replacement(population)

        iteration = iteration + 1

        if iteration % 10000 == 0:
            thousand_iteration = thousand_iteration + 1
            iteration = 0
            if time.perf_counter() - start_time < limit_sec:
                break