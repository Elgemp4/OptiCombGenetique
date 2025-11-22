import random
import time
from typing import Callable

from parser import read_file
from solution import Solution


def genetic(file, duration,
            select_reproduction: Callable[[list[Solution], int],list[Solution]],
            crossover: Callable[[Solution, Solution, int, int, int],tuple[Solution, Solution]],
            mutate,
            select_replacement,
            initiate_population,
            initial_count,
            reproduce_count,
            select_count):
    X, m, n, rank, lower_w, upper_w, lower_h, upper_h = read_file(file)
    thousand_iteration = 0
    iteration = 0

    historic = []

    population = initiate_population(X, m, n, rank, lower_w, upper_w, lower_h, upper_h, initial_count)

    limit_sec = duration * 60
    start_time = time.perf_counter()

    best_solution = None
    try:
        while True:
            reproducing_population = select_reproduction(population, reproduce_count)

            shuffled_population = reproducing_population[:]

            random.shuffle(shuffled_population)

            for i in range(len(shuffled_population) -1):
                (child1, child2) = crossover(shuffled_population[i], shuffled_population[i+1], m, n, rank)
                child1.compute_score(X)
                child2.compute_score(X)
                child1 = mutate(child1, lower_w, upper_w, lower_h, upper_h, X)
                child2 = mutate(child2, lower_w, upper_w, lower_h, upper_h, X)
                population.append(child1)
                population.append(child2)
            population = select_replacement(population, select_count)

            for pop in population:
                if best_solution is None or pop.score < best_solution.score:
                    historic.append(pop)
                    best_solution = pop
                    print(f"New best solution with cost : {0}", best_solution.score)

            iteration = iteration + 1
            if iteration % 1000 == 0:
                thousand_iteration = thousand_iteration + 1
                iteration = 0
                print("hundred iterations")
                if time.perf_counter() - start_time > limit_sec:
                    print("stop")
                    break
        return best_solution
    except (KeyboardInterrupt):
        return best_solution