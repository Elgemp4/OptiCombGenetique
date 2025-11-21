import time
from parser import read_file


def genetic(file, duration, select_reproduction, cross, mutate, select_replacement, initiate_population):
    X, m, n, rank, lower_w, upper_w, lower_h, upper_h = read_file(file)
    thousand_iteration = 0
    iteration = 0
    population = initiate_population(rank, lower_w, lower_h, upper_w, upper_h)
    limit_sec = duration * 60
    start_time = time.perf_counter()

    best_solution = None

    while True:
        reproducing_population = select_reproduction(population)
        new_childrens = cross(reproducing_population)

        new_childrens = mutate(new_childrens)
        population.append(new_childrens)

        population = select_replacement(population)

        iteration = iteration + 1

        if iteration % 10000 == 0:
            thousand_iteration = thousand_iteration + 1
            iteration = 0
            if time.perf_counter() - start_time < limit_sec:
                break