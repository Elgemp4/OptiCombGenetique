import random
import time
from concurrent.futures.process import ProcessPoolExecutor
from typing import Callable

from parser import read_file
from plot import plot_score_evolution
from solution import Solution


def create_one_individu_process(args):
    """
    Process to create one solution
    :param args:
    :return:
    """
    (seed_val, init, m, n, rank, lower_w, upper_w, lower_h, upper_h, X) = args

    random.seed(seed_val)

    sol = init(X, m, n, rank, lower_w, upper_w, lower_h, upper_h)

    return sol


def process_children_process(args):
    """
    Process to mutate a child and compute its score
    :param args:
    :return:
    """
    (child, mutation_func, lower_w, upper_w, lower_h, upper_h, X) = args
    child.compute_score(X)
    # 1. Mutate
    # (No need to re-seed here as we are transforming existing data)
    child = mutation_func(child, lower_w, upper_w, lower_h, upper_h, X)

    # 2. Score
    child.compute_score(X)

    return child

def genetic(file, duration,
            select_reproduction: Callable[[list[Solution], int],list[Solution]],
            crossover: Callable[[Solution, Solution, int, int, int],tuple[Solution, Solution]],
            mutate_search,
            mutate_intensify,
            select_replacement,
            initiate_population,
            initial_count,
            reproduce_count,
            select_count):
    """
    The main loop for the genetic algorithm
    :param file:
    :param duration:
    :param select_reproduction:
    :param crossover:
    :param mutate_search:
    :param mutate_intensify:
    :param select_replacement:
    :param initiate_population:
    :param initial_count:
    :param reproduce_count:
    :param select_count:
    :return:
    """
    best_solution = None
    try:
        with ProcessPoolExecutor() as executor:
            X, m, n, rank, lower_w, upper_w, lower_h, upper_h = read_file(file)

            score_historic = []
            time_historic = []

            limit_sec = duration * 60
            start_time = time.perf_counter()

            method = ""
            last_score = 0

            init_tasks = []
            for i in range(initial_count):
                seed_val = time.time() + i
                init_tasks.append((seed_val, initiate_population, m, n, rank, lower_w, upper_w, lower_h, upper_h, X))

            population = list(executor.map(create_one_individu_process, init_tasks))

            while True:
                reproducing_population = select_reproduction(population, reproduce_count)

                shuffled_population = reproducing_population[:]

                random.shuffle(shuffled_population)
                evolution_tasks = []

                for i in range(len(shuffled_population) -1):
                    (child1, child2) = crossover(shuffled_population[i], shuffled_population[i+1], m, n, rank)
                    if random.random() < .5:
                        method= "search"
                        mutate = mutate_search
                    else:
                        method = "intensify"
                        mutate = mutate_intensify

                    evolution_tasks.append((child1, mutate, lower_w, upper_w, lower_h, upper_h, X))
                    evolution_tasks.append((child2, mutate, lower_w, upper_w, lower_h, upper_h, X))


                new_childs = list(executor.map(process_children_process, evolution_tasks))
                population.extend(new_childs)
                population = select_replacement(population, select_count)

                for pop in population:
                    if best_solution is None or pop.score < best_solution.score:
                        score_historic.append(pop.score)
                        time_historic.append(time.perf_counter() - start_time)
                        best_solution = pop
                        print(f"New best solution with cost : {best_solution.score}, ({last_score - best_solution.score}) - {method}")
                        last_score = best_solution.score
                        if(best_solution.score == 0):
                            break;

                if time.perf_counter() - start_time > limit_sec:
                    print("stop")
                    break

            plot_score_evolution(score_historic,time_historic)
            return best_solution
    except (KeyboardInterrupt):
        plot_score_evolution(score_historic,time_historic)
        return best_solution
