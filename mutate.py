import random

import numpy as np

from solution import Solution


def loop(min_value: int, max_value: int, value: int) -> int:
    return ((value - min_value) % (max_value - min_value)) + min_value

def mutate_local_search_addition(solution: Solution, lower_w: int, higher_w: int, lower_h: int, higher_h: int, X: np.ndarray) -> Solution:
    best_solution = solution
    while True:
        (best,has_changed) = get_best_neighbour(best_solution, lower_w, higher_w, lower_h, higher_h, X)
        best_solution = best
        if not has_changed:
            break

    return best_solution


def get_best_neighbour(solution: Solution, lower_w: int, higher_w: int, lower_h: int, higher_h: int, X: np.ndarray) -> tuple[Solution, bool]:
    height_w = len(solution.get_W())
    width_w = len(solution.get_W()[0])
    height_h = len(solution.get_H())
    width_h = len(solution.get_H()[0])
    best = solution
    has_changed = False

    for y in range(height_w):
        for x in range(width_w):

            new = solution.clone()
            new.change_w_at(y,x,loop(lower_w, higher_w, solution.get_W()[y, x] + random.randint(lower_w, higher_w)))

            if best is None or best.score > new.score:
                best = new
                has_changed = True

    for y in range(height_h):
        for x in range(width_h):
            new = solution.clone()
            new.change_h_at(y,x,loop(lower_h, higher_h, solution.get_H()[y, x] + random.randint(lower_h, higher_h)))

            if best is None or best.score > new.score:
                best = new
                has_changed = True

    return best, has_changed



def mutate(solution: Solution, lower_w: int, higher_w: int, lower_h: int, higher_h: int, X: np.ndarray):
    if random.randint(0,100) < 3:
        height = len(solution.get_W())
        width = len(solution.get_W()[0])
        solution.change_w_at(random.randint(0,height-1),random.randint(0,width-1), loop(lower_w, higher_w, random.randint(lower_w, higher_w)))

        height = len(solution.get_H())
        width = len(solution.get_H()[0])
        solution.change_h_at(random.randint(0,height-1), random.randint(0, width-1), loop(lower_h, higher_h, random.randint(lower_h, higher_h)))

    return solution