import numpy as np

from solution import Solution


def loop(min_value: int, max_value: int, value: int) -> int:
    return ((value - min_value) % (max_value - min_value)) + min_value

def mutate_local_search_addition(solution: Solution, lower_w: int, higher_w: int, lower_h: int, higher_h: int, X: np.ndarray) -> Solution:
    while True:
        (best,has_changed) = get_best_neighbour(solution, lower_w, higher_w, lower_h, higher_h, X)
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
            new_w = solution.get_W()[:]
            new_h = solution.get_H()[:]
            new_w[y, x] = loop(lower_w, higher_w, new_w[y, x] + 1)

            new_sol = Solution(new_w, new_h)
            new_sol.compute_score(X)

            if best is not None and best.score > new_sol.score:
                best = new_sol
                has_changed = True

    for y in range(height_h):
        for x in range(width_h):
            new_w = solution.get_W()[:]
            new_h = solution.get_H()[:]
            new_h[y, x] = loop(lower_h, higher_h, new_h[y, x] + 1)

            new_sol = Solution(new_w, new_h)
            new_sol.compute_score(X)

            if best.score > new_sol.score:
                best = new_sol
                has_changed = True

    return best, has_changed